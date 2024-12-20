from flask import Flask, request, jsonify
from preprocess import processing_text_for_query, processing_text_for_db_rerank, processing_text_for_query_rerank
import pandas as pd
import numpy as np
import torch
from pymongo import MongoClient
from flask_cors import CORS
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
from pyngrok import ngrok
import re
import joblib
import requests
import json
app = Flask(__name__)
CORS(app)
model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype="auto",
    trust_remote_code=True,
)

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# corrector = pipeline("text2text-generation",
#                      model="bmd1905/vietnamese-correction")


# def correction_text(texts):
#     return corrector(texts, max_length=512)

# Tải lại mô hình
loaded_model = joblib.load("test_model/logistic_regression_model.pkl")
# Tải lại encoder
loaded_encoder = joblib.load("test_model/label_encoder.pkl")


def jaccard_similarity(query, document):
    query_words = set(re.split(r'\s+', query.lower().strip()))
    doc_words = set(re.split(r'\s+', document.lower().strip()))
    intersection = query_words.intersection(doc_words)
    union = query_words.union(doc_words)
    return len(intersection) / len(union)


async def embeddings_model(question):
   # Tokenize và chuyển đổi câu văn bản thành tensor
    tokens = tokenizer(question, return_tensors='pt',
                       padding=True, truncation=True)
    # Sử dụng model để mã hóa câu văn bản thành vector
    with torch.no_grad():
        output = phobert(**tokens)
    # Lấy vector biểu diễn từ outputs của model
    # Trung bình các vector token
    return output.last_hidden_state.mean(dim=1).numpy()


# Kết nối tới MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot"]
collection = db["q&a"]
collection_feedback = db["feedback"]

cursor = collection.find(
    {}, {"question": 1, "answer": 1, "processed_question": 1, "vector_embeddings": 1, "tag": 1, "_id": 0})

# Chuyển dữ liệu từ MongoDB thành DataFrame
dataFrame = pd.DataFrame(list(cursor))
dataFrame['vector_embeddings'] = dataFrame['vector_embeddings'].apply(
    lambda x: np.array(json.loads(x)))
questions_vector = dataFrame['vector_embeddings']
processed_questions = dataFrame['processed_question']
questions = dataFrame['question']
answers = dataFrame['answer']
tags = dataFrame['tag']


async def get_response(user_query):
    query = user_query
    processed_query = processing_text_for_query(user_query)
    query_vector = await embeddings_model(processed_query)
    # Sử dụng mô hình đã tải để dự đoán
    predicted_label = loaded_model.predict(query_vector)
    predicted_label_string = loaded_encoder.inverse_transform(predicted_label)

    cosine_similarities = [cosine_similarity(
        query_vector, qv).flatten() for qv in questions_vector]
    jaccard_similarities = [jaccard_similarity(
        processed_query, q) for q in dataFrame['processed_question']]

    alpha = 0.6
    beta = 0.4
    jaccard_array = np.array(jaccard_similarities).reshape(
        len(dataFrame['processed_question']), 1)

    combined_scores = alpha * \
        np.array(cosine_similarities) + beta * jaccard_array
    sorted_indices = np.argsort(combined_scores, axis=0)

    documents = []
    pos = []
    n = 10
    largest_indices = sorted_indices[::-1][:n]
    for i in largest_indices:
        documents.append(processing_text_for_db_rerank(questions[int(i)]))
        pos.append([int(i)])
    query_rerank = processing_text_for_query_rerank(user_query)

    result = model.rerank(
        query_rerank,
        documents,
        max_query_length=512,
        max_length=1024,
        top_n=3
    )
    print(query_rerank)
    print(documents)
    print(result)
    if result[0]['relevance_score'] > 0.67:
        return {
            "response": answers[pos[result[0]['index']][0]],
            "question": questions[pos[result[0]['index']][0]],
            "tag": tags[pos[result[0]['index']][0]],
            "tag_predict": predicted_label_string[0],
            "query": query,
            "result": result,

        }
    else:
        return {
            "response": "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.",
            "question": questions[pos[result[0]['index']][0]],
            "tag":  tags[pos[result[0]['index']][0]],
            "tag_predict": predicted_label_string[0],
            "query": query,
            "result": None,

        }


global previous_questions
previous_questions = []


async def send_to_google_sheet(response_data):
    try:
        # Lấy dữ liệu từ response_data
        response = response_data["response"]
        tag = response_data['tag']
        tag_predict = response_data['tag_predict']
        query = response_data['query']
        question = response_data['question']

        # Dữ liệu cần gửi
        data = {
            "query": query,
            "question": question,
            "answer": response,
            "tag": tag,
            "tag_predict": tag_predict,
        }

        # Gửi POST request đến SheetBest API
        url = "https://api.sheetbest.com/sheets/740ee395-c884-4476-85d0-27b094b0b5c4"
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=data, headers=headers)
        # Kiểm tra kết quả trả về
        if response.status_code == 200:
            print("Dữ liệu đã được gửi thành công!")
        else:
            print(
                f"Lỗi khi gửi dữ liệu: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Lỗi: {e}")


async def handle_user_question(question):
    try:
        response_data = await get_response(question)
        response = response_data["response"]
        result = response_data["result"]
        tag = response_data['tag']
        tag_predict = response_data['tag_predict']
        query = response_data['query']

        # Kiểm tra câu đầu tiên với ngưỡng 0.6
        if result and result[0]['relevance_score'] > 0.67:
            await send_to_google_sheet(response_data)
            previous_questions.append(result[0]['document'])
            return {
                "response": response,
                "tag": tag,
                "tag_predict": tag_predict,
                "query": query,
            }
        else:
            for i in range(1, min(2, len(previous_questions) + 1)):
                last_question = previous_questions[-i]
                combined_question = last_question + " " + question
                combined_response_data = await get_response(combined_question)
                combined_result = combined_response_data["result"]

                if combined_result and combined_result[0]['relevance_score'] > 0.84:
                    await send_to_google_sheet(combined_response_data)
                    return {
                        "response": combined_response_data["response"],
                        "tag": combined_response_data["tag"],
                        "tag_predict": combined_response_data["tag_predict"],
                        "query": combined_response_data["query"],
                    }

            await send_to_google_sheet(response_data)
        # Nếu không tìm được phản hồi hợp lệ
        return {
            "response": "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.",
            "tag": None,
            "tag_predict": None,
            "query": query,
        }

    except Exception:
        return {
            "response": "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.",
            "tag": None,
            "tag_predict": None,
            "query": question,
        }


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, (np.integer, int)):  # Chuyển kiểu numpy.int64
        return int(data)
    elif isinstance(data, (np.floating, float)):  # Chuyển kiểu numpy.float64
        return float(data)
    else:
        return data


@app.route('/api/chatbot', methods=['GET'])
async def chatbot_response():
    # Lấy tham số query từ URL
    query = request.args.get('q')
    response_data = await handle_user_question(query)
    # Chuyển đổi dữ liệu
    serializable_data = convert_to_serializable(response_data)
    print(serializable_data)
    return jsonify(serializable_data)


@app.route('/api/save_feedback', methods=['POST'])
def save_feedback():
    data = request.json
    email = data.get('email')
    message = data.get('message')

    if email and message:
        feedback_data = {
            "email": email,
            "message": message
        }
        # Lưu dữ liệu vào MongoDB
        result = collection_feedback.insert_one(feedback_data)
        return jsonify({"success": True, "feedback_id": str(result.inserted_id)}), 201
    else:
        return jsonify({"error": "Invalid data"}), 400


# NGROK_TOKEN = "2q1klyCrZwPfd2cQiQX7sgapgWU_7PJ2FQmNMHsDBNpkqEK3h"
# ngrok.set_auth_token(NGROK_TOKEN)
# ngrok_tunnel = ngrok.connect(5000)
# print("Public URL:", ngrok_tunnel.public_url)
# nest_asyncio.apply()

if __name__ == '__main__':
    app.run(debug=False, port=5000)
