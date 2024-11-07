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

cursor = collection.find(
    {}, {"question": 1, "answer": 1, "processed_question": 1, "vector_embeddings": 1, "_id": 0})

# Chuyển dữ liệu từ MongoDB thành DataFrame
dataFrame = pd.DataFrame(list(cursor))
dataFrame['vector_embeddings'] = dataFrame['vector_embeddings'].apply(
    lambda x: np.array(json.loads(x)))
questions_vector = dataFrame['vector_embeddings']
processed_questions = dataFrame['processed_question']
questions = dataFrame['question']
answers = dataFrame['answer']


async def get_response(user_query):
    processed_query = processing_text_for_query(user_query)
    query_vector = await embeddings_model(processed_query)

    cosine_similarities = [cosine_similarity(
        query_vector, qv).flatten() for qv in questions_vector]
    sorted_indices = np.argsort(cosine_similarities, axis=0)

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
    if result[0]['relevance_score'] > 0.7:
        return answers[pos[result[0]['index']][0]]
    else:
        return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."


global previous_questions
previous_questions = []


async def handle_user_question(question):
    try:
        response = await get_response(question)
        if response == "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.":
            for i in range(1, min(3, len(previous_questions) + 1)):
                last_question = previous_questions[-i]
                combined_question = last_question + " " + question
                response = await get_response(combined_question)
                if response != "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.":
                    return response

        # Thêm câu hỏi hiện tại vào danh sách câu hỏi trước đó
        previous_questions.append(question)
        return response
    except (Exception):
        return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."


@app.route('/api/chatbot', methods=['GET'])
async def chatbot_response():
    # Lấy tham số query từ URL
    query = request.args.get('q')
    response = await handle_user_question(query)
    return jsonify(response)

NGROK_TOKEN = "2KXuaD0CZC1wD6xl0aycvptytsm_dVtVE8o12y5JeGw55HoQ"
ngrok.set_auth_token(NGROK_TOKEN)
ngrok_tunnel = ngrok.connect(5000)
print("Public URL:", ngrok_tunnel.public_url)
nest_asyncio.apply()

if __name__ == '__main__':
    app.run(debug=False, port=5000)
