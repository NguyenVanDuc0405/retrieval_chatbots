from preprocess import processing_text_for_query, processing_text_for_db_rerank, processing_text_for_query_rerank
import pandas as pd
import numpy as np
import torch
from pymongo import MongoClient

from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import requests
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
headers = {"Authorization": "Bearer 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"}

model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype="auto",
    trust_remote_code=True,
)

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


def jaccard_similarity(query, document):
    query_words = set(re.split(r'\s+', query.lower().strip()))
    doc_words = set(re.split(r'\s+', document.lower().strip()))
    intersection = query_words.intersection(doc_words)
    union = query_words.union(doc_words)
    return len(intersection) / len(union)


def embeddings_model(question):
   # Tokenize và chuyển đổi câu văn bản thành tensor
    tokens = tokenizer(question, return_tensors='pt',
                       padding=True, truncation=True)
    # Sử dụng model để mã hóa câu văn bản thành vector
    with torch.no_grad():
        output = phobert(**tokens)
    # Lấy vector biểu diễn từ outputs của model
    # Trung bình các vector token
    return output.last_hidden_state.mean(dim=1).numpy()


def reranker(query_rerank, documents):
    payload = {
        "inputs": {
            "source_sentence": query_rerank,
            "sentences": documents
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


corrector = pipeline("text2text-generation",
                     model="bmd1905/vietnamese-correction")


def correction_model(texts):
    return corrector(texts, max_length=512)


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


def get_response(user_query):
    processed_query = processing_text_for_query(user_query)
    query_vector = embeddings_model(processed_query)

    # cosine_similarities = [cosine_similarity(
    #     query_vector, qv).flatten() for qv in questions_vector]
    # jaccard_similarities = [jaccard_similarity(
    #     processed_query, q) for q in dataFrame['processed_question']]

    # # Kết hợp kết quả từ hai độ tương đồng
    # alpha = 0.7
    # beta = 0.3
    # # Chuyển đổi jaccard_similarities thành mảng một chiều
    # jaccard_array = np.array(jaccard_similarities).reshape(
    #     len(dataFrame['processed_question']), 1)

    # # Kết hợp điểm số từ hai độ tương đồng
    # combined_scores = alpha * \
    #     np.array(cosine_similarities) + beta * jaccard_array
    # sorted_indices = np.argsort(combined_scores, axis=0)

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
    if result[0]['relevance_score'] > 0.67:
        return answers[pos[result[0]['index']][0]]
    else:
        return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."

    # result = model.rerank(
    #     query_rerank,
    #     documents,
    #     max_query_length=512,
    #     max_length=1024,
    #     top_n=3
    # )
    # print(query_rerank)
    # print(documents)
    # print(result)

    # result = reranker(query_rerank, documents)
    # # In ra câu hỏi và tài liệu
    # print(result)
    # print(query_rerank)
    # print(documents)

    # # Xử lý kết quả từ API
    # # Kiểm tra xem result có phải là list không và không rỗng
    # if isinstance(result, list) and result:
    #     # Tìm score cao nhất và chỉ số tương ứng
    #     # Tìm phần tử có score cao nhất
    #     # Tìm độ tương đồng cao nhất và chỉ số tương ứng
    #     max_score = max(result)  # Độ tương đồng cao nhất
    #     # Chỉ số của độ tương đồng cao nhất
    #     max_index = result.index(max_score)
    #     # Chỉ số tương ứng trong documents
    #     print(max_score)
    #     print(max_index)

    #     # Kiểm tra xem score có đủ điều kiện không
    #     if max_score > 0.7:
    #         # Trả về câu trả lời tương ứng
    #         return answers[pos[max_index][0]]
    #     else:
    #         return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."
    # else:
    #     return "Đã xảy ra lỗi khi gọi API."


global previous_questions
previous_questions = []


def handle_user_question(question):
    try:
        response = get_response(question)
        if response == "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.":
            for i in range(1, min(3, len(previous_questions) + 1)):
                last_question = previous_questions[-i]
                combined_question = last_question + " " + question
                response = get_response(combined_question)
                if response != "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.":
                    return response

        # Thêm câu hỏi hiện tại vào danh sách câu hỏi trước đó
        previous_questions.append(question)
        return response
    except (Exception):
        return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."


# # Chạy chatbot
# while True:
#     user_input = input("You: ")

#     if user_input.lower() == "exit":
#         break
#     response = handle_user_question(user_input)
#     print(f"Bot: {response}")
