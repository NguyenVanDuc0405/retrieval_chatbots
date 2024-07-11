from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re


# def cosine_similarity(vec1, vec2):
#     # Tính tích vô hướng (dot product) của hai vector
#     dot_product = np.dot(vec1, vec2)

#     # Tính độ lớn (norm) của từng vector
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)

#     # Tính độ tương đồng cosine
#     if norm_vec1 == 0 or norm_vec2 == 0:
#         return 0.0
#     else:
#         return dot_product / (norm_vec1 * norm_vec2)

# Tính toán Jaccard similarity


def jaccard_similarity(query, document):
    query_words = set(re.split(r'\s+', query.lower().strip()))
    doc_words = set(re.split(r'\s+', document.lower().strip()))
    intersection = query_words.intersection(doc_words)
    union = query_words.union(doc_words)
    return len(intersection) / len(union)


df = pd.read_csv('questions_answers.csv')

# Tách câu hỏi và câu trả lời
questions = df['question']
answers = df['answer']


# Vector hóa câu hỏi
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


def get_response(user_query):
    # vector hóa câu truy vấn
    user_query_vector = vectorizer.transform([user_query])

    cosine_similarities = cosine_similarity(
        user_query_vector, question_vectors).flatten()

    # # Tính toán độ tương đồng cosine giữa câu truy vấn của người dùng và các câu hỏi trong database
    # cosine_similarities = [cosine_similarity(
    #     user_query_vector.toarray()[0], qv.toarray()[0]) for qv in question_vectors]
    jaccard_similarities = [jaccard_similarity(
        user_query, q) for q in questions]

    # Kết hợp kết quả từ hai độ tương đồng
    alpha = 0.6
    beta = 0.4
    combined_scores = alpha * \
        np.array(cosine_similarities) + beta * np.array(jaccard_similarities)

    best_match_index = np.argmax(combined_scores)

    if np.max(combined_scores) > 0.6:
        print(np.max(combined_scores))
        print(questions[best_match_index])
        return answers[best_match_index]

    else:
        print(np.max(combined_scores))
        return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."


# Chạy chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input)
    print(f"Bot: {response}")


# xử lý chỗ phương thức xét tuyển đang bị trl sai câu hỏi
