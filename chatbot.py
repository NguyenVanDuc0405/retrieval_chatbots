from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

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
    # tính toán độ tương đồng cosine giữa 2 vector truy vấn
    # của người dùng và vector câu hỏi đã biết
    similarities = cosine_similarity(
        user_query_vector, question_vectors).flatten()

    best_match_index = np.argmax(similarities)
    return answers[best_match_index]


# Chạy chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input)
    print(f"Bot: {response}")


# xử lý chỗ phương thức xét tuyển đang bị trl sai câu hỏi
