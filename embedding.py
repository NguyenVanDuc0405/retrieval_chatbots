import pandas as pd
import numpy as np
from preprocess import processing_text_for_db
import json
import torch
from transformers import pipeline
from model import embeddings_model

df = pd.read_csv('questions_answers.csv')
questions = df['question']
processed_questions = []
for question in questions:  # Sử dụng tqdm để theo dõi tiến trình nếu cần
    # Áp dụng các hàm tiền xử lý
    processed_text = processing_text_for_db(question)
    # Lưu kết quả vào list processed_questions
    processed_questions.append(processed_text)

# Thêm cột mới vào DataFrame hoặc ghi đè lên cột 'question' hiện tại
df['processed_question'] = processed_questions
df['vector_embeddings'] = df['processed_question'].apply(
    embeddings_model)
# Chuyển đổi mảng NumPy thành danh sách và sau đó thành chuỗi JSON
df['vector_embeddings'] = df['vector_embeddings'].apply(
    lambda x: json.dumps(x.tolist()))

# Lưu DataFrame thành file CSV
df.to_csv('embeddings_ver2.csv', index=False)
