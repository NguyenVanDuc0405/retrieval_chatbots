import pandas as pd
import numpy as np
from preprocess import processing_text_for_db
import json
from model import embeddings_model
from pymongo import MongoClient

df = pd.read_csv('data/questions_answers.csv')
questions = df['question']
processed_questions = []
for question in questions:
    processed_text = processing_text_for_db(question)
    processed_questions.append(processed_text)
df['processed_question'] = processed_questions
df['vector_embeddings'] = df['processed_question'].apply(
    embeddings_model)
df['vector_embeddings'] = df['vector_embeddings'].apply(
    lambda x: json.dumps(x.tolist()))
df.to_csv('data/embeddings_ver2.csv', index=False)
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot"]
collection = db["q&a"]
collection.delete_many({})
df = pd.read_csv('data/embeddings_ver2.csv')
data_dict = df.to_dict("records")
collection.insert_many(data_dict)
print("Lưu dữ liệu vào MongoDB thành công!")
