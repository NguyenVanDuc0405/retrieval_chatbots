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

# Kết nối tới MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot"]
collection_feedback = db["feedback"]


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


if __name__ == '__main__':
    app.run(debug=True)
