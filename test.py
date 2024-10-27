from pymongo import MongoClient
import pandas as pd

# Kết nối tới MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot"]  # Thay thế 'ten_database' bằng tên database bạn đã lưu
# Thay thế 'ten_collection' bằng tên collection bạn đã lưu
collection = db["q&a"]

# Lấy các cột mong muốn, ví dụ: 'processed_question' và 'vector_embeddings'
cursor = collection.find(
    {}, {"question": 1, "answer": 1, "processed_question": 1, "vector_embeddings": 1, "_id": 0})

# Chuyển dữ liệu từ MongoDB thành DataFrame
df_selected_columns = pd.DataFrame(list(cursor))

# Hiển thị DataFrame
print(df_selected_columns)
