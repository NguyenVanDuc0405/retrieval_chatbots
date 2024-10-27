from pymongo import MongoClient
import pandas as pd

# Kết nối đến MongoDB (thay thế URL bằng URL của bạn)
client = MongoClient("mongodb://localhost:27017/")

# Chọn database (nếu database chưa tồn tại, MongoDB sẽ tạo mới)
db = client["chatbot"]

# Chọn collection (nếu collection chưa tồn tại, MongoDB sẽ tạo mới)
collection = db["q&a"]

df = pd.read_csv('data/embeddings_ver2.csv')
# Chuyển đổi DataFrame thành danh sách dictionary
data_dict = df.to_dict("records")

# Chèn các document vào MongoDB collection
collection.insert_many(data_dict)

print("Lưu dữ liệu vào MongoDB thành công!")
