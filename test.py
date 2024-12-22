from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
)

# Example query and documents
query = "điểm chuẩn của ngành công nghệ thông tin năm 2024 miền nam năng lực"
documents = [
    "điểm chuẩn của ngành công nghệ thông tin năm 2024 là miền nam năng lực",
    "điểm chuẩn ngành công nghệ thông tin theo phương thức thi đánh giá năng lực năm 2024 tại miền nam là bao nhiêu",
    "học bổng đặc biệt",
    "chi tiết học bổng toàn phần",
]

# construct sentence pairs
sentence_pairs = [[query, doc] for doc in documents]

scores = model.predict(sentence_pairs, convert_to_tensor=True).tolist()

rankings = model.rank(
    query, documents, return_documents=True, convert_to_tensor=True)
print(f"Query: {query}")
for ranking in rankings:
    print(
        f"ID: {ranking['corpus_id']}, Score: {ranking['score']:.4f}, Text: {ranking['text']}")
