from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
)

# Example query and documents
query = "những ngành học mới trong học viện năm 2024 tại sao phần ở sau này lại không quan trọng, độ chính xác cao như thế, không ổn tí nào"
documents = [
    "những ngành học mới trong học viện năm 2024",
    "những ngành học mới trong học viện",
    "điểm chuẩn ngành công nghệ thông tin theo phương thức thi thpt năm 2024 cơ sở miền bắc",
    "điểm chuẩn của ngành công nghệ thông tin năm 2024 thpt miền bắc",
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
