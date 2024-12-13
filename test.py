# from sentence_transformers import CrossEncoder

# model = CrossEncoder(
#     "jinaai/jina-reranker-v2-base-multilingual",
#     automodel_args={"torch_dtype": "auto"},
#     trust_remote_code=True,
# )

# # Example query and documents
# query = "những ngành học mới trong học viện năm 2024 tại sao phần ở sau này lại không quan trọng, độ chính xác cao như thế, không ổn tí nào"
# documents = [
#     "những ngành học mới trong học viện năm 2024",
#     "những ngành học mới trong học viện",
#     "điểm chuẩn ngành công nghệ thông tin theo phương thức thi thpt năm 2024 cơ sở miền bắc",
#     "điểm chuẩn của ngành công nghệ thông tin năm 2024 thpt miền bắc",
# ]

# # construct sentence pairs
# sentence_pairs = [[query, doc] for doc in documents]

# scores = model.predict(sentence_pairs, convert_to_tensor=True).tolist()

# rankings = model.rank(
#     query, documents, return_documents=True, convert_to_tensor=True)
# print(f"Query: {query}")
# for ranking in rankings:
#     print(
#         f"ID: {ranking['corpus_id']}, Score: {ranking['score']:.4f}, Text: {ranking['text']}")

import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "1 + 1 bằng mấy."}
    ],
    temperature=0.7,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(response["choices"][0]["message"]["content"])
