import torch
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


def embeddings_model(question):

   # Tokenize và chuyển đổi câu văn bản thành tensor
    tokens = tokenizer(question, return_tensors='pt',
                       padding=True, truncation=True)
    # Sử dụng model để mã hóa câu văn bản thành vector
    with torch.no_grad():
        output = phobert(**tokens)
    # Lấy vector biểu diễn từ outputs của model
    # Trung bình các vector token
    return output.last_hidden_state.mean(dim=1).numpy()


def correction_model(texts, MAX_LENGTH):
    corrector = pipeline("text2text-generation",
                         model="bmd1905/vietnamese-correction")
    return corrector(texts, max_length=MAX_LENGTH)


def rerank_model(query, documents):
    model = AutoModelForSequenceClassification.from_pretrained(
        'jinaai/jina-reranker-v2-base-multilingual',
        torch_dtype="auto",
        trust_remote_code=True,
    )
    query = query

    documents = documents
    # construct sentence pairs
    sentence_pairs = [[query, doc] for doc in documents]

    scores = model.compute_score(sentence_pairs, max_length=1024)
    print("Scores:\n")
    print(scores)
    result = model.rerank(
        query,
        documents,
        max_query_length=512,
        max_length=1024,
        top_n=3
    )
    print(result)

    print(result[0]['document'])
