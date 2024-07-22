from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
import string
import torch
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize


def jaccard_similarity(query, document):
    query_words = set(re.split(r'\s+', query.lower().strip()))
    doc_words = set(re.split(r'\s+', document.lower().strip()))
    intersection = query_words.intersection(doc_words)
    union = query_words.union(doc_words)
    return len(intersection) / len(union)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def to_lowercase(text):
    return text.lower()


def replace_comma(text):
    return " ".join(text.split(","))


# Tạo bộ từ điển các từ viết tắt
abbreviation_dict = {
    "cntt": "công nghệ thông tin",
    "attt": "an toàn thông tin",
    "iot": "công nghệ internet vạn vật",
    "fintech": "công nghệ tài chính",
    "cndpt": "công nghệ đa phương tiện",
    "ttdpt": "truyền thông đa phương tiện",
    "qtkd": "quản trị kinh doanh",
    "tmdt": "thương mại điện tử",
    "khmt": "khoa học máy tính",
    "clc": "chất lượng cao",
    "trường": "học viện",
    "bạn": "bot",


}

# Hàm để thay thế các từ viết tắt trong câu bằng từ đầy đủ


def replace_abbreviations(text, abbreviation_dict):
    words = text.split()
    new_words = []
    for word in words:
        if word in abbreviation_dict:
            new_words.append(abbreviation_dict[word])
        else:
            new_words.append(word)
    return ' '.join(new_words)


text_dict = {
    "năm nay": "năm 2024",
    "hiện nay": "năm 2024",
}


def replace_text(text, text_dict):
    for key, value in text_dict.items():
        text = re.sub(r'\b{}\b'.format(re.escape(key)),
                      value, text, flags=re.IGNORECASE)
    return text


def tokenizerText(text):
    return word_tokenize(text, format="text")


# Khởi tạo tập hợp để lưu trữ các từ dừng
stopwords = set()
# Đọc từng hàng trong file stopwords.txt
with open('stopwords.txt', 'r', encoding='utf-8') as fp:
    for line in fp:
        word = line.strip()  # Loại bỏ khoảng trắng đầu và cuối dòng
        if word:  # Kiểm tra xem dòng không rỗng
            stopwords.add(word)


def remove_stopwords(line):
    words = []
    for word in line.split():
        if word not in stopwords:
            words.append(word)
    return ' '.join(words)


def processing_text_for_db(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    text = tokenizerText(text)
    text = remove_stopwords(text)
    return text


def processing_text_for_query(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    text = replace_abbreviations(text, abbreviation_dict)
    text = replace_text(text, text_dict)
    text = tokenizerText(text)
    text = remove_stopwords(text)
    return text


# # Function to encode each question into vector representation
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


def encode_question(question):

   # Tokenize và chuyển đổi câu văn bản thành tensor
    tokens = tokenizer(question, return_tensors='pt',
                       padding=True, truncation=True, max_length=512)

    # Sử dụng model để mã hóa câu văn bản thành vector
    with torch.no_grad():
        output = phobert(**tokens)

    # Lấy vector biểu diễn từ outputs của model
    # Trung bình các vector token
    return output.last_hidden_state.mean(dim=1).numpy()


df = pd.read_csv('questions_answers.csv')
# Tách câu hỏi và câu trả lời
questions = df['question']
answers = df['answer']

processed_questions = []
for question in questions:
    # Áp dụng các hàm tiền xử lý
    processed_text = processing_text_for_db(question)

    # Lưu kết quả vào list processed_questions
    processed_questions.append(processed_text)

# Thêm cột mới vào DataFrame hoặc ghi đè lên cột 'question' hiện tại
df['processed_question'] = processed_questions

df['question_vector'] = df['processed_question'].apply(encode_question)
global previous_questions
previous_questions = []


def get_response(user_query):
    processed_query = processing_text_for_query(user_query)
    query_vector = encode_question(processed_query)
    # Tính toán độ tương đồng cosine giữa câu truy vấn của người dùng và các câu hỏi trong database
    cosine_similarities = [cosine_similarity(
        query_vector, qv).flatten() for qv in df['question_vector']]
    jaccard_similarities = [jaccard_similarity(
        processed_query, q) for q in df['processed_question']]

    # Kết hợp kết quả từ hai độ tương đồng
    alpha = 0.6
    beta = 0.4
    # Chuyển đổi jaccard_similarities thành mảng một chiều
    jaccard_array = np.array(jaccard_similarities).reshape(
        len(df['processed_question']), 1)

    # Kết hợp điểm số từ hai độ tương đồng
    combined_scores = alpha * \
        np.array(cosine_similarities) + beta * jaccard_array

    best_match_index = np.argmax(combined_scores)
    if np.max(combined_scores) > 0.7:
        print(np.max(combined_scores))
        print(len(previous_questions))
        print(processed_query)
        print(processed_questions[best_match_index])
        print(questions[best_match_index])
        return answers[best_match_index]
    else:
        print(processed_query)
        print(np.max(combined_scores))
        return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."


# Khởi tạo một danh sách để lưu trữ các câu hỏi người dùng trước đó


def handle_user_question(question):
    try:
        response = get_response(question)
        if response == "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.":
            for i in range(1, min(3, len(previous_questions) + 1)):
                last_question = previous_questions[-i]
                combined_question = last_question + " " + question
                response = get_response(combined_question)
                if response != "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn.":
                    return response

        # Thêm câu hỏi hiện tại vào danh sách câu hỏi trước đó
        previous_questions.append(question)
        return response
    except (Exception):
        return "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."

        # if (get_response(question) == "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."):
        #     last_question = previous_questions[-1]
        #     combined_question = last_question + " " + question
        #     response = get_response(combined_question)
        #     previous_questions.append(question)

        #     if (response == "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."):
        #         last_question = previous_questions[-2]
        #         combined_question = last_question + " " + question
        #         response = get_response(combined_question)
        #         previous_questions.append(question)

        #         if (response == "Tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi đầy đủ hơn."):
        #             last_question = previous_questions[-3]
        #             combined_question = last_question + " " + question
        #             previous_questions.append(question)

        #             # Gọi hàm xử lý và trả lời ở đây cho câu hỏi kết hợp
        #             response = get_response(combined_question)
        #         else:
        #             return response
        #     else:
        #         return response
        # else:
        #     # Gọi hàm xử lý và trả lời ở đây cho câu hỏi hiện tại
        #     response = get_response(question)
        #     previous_questions.append(question)
        #     return response


# Vector hóa câu hỏi
# vectorizer = TfidfVectorizer()
# question_vectors = vectorizer.fit_transform(questions)


# Chạy chatbot
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break
    response = handle_user_question(user_input)
    print(f"Bot: {response}")
