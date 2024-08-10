import re
import string
from underthesea import word_tokenize


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
    "tuyển sinh": "xét tuyển",
    "khối": "tổ hợp",

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


def processing_text_for_db_rerank(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    return text


def processing_text_for_query_rerank(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    text = replace_abbreviations(text, abbreviation_dict)
    text = replace_text(text, text_dict)
    return text


# print(processing_text_for_query_rerank(
#     "trường có các phương thức xét tuyển  nào"))
