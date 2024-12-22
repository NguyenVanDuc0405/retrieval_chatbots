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
    "ny": "người yêu",
    "ptit": "học viện",
    "hcm": "hồ chí minh",
    "tp": "thành phố",
    "clb": "câu lạc bộ",
    "website": "trang web",
    "đk": "đăng ký",
    "nvqs": "nghĩa vụ quân sự",
    "ktx": "ký túc xá",
    "xtkh": "xét tuyển kết hợp"


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
    "năm gần nhất": "năm 2024",
    "hiện nay": "năm 2024",
    "điểm đầu vào": "điểm chuẩn",
    "vị trí": "địa chỉ",
    # "nam": "miền nam",
    # "bắc": "miền bắc",
    "cơ sở miền nam": "miền nam",
    "cơ sở miền bắc": "miền bắc",
    "năng lực, tư duy": "năng lực",
    "song ngành": "song bằng",
    "hợp tác": "liên kết",



}


def replace_text(text, text_dict):
    for key, value in text_dict.items():
        text = re.sub(r'\b{}\b'.format(re.escape(key)),
                      value, text, flags=re.IGNORECASE)
    return text


def replace_key(text):
    if "hình ảnh" in text:
        return "Những câu hỏi liên quan đến hình ảnh"
    if "ảnh" in text:
        return "Những câu hỏi liên quan đến hình ảnh"
    if "image" in text:
        return "Những câu hỏi liên quan đến hình ảnh"
    if "cảm ơn" in text:
        return "Những câu hỏi liên quan đến cảm ơn"
    if "thanks" in text:
        return "Những câu hỏi liên quan đến cảm ơn"
    if "thank you" in text:
        return "Những câu hỏi liên quan đến cảm ơn"
    return text


def tokenizerText(text):
    return word_tokenize(text, format="text")


# stopwords = set()
# with open('data/stopwords.txt', 'r', encoding='utf-8') as fp:
#     for line in fp:
#         word = line.strip()
#         if word:
#             stopwords.add(word)


# def remove_stopwords(line):
#     words = []
#     for word in line.split():
#         if word not in stopwords:
#             words.append(word)
#     return ' '.join(words)


stopwords_VN = set()
with open('data/vietnamese_stopwords.txt', 'r', encoding='utf-8') as fp:
    for line in fp:
        word = line.strip()
        if word:
            stopwords_VN.add(word)


def remove_stopwords_VN(line):
    for word in stopwords_VN:
        line = re.sub(r'\b' + re.escape(word) + r'\b', '', line)
    line = ' '.join(line.split())
    return line.strip()


def processing_text_for_db(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    text = remove_stopwords_VN(text)
    text = tokenizerText(text)
    return text


def processing_text_for_query(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    text = remove_stopwords_VN(text)
    text = replace_abbreviations(text, abbreviation_dict)
    text = replace_key(text)
    text = replace_text(text, text_dict)
    text = tokenizerText(text)
    return text


def processing_text_for_db_rerank(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    # text = remove_stopwords_VN(text)
    return text


def processing_text_for_query_rerank(text):
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_comma(text)
    text = remove_stopwords_VN(text)
    text = replace_abbreviations(text, abbreviation_dict)
    text = replace_key(text)
    text = replace_text(text, text_dict)
    return text
