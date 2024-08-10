from flask import Flask, request, jsonify
from chatbot import handle_user_question
app = Flask(__name__)

# Endpoint để nhận câu query từ Streamlit


@app.route('/query', methods=['POST'])
def handle_query():
    # Nhận câu query từ Streamlit
    data = request.get_json()
    query = data['query']

    # Xử lý câu query (ở đây có thể làm bất kỳ xử lý nào cần thiết)
    response = handle_user_question(query)

    # Trả về câu phản hồi cho Streamlit
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
