import streamlit as st
import time

# Ví dụ về chuỗi có ký tự xuống dòng
text = '''Miền bắc Học Viện hiện tại đào tạo các ngành như:
Ngành Công nghệ thông tin
Ngành An toàn thông tin
Ngành Kỹ thuật Điện tử viễn thông
Ngành Công nghệ kỹ thuật Điện điện tử
Ngành Công nghệ đa phương tiện
Ngành Truyền thông đa phương tiện
Ngành Marketing
Ngành Quản trị kinh doanh
Ngành Kế toán
Ngành Thương mại điện tử
Ngành Công nghệ thông tin (cử nhân, định hướng ứng dụng)
Ngành Công nghệ tài chính Fintech
Ngành Kỹ thuật Điều khiển và Tự động hoá
Ngành Khoa học máy tính (định hướng Khoa học dữ liệu)
'''

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)


if prompt := st.chat_input("Tin nhắn: "):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    def response_generator():
        response = text
        # Thay thế newline bằng <br> cho HTML
        response = response.replace("\n", "<br>")
        for word in response.split():
            yield word + " "
            time.sleep(0.05)
        yield ""  # Đảm bảo rằng các ký tự newline được hiển thị cuối cùng

    # Create a placeholder for the response
    response_placeholder = st.chat_message("assistant").empty()

    # Collect response parts and update the placeholder
    response_parts = []
    for part in response_generator():
        response_parts.append(part)
        response_placeholder.markdown(
            "".join(response_parts), unsafe_allow_html=True)

    # Final response
    final_response = "".join(response_parts)
    # response_placeholder.markdown(final_response, unsafe_allow_html=True)
    print(final_response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": final_response})
