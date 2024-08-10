import streamlit as st
from chatbot import handle_user_question
import requests
import time
# # Địa chỉ của Flask API
# API_URL = 'http://localhost:5000/query'

st.title("Chatbot Demo")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Tin nhắn: "):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    def response_generator():
        response = handle_user_question(prompt)
        response = response.replace("\n", "<br>")
        for word in response.split():
            yield word + " "
            time.sleep(0.05)
        yield ""  # Đảm bảo rằng các ký tự newline được hiển thị cuối cùng
        # response = f"Bot: {response}"
    # Get chatbot response
    # Display assistant response in chat message container
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

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": final_response})
