import streamlit as st
import time


# Functions
def get_response(user_input):
    for letter in user_input:
        yield letter
        time.sleep(0.05)


# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "ai", "content": "Como posso ajudá-lo?"}]


# UI
st.title("IRC: Assistente de Riscos Climáticos")


# Chat
# Messages in Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# New messages
if user_input := st.chat_input("Faça um pergunta..."):
    # User Input
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # AI Response
    with st.chat_message("ai"):
        response = st.write_stream(get_response(user_input))
    st.session_state.chat_history.append({"role": "ai", "content": response})