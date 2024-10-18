from langchain_core.messages import SystemMessage, HumanMessage
from agent import create_workflow
from time import sleep
import streamlit as st


# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "ai", "content": "Olá! Eu sou o **IRC Climate Risk Assistant**. Como posso te ajudar?"}]

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {"configurable": {"thread_id": "124940"}}


# Functions
@st.cache_resource
def instantiate_agent():
    return create_workflow()

def generate_stream_from_response(response):
    for letter in response:
        sleep(0.01)
        yield letter

def get_response(user_input):
    return agent.invoke({"messages": [("human", user_input)]}, config=st.session_state.agent_memory)


agent = instantiate_agent()


# UI
st.logo('./img/irbped_logo.png')

st.title("IRC Climate Risk Assistant 🌿🌦️")


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
        with st.spinner("Pensando..."):
            response = get_response(user_input)['messages'][-1].content
        st.write_stream(generate_stream_from_response(response))
    st.session_state.chat_history.append({"role": "ai", "content": response})