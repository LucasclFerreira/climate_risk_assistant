from langchain_core.messages import SystemMessage, HumanMessage
from agent import create_workflow
from time import sleep
import streamlit as st
import re


# Session State
if "chat_history" not in st.session_state:
    initial_message = """
    Olá! Eu sou o **IRC Climate Risk Assistant**, seu assistente especializado em riscos climáticos e seu impacto no seguro rural.

    Você pode perguntar, por exemplo:
    > *Qual o desastre climático mais comum em Minas Gerais?*\n
    > *Qual a sinistralidade para soja no Paraná nesses útlimos três anos?*\n
    > *Qual é o impacto do La Niña na agricultura e qual a chance desse fenômeno em 2025?*
    
    Estou pronto para te ajudar! **O que você gostaria de saber hoje?**
    """
    st.session_state.chat_history = [{"role": "ai", "content": initial_message}]

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
# st.logo('./img/irbped_logo.png')

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

    pre_prompt = """Caso o usuario peça para gerar um grafico gere código python para plotar um grafico interativo no streamlit usando 
ploty sobre a query do usuario, gere apenas o codigo, e faça uma analise abaixo sobre o grafico. Query do Usuário: {query}"""

    # AI Response
    with st.chat_message("ai"):
        with st.spinner("Pensando..."):
            response = get_response(pre_prompt.format(query=user_input) )['messages'][-1].content

            pattern = r"(.*?)```python\n(.*?)\n```(.*)"
            match = re.match(pattern, response, re.DOTALL)

            if match:
                text_before = match.group(1).strip()
                python_code = match.group(2).strip()
                text_after = match.group(3).strip()

                print('[LOG]: SCRIPT GERADO:')
                print(python_code)

                exec(python_code)
                st.write_stream(generate_stream_from_response(text_after))

            else:
                st.write_stream(generate_stream_from_response(response))
    st.session_state.chat_history.append({"role": "ai", "content": response})