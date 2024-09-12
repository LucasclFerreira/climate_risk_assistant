from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain.tools import tool
import streamlit as st
import pandas as pd


OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
memory = MemorySaver()

vectorstore = PineconeVectorStore.from_existing_index('irc-chatbot', embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.8}
)


# Tools
@tool
def get_current_date():
    """"
        Use this tool every time there is a word or number related to time or date in the question. For example, sentences that contain phrases like "last two years", "today", "next semester".
        Returns the current date in YYYY-mm-dd format.
    """
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d")

@tool
def get_insurance_policies_data(query: str): 
    """
    Use this tool to query data from a rural insurance policies dataframe.

    Keyword arguments:
    query -- user's question to be answered by the pandas dataframe agent
    """
    df_apolices = pd.read_parquet("./data/psr2016a2021_tratado.parquet")
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    prefix = f"""Aqui estão os valores possíveis para a coluna "cultura" em formato de lista: {df_apolices.cultura.unique()}\nE aqui estão os valores possíveis para a coluna "evento_preponderante" (que representa o desastre climático que causou o sinistro) em formato de lista: {df_apolices.evento_preponderante.unique()}\nLembre-se que o evento_preponderante "-" significa que não houve sinistro na apólice, ignore tal valor quando for relevante.\nComo um especialista em seguro rural você conhece todos os termos relevantes como índice de sinistralidade, taxa do prêmio, importância segurada, entre outros.
    """

    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df_apolices,
        prefix=prefix,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    response = pandas_agent.pick('output').invoke(query)

    return response

@tool
def get_natural_disasters_data(query: str): 
    """
    Use this tool to query data from natural disasters and extreme weather events

    Keyword arguments:
    query -- user's question to be answered by the pandas dataframe agent
    """
    df_desastres = pd.read_csv("./data/desastres_brasil.csv")
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    prefix = f"""Aqui estão os valores possíveis para a coluna "grupo_de_desastre" em formato de lista: {df_desastres.grupo_de_desastre.unique()}\nE aqui estão os valores possíveis para a coluna "descricao_tipologia" (que representa o desastre climático) em formato de lista: {df_desastres.desastre.unique()}\n
    """

    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df_desastres,
        prefix=prefix,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    response = pandas_agent.pick('output').invoke(query)

    return response


# Instatiating Tools
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='inmet_docs_retriever',
    description='Procura e retorna trechos relevantes dos Boletins Agroclimatológicos do Instituto Nacional de Meteorologia (INMET)'
)
insurance_tool = get_insurance_policies_data
disaster_tool = get_natural_disasters_data
datetime_tool = get_current_date

tools = [retriever_tool, insurance_tool, disaster_tool, datetime_tool]


# Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)