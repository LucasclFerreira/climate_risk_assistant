from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain.tools import tool
from datetime import datetime
import streamlit as st
import pandas as pd

from rag import create_workflow


OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']


# Tools
@tool
def get_insurance_policies_data(query: str): 
    """
    Use this tool to query data from a rural insurance policies dataframe.

    Keyword arguments:
    query -- user's question to be answered by the pandas dataframe agent
    """
    df_apolices = pd.read_parquet("./data/psr_LLM.parquet")
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    now = datetime.now()

    prefix = f"""
        O dataframe possui dados desde 2006 até 2023. Não se esqueça que estamos no mês {now.strftime("%m")} do ano de {now.strftime("%Y")}.

        Aqui estão os valores possíveis para a coluna "cultura" em formato de lista:

        <cultura>{df_apolices.cultura.unique()}</cultura>

        Lembre-se que é importante separar a 1ª e 2ª safra do milho.

        E aqui estão os valores possíveis para a coluna "evento_preponderante" (que representa o desastre climático que causou o sinistro) em formato de lista: 
        
        <evento_preoponderante>{df_apolices.evento_preponderante.unique()}</evento_preponderante>

        Lembre-se que o se o "evento_preponderante" for "-", significa que não houve sinistro na apólice, ou seja, não ocorreu nenhum evento climático
        
        Nunca esqueça que o cálculo da sinistralidade (índice de sinistralidade) deve incluir qualquer valor da coluna "evento_preponderante", incluindo apólices em que o valor é "-".

        Como um especialista em seguro rural você conhece todos os termos relevantes como índice de sinistralidade, taxa do prêmio, importância segurada, entre outros.

        Não responda com exemplos, mas com informações estatísticas.
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
    df_desastres = pd.read_parquet("./data/desastres_LLM.parquet")
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    now = datetime.now()

    prefix = f"""
        O dataframe possui dados desde 1991 até 2023. Não se esqueça que estamos no mês {now.strftime("%m")} do ano de {now.strftime("%Y")}.

        Aqui estão os valores possíveis para a coluna "grupo_de_desastre" em formato de lista:

        <grupo_de_desastre>{df_desastres.grupo_de_desastre.unique()}</grupo_de_desastre>
        
        E aqui estão os valores possíveis para a coluna "desastre" (que representa o desastre climático) em formato de lista:
        
        <desastre>{df_desastres.desastre.unique()}</desastre>

        Não responda com exemplos, mas com informações estatísticas.

        Sempre que não for indicado um ano específico na pergunta, utilize a biblioteca datetime do python.
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


retrieval_agent = create_workflow()
@tool
def retrieve_climate_report_documents(query: str):
    """
    Use this tool to retrieve relevant documents to answer the user's question

    Keyword arguments:
    query -- user's question to be answered
    """

    inputs = {"question": query, "iteration": 0}

    response = retrieval_agent.pick('generation').invoke(inputs, {"recursion_limit": 5})
    return response


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01, api_key=OPENAI_API_KEY)

tools = [retrieve_climate_report_documents, get_insurance_policies_data, get_natural_disasters_data]
llm_with_tools = llm.bind_tools(tools)


# State Graph
class State(MessagesState):
    pass

# Nodes
def reasoner(state: State):
    system_prompt = """
    You are a helpful climate assistant tasked with answering the user input by generating insights based on the information retrieved from climate reports and databases.

    You can not answer questions out of the topics: climate, weather, and insurance.
    
    Use five sentences maximum and keep the answer concise. 

    When using retrieved information, cite all the sources used with the title and page of the documents used in the answer. 

    Everytime you use a tool to get data (about insurance or natural disasters), retrieve climate report documents to complement the answer.
    """
    return {'messages': [llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state['messages'])]}

tool_node = ToolNode(tools)

# Graph

def create_workflow():
    workflow = StateGraph(State)
    workflow.add_node('reasoner', reasoner)
    workflow.add_node('tools', tool_node)

    # Edges
    workflow.add_edge(START, 'reasoner')
    workflow.add_conditional_edges('reasoner', tools_condition)
    workflow.add_edge('tools', 'reasoner')

    # Agent
    memory = MemorySaver()
    agent = workflow.compile(checkpointer=memory)
    return agent