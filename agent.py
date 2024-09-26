from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain.tools import tool
from datetime import datetime
import streamlit as st
import pandas as pd


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
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.01)
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

        

        Sempre indique que a resposta tem como base dados do Programa de Subscriação do Seguro Rural (PSR) e exiba o link de acesso: https://dados.agricultura.gov.br/dataset/sisser3
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
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.01)
    now = datetime.now()

    prefix = f"""
        O dataframe possui dados desde 1991 até 2023. Não se esqueça que estamos no mês {now.strftime("%m")} do ano de {now.strftime("%Y")}.

        Aqui estão os valores possíveis para a coluna "grupo_de_desastre" em formato de lista:

        <grupo_de_desastre>{df_desastres.grupo_de_desastre.unique()}</grupo_de_desastre>
        
        E aqui estão os valores possíveis para a coluna "desastre" (que representa o desastre climático) em formato de lista:
        
        <desastre>{df_desastres.desastre.unique()}</desastre>

        Não responda com exemplos, mas com informações estatísticas.

        Sempre que não for indicado um ano específico na pergunta, utilize a biblioteca datetime do python.

        Sempre indique que a resposta obtida tem como base os dados do Atlas de Desastres no Brasil e exiba o link de acesso: https://atlasdigital.mdr.gov.br/paginas/downloads.xhtml)
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

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore.from_existing_index('irc-chatbot', embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.8}
)

retrieve_climate_report_documents = create_retriever_tool(
    retriever=retriever,
    name='inmet_docs_retriever',
    description='Procura e retorna trechos relevantes dos Boletins Agroclimatológicos do Instituto Nacional de Meteorologia (INMET)',
)

tools = [retrieve_climate_report_documents, get_insurance_policies_data, get_natural_disasters_data]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01, api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)


# State Graph
class State(MessagesState):
    pass

# Nodes
def reasoner(state: State):
    system_prompt = 'You are a helpful climate assistant tasked with answering the user input by generating insights based on the information retrieved from climate reports, disaster and insurance database.'
    return {'messages': [llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state['messages'])]}

tool_node = ToolNode(tools)

# Graph
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