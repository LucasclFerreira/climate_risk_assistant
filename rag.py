from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from pinecone.grpc import PineconeGRPC as Pinecone
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.schema import Document
from typing import List
import streamlit as st

from llms.retrieval_grader import retrieval_grader
from llms.rag_generation import rag_chain


PINECONE_HOST = st.secrets["PINECONE_HOST"]

pc = Pinecone()
index = pc.Index(host=PINECONE_HOST)


def format_docs(docs) -> str:
    formatted = [f"Article source: {doc.metadata['title']} on page {int(doc.metadata['page'])}\nArticle snippet: {doc.page_content}" for doc in docs]
    return '\n\n' + '\n\n'.join(formatted)


# Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        iteration: number of rewrite-retrieve iterations
        generation: LLM generation
        rewrite_question: whether the question should be re-written
        documents: list of documents
    """

    question: str
    iteration: str
    generation: str
    rewrite_question: str
    documents: List[Document]

# Nodes
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    question = state["question"]

    embedding = OpenAIEmbeddings(model='text-embedding-3-small')

    # Vectorstores
    vectorstore_inmet = PineconeVectorStore(index, embedding, namespace='INMET')
    vectorstore_mapa = PineconeVectorStore(index, embedding, namespace='MAPA')
    vectorstore_ipcc = PineconeVectorStore(index, embedding, namespace='IPCC')
    vectorstore_wmo = PineconeVectorStore(index, embedding, namespace='WMO')
    vectorstore_articles= PineconeVectorStore(index, embedding, namespace='Articles')

    # Retrievers
    k = 3
    search_type = "similarity_score_threshold"
    score_thresold = 0.5

    retriever_inmet = vectorstore_inmet.as_retriever(search_type=search_type, search_kwargs={"k": k,"score_threshold": score_thresold}, )
    retriever_mapa = vectorstore_mapa.as_retriever(search_type=search_type, search_kwargs={"k": k, "score_threshold": score_thresold})
    retriever_ipcc = vectorstore_ipcc.as_retriever(search_type=search_type, search_kwargs={"k": k, "score_threshold": score_thresold})
    retriever_wmo = vectorstore_wmo.as_retriever(search_type=search_type, search_kwargs={"k": k, "score_threshold": score_thresold})
    retriever_articles = vectorstore_articles.as_retriever(search_type=search_type, search_kwargs={"k": k, "score_threshold": score_thresold})

    # Parallel Retrieval Chain
    chain = RunnableParallel(
        inmet=retriever_inmet,
        mapa=retriever_mapa,
        ipcc=retriever_ipcc,
        wmo=retriever_wmo,
        articles=retriever_articles
    )

    retrieved_documents = chain.invoke(question)
    combined_documents = [doc for docs in retrieved_documents.values() for doc in docs]

    print(f"Total documents: {len(combined_documents)}")
    return {"documents": combined_documents, "question": question}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(doc)

    print(f'Total documents after filtering: {len(filtered_docs)}')
    return {"documents": filtered_docs, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
   
    question = state["question"]
    documents = state["documents"]

    formatted_documents = format_docs(documents)

    generation = rag_chain.invoke({"context": formatted_documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# Conditional Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    if len(state["documents"]) < 1:
        return "no documents"
    else:
        return "generate"
    

# Workflow
def create_workflow():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "no documents": END,
            "generate": "generate",
        },
    )
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app