from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Modelo de validação para a saída do LLM, garantindo um retorno binário de relevância."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def create_llm(model_name: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    """
    Retorna um modelo de IA configurado com saída estruturada.
    :param model_name: Nome do modelo da OpenAI.
    :param temperature: Controla a aleatoriedade da resposta.
    :return: Modelo de IA configurado.
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return llm.with_structured_output(GradeDocuments)


def create_prompt_template() -> ChatPromptTemplate:
    """
    Cria um template estruturado para avaliação da relevância de um documento.
    :return: Instância de ChatPromptTemplate.
    """
    system_message = """You are a grader assessing relevance of a retrieved document to a user question. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )


def create_retrieval_grader() -> callable:
    """
    Cria o pipeline que une o prompt e o modelo de IA.
    :return: Pipeline pronto para receber inputs.
    """
    prompt = create_prompt_template()
    model = create_llm()
    return prompt | model 



    