from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ('human', '''
        You are an expert in meteorology and climate risk in the context of reinsurance company with products for both urban and agricultural insurance. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise. Always cite the title and page of the documents utilized in the answer. The answer should be in portuguese.
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:
    ''')
])

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Post-processing
def format_docs(docs):
    formatted = [f"Article source: {doc.metadata['title']} on page {int(doc.metadata['page'])}\nArticle snippet: {doc.page_content}" for doc in docs]
    return '\n\n' + '\n\n'.join(formatted)

# Chain
rag_chain = prompt | llm | StrOutputParser()