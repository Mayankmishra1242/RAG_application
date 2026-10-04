import os 
from dotenv import load_dotenv 

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from retriver import retriever

load_dotenv()

os.environ["OPENAI_API_KEY"]= os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"]=os.getenv("OPENROUTER_API_URL")


llm= ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0
)

prompt=ChatPromptTemplate.from_template(
    """
     Answer the question accoring to the context.

     Context:
     {context}

     Question:
     {question}

     Answer:
    """
)

chain = prompt | llm | StrOutputParser()

query = input("Ask your question:")

document= retriever.invoke(query)

context = "\n\n".join(doc.page_content for doc in document)

answer= chain.invoke(
    {
        "context":context,
        "question":query
    }
)

print("\nAnswer:", answer)