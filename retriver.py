import os 
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma

load_dotenv()

os.environ["OPENAI_API_KEY"]= os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"]=os.getenv("OPENROUTER_API_URL")


embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embeddings
)


retriever= vectorstore.as_retriever(
    search_type ="similarity",
    search_kwargs= {"k":3}
)


# query = input("Ask your question:")

# document= retriever.invoke(query)


# for doc in document:
#     print(doc.page_content)
#     # print("Source:", doc.metadata.get("source"))