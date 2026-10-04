import os 
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma

load_dotenv()

os.environ["OPENAI_API_KEY"]= os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"]=os.getenv("OPENROUTER_API_URL")


pdf_loader = DirectoryLoader(
    "docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

document= pdf_loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunk=splitter.split_documents(document)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=chunk,
    embedding=embeddings,
    persist_directory="db/chroma_db"

)

print(
    vectorstore.get(
        include=["embeddings"]
    )
)