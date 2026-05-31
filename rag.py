import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHROMA_DIR = "./chroma_db"
DATA_DIR = "./data"

_retriever = None


def setup_rag_system():
    global _retriever

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing ChromaDB index from disk...")
        vector_store = Chroma(
            persist_directory=CHROMA_DIR, embedding_function=embeddings
        )
    else:
        print("Building ChromaDB index from documents...")
        loader = DirectoryLoader(
            DATA_DIR,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            chunks, embeddings, persist_directory=CHROMA_DIR
        )

    _retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )


@asynccontextmanager
async def lifespan(app):
    setup_rag_system()
    yield


def get_rag_response(query: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    docs = _retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)

    prompt = (
        "أجب على السؤال التالي باللغة العربية فقط، بناءً على المعلومات المقدمة.\n\n"
        f"السياق:\n{context}\n\n"
        f"السؤال: {query}"
    )

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
