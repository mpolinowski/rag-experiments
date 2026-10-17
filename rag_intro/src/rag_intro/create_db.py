import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


asset_path = Path(__file__).resolve().parent.parent / "rag_intro" / "assets"
asset_path.parent.mkdir(parents=True, exist_ok=True)

db_path = Path(__file__).resolve().parent.parent / "rag_intro" / "db"
db_path.parent.mkdir(parents=True, exist_ok=True)

# Load the text source from pdf file

loader = PyPDFLoader(asset_path)
text = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(text)

# Create vector embeddings from pages

embed_model = OpenAIEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    check_embedding_ctx_length=False
)

# Create db with embeddings

db = Chroma.from_documents(chunks, embed_model, persist_directory=str(db_path))
