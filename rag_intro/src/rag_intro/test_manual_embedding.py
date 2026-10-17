import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

asset_path = Path("src/rag_intro/assets/IMX415-AAQR_Flyer.pdf")
asset_path.parent.mkdir(parents=True, exist_ok=True)

# # DEBUG
# if os.path.isfile(asset_path):
#     print(str(asset_path))


# Load the text source from pdf file

loader = PyPDFLoader(asset_path)
text = loader.load()

# # DEBUG
# pages = loader.load_and_split()
# print(len(pages))
# print(pages[0])

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(text)

# # DEBUG
# print(type(chunks))
# print(len(chunks))
# print(chunks[0])


# Create vector embeddings from pages

embed_model = OpenAIEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    check_embedding_ctx_length=False
)

embedded_chunks = embed_model.embed_documents([str(chunk) for chunk in chunks])

# # DEBUG
# print(type(embedded_chunks))
# print(len(embedded_chunks))
# print(embedded_chunks[0])
