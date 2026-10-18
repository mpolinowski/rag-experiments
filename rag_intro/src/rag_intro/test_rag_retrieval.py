from pathlib import Path
from pydantic import SecretStr

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI

db_path = Path("src/rag_intro/db/")
db_path.parent.mkdir(parents=True, exist_ok=True)

# DB connection and text retrieval

embed_model = OpenAIEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
    check_embedding_ctx_length=False
)

db_con = Chroma(persist_directory=str(db_path), embedding_function=embed_model)

# Text generation from retrieved text

user_query = "What is the diagonal of the SONY IMX415-AAQR STARVIS sensor?"
retrieved_chunks = db_con.similarity_search(user_query, k=3)

content = []

for i, chunk in enumerate(retrieved_chunks):
    # # DEBUG
    # print(chunk.page_content)
    content.append(chunk.page_content)

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
)

print(client.invoke(f"{user_query}. Use the following context to answer the question: {str(content)}").content)
