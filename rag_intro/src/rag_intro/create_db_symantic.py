from pathlib import Path
from pydantic import SecretStr

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


asset_path = Path(__file__).resolve().parent.parent / "rag_intro" / "assets"
asset_path.parent.mkdir(parents=True, exist_ok=True)

db_path = Path(__file__).resolve().parent.parent / "rag_intro" / "db"
db_path.parent.mkdir(parents=True, exist_ok=True)

# Load the text source from pdf file

loader = PyPDFLoader(str(asset_path) + "/IMX415-AAQR_Flyer.pdf")
docs = loader.load()
full_text = "\n".join([page.page_content for page in docs])

# Create vector embeddings from pages

embed_model = OpenAIEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
    check_embedding_ctx_length=False
)

splitter = SemanticChunker(embed_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=0.6)
semantic_chunks = splitter.create_documents([full_text])

# # DEBUG
# print(len(semantic_chunks)) # returns 19 -> much larger chunks now
# print(semantic_chunks[0])
# # -> The IMX415-AAQR is a diagonal 6.4 mm (Type 1/2.8) CMOS active pixel type solid-state image sensor with a square pixel array and 8.46 M effective pixels. This chip operates with analog 2.9 V, digital 1.1 V, and interface 1.8 V triple power supply, and has low power consumption.

# Create db with embeddings

db = Chroma.from_documents(semantic_chunks, embed_model, persist_directory=str(db_path))
