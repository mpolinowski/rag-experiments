"""
Test ingestion + retrieval for the commented object-detection codebase.

Pipeline:
    assets/object-detection/**/*.py (plus README.md + Dockerfile)
        -> RecursiveCharacterTextSplitter.from_language("python")  [1000/150]
        -> qwen3-embedding:0.6b (local Ollama, OpenAI-compatible API)
        -> Chroma collection "object_detection" (persisted under db/)

Then a fixed set of natural-language test queries is run against the
collection to verify that the new module/function docstrings actually make
the right code sections retrievable.

Run with the uv venv (langchain packages live there, not system python3):
    rag_for_code/.venv/bin/python src/rag_for_code/test_object_detection_ingestion.py
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import SecretStr

HERE = Path(__file__).resolve().parent
CODEBASE_ROOT = HERE / "assets" / "object-detection"
DB_DIR = HERE / "db" / "object_detection"
DB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
    check_embedding_ctx_length=False,
)

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=True,
)

sources: list[tuple[Path, str]] = (
    sorted(CODEBASE_ROOT.rglob("*.py"))
    + [CODEBASE_ROOT / "README.md", CODEBASE_ROOT / "Dockerfile"]
)

documents: list[Document] = []
for path in sources:
    if not path.is_file():
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue  # binary payload (weights, model bins)

    language = "python" if path.suffix == ".py" else path.suffix.lstrip(".") or "text"
    doc = Document(
        page_content=text,
        metadata={
            "source": path.relative_to(CODEBASE_ROOT).as_posix(),
            "absolute_path": str(path),
            "language": language,
        },
    )
    chunks = splitter.split_documents([doc])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    documents.extend(chunks)
    print(f"{len(chunks):3d} chunk(s)  {path.relative_to(CODEBASE_ROOT)}")

print(f"\nTotal: {len(documents)} chunks from {len(sources)} files")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="object_detection",
    persist_directory=str(DB_DIR),
)

# ---------------------------------------------------------------------------
# Retrieval test queries
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "How are incoming jobs picked up and queued into the pipeline?",
    "How is motion in the alarm video detected before object detection runs?",
    "How does the YOLO object detection model server request/response over Redis work?",
    "How are face recognition models trained and where are they stored?",
    "How does the plate OCR model server reply with the plate text?",
    "How is the VLM scene description generated for an alarm?",
    "How are the detection results committed to the database after object detection?",
    "What happens when a model server does not answer in time?",
]

k = 5
for query in TEST_QUERIES:
    results = vectorstore.similarity_search(query, k=k)
    print("\n" + "=" * 78)
    print(f"Q: {query}")
    print("=" * 78)
    for rank, doc in enumerate(results, 1):
        snippet = " ".join(doc.page_content.split())[:140]
        print(f"  {rank}. {doc.metadata['source']}  [chunk {doc.metadata['chunk_index']}]")
        print(f"     {snippet}")
