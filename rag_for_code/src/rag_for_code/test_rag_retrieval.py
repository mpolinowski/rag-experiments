"""
PoC: parent-document retrieval on top of the commented object-detection index.

Two-stage lookup:

Stage 1 (`initial_search`)
    Vector similarity search over the chunk-level Chroma collection
    (`db/object_detection`). Returns the top-k chunks together with the
    metadata the ingestion step saved: `absolute_path`, `chunk_index`,
    `start_index` (char offset of the chunk in the original file).

Stage 2 (`get_parent_context`)
    Takes the (absolute_path, start_index) of *one* hit and reads the
    source file back, then extracts the whole enclosing top-level
    function/class that contains that offset - so a coding agent gets the
    full implementation instead of a ~1000-char slice. Falls back to a
    character window when the offset is missing or not inside any
    function (e.g. a config module).

Run with the uv venv (langchain packages live there, not system python3):
    rag_for_code/.venv/bin/python src/rag_for_code/test_rag_retrieval.py
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

HERE = Path(__file__).resolve().parent
DB_DIR = HERE / "db" / "object_detection"

MAX_CONTEXT_CHARS = 12_000  # keep parent context bounded for the PoC


@dataclass
class SearchHit:
    """One ranked chunk from stage 1, with everything stage 2 needs."""

    source: str
    absolute_path: str
    chunk_index: int
    start_index: int | None
    snippet: str


def get_vector_store() -> Chroma:
    """Open the persisted Chroma collection (created by
    `test_object_detection_ingestion.py`)."""
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url="http://localhost:11434/v1",
        api_key=SecretStr("ollama"),
        check_embedding_ctx_length=False,
    )
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="object_detection",
    )


# ---------------------------------------------------------------------------
# Stage 1 - initial similarity search
# ---------------------------------------------------------------------------


def initial_search(query: str, k: int = 5) -> list[SearchHit]:
    """Run the vector search and rank the results for inspection.

    The chunk metadata carries `absolute_path` and `start_index`; those are
    exactly what `get_parent_context` consumes in stage 2.
    """
    vector_store = get_vector_store()
    docs: list[Document] = vector_store.similarity_search(query, k=k)

    hits: list[SearchHit] = []
    for doc in docs:
        meta = doc.metadata
        hits.append(
            SearchHit(
                source=meta.get("source", "?"),
                absolute_path=meta.get("absolute_path", ""),
                chunk_index=meta.get("chunk_index", -1),
                start_index=meta.get("start_index"),
                snippet=" ".join(doc.page_content.split())[:160],
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Stage 2 - fetch the parent block (whole function) behind one hit
# ---------------------------------------------------------------------------


def _enclosing_definition(source: str, start_index: int) -> tuple[str, str] | None:
    """Return the source text of the smallest function/class whose body
    contains `start_index`.

    Uses AST line spans (cheap, no per-node char offsets needed): map the
    char offset to a line, collect every FunctionDef / AsyncFunctionDef /
    ClassDef that spans it, and pick the most specific one (smallest span).
    """
    tree = ast.parse(source)
    line_of_offset = source.count("\n", 0, start_index) + 1

    candidates = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if node.lineno <= line_of_offset <= (node.end_lineno or node.lineno):
            span = (node.end_lineno or node.lineno) - node.lineno
            candidates.append((span, node))

    if not candidates:
        return None

    _, best = min(candidates, key=lambda pair: pair[0])
    lines = source.splitlines()
    text = "\n".join(lines[best.lineno - 1 : best.end_lineno])
    return text, f"{type(best).__name__}: {getattr(best, 'name', '<module>')}"


def _window_around(source: str, start_index: int, radius: int = 800) -> str:
    """Fallback: character window when the offset isn't inside a def."""
    start = max(0, start_index - radius)
    return source[start : start_index + radius]


def get_parent_context(hit: SearchHit) -> str:
    """Fetch the full source block (function/class) behind one search hit.

    This is the "parent retrieval" half of the pattern: the chunk said
    *where*, the source at `absolute_path` + `start_index` says *what
    exactly*. Output is truncated to `MAX_CONTEXT_CHARS` as a safety cap.
    """
    path = Path(hit.absolute_path)
    if not path.is_file():
        return f"[parent context unavailable: file not found: {path}]"

    source = path.read_text(encoding="utf-8")

    if hit.start_index is None:
        # Chunk lost its offset (or old ingestion) - fall back to the
        # module header, which usually carries the file's purpose.
        return source[:MAX_CONTEXT_CHARS]

    enclosing = _enclosing_definition(source, hit.start_index)
    if enclosing is not None:
        text, description = enclosing
        header = f"# Parent block ({description}) in {path.name}"
    else:
        text = _window_around(source, hit.start_index)
        header = f"# Character window around offset {hit.start_index} in {path.name}"

    body = text if len(text) <= MAX_CONTEXT_CHARS else text[:MAX_CONTEXT_CHARS] + "\n# ... [truncated]"
    return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# PoC driver
# ---------------------------------------------------------------------------


def run_poc(query: str, k: int = 3, follow_up: int = 1) -> None:
    """Search, show the ranked chunks, then pull full context for the
    top `follow_up` hit(s) - the classic two-stage RAG-for-code flow."""
    print("=" * 78)
    print(f"Query: {query}")
    print("=" * 78)

    hits = initial_search(query, k=k)
    for rank, hit in enumerate(hits, 1):
        offset = "n/a" if hit.start_index is None else str(hit.start_index)
        print(f"  {rank}. {hit.source}  [chunk {hit.chunk_index}, offset {offset}]")
        print(f"     {hit.snippet}")

    print("\n" + "-" * 78)
    print(f"Stage 2 - parent context for top hit: {hits[0].source}")
    print("-" * 78)
    context = get_parent_context(hits[0])
    print(context)
    print(
        f"\n[stage 2 delivered {len(context)} chars "
        f"vs. {len(hits[0].snippet)}-char chunk slice]"
    )


if __name__ == "__main__":
    run_poc("How are incoming jobs picked up and queued into the pipeline?")
    run_poc("How is the VLM scene description generated for an alarm?")
