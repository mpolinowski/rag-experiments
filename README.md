# RAG & LangGraph — Hands-on Experiments

> 🎓 Educational experiments on the building blocks of **RAG pipelines** and
> **LangGraph state-machine agents** in Python, on
> [LangChain 1.x](https://python.langchain.com/docs/).
> Each folder is a self-contained [`uv`](https://docs.astral.sh/uv/) package that
> builds **one** idea on top of the last — from a raw API call, through embeddings
> into a vector DB, all the way to human-in-the-loop graph control flow.
> Gets to know what each library piece actually does.

* [LangGraph concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/)
* [LangGraph memory / checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
* [LangGraph human-in-the-loop](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/)
* [LangChain RAG](https://python.langchain.com/docs/concepts/rag/)
* [Chroma vector DB](https://docs.trychroma.com/)

---

# Just Imagine

> You have an INSTAR Surveillance Camera system. Someone wants to know two
> different things: *when was that car on license plate `K:SX:712` last seen* —
> and *what is the diagonal of the SONY IMX415-AAQR STARVIS sensor* those cameras
> are built on. Here comes the power of RAG — but only for the half the model can't
> answer from memory: the sensor question is answered by retrieving passages from
> the actual datasheet (a PDF in `rag_intro/assets/`), while the plate question
> comes from a (mock) database. One agent, two tools — the model decides which one
> each question is for. All thanks to Python, LangChain and a local LLM!
>
> ```
> Q1 (routes to Chroma tool): What is the diagonal of the SONY IMX415-AAQR STARVIS sensor?
> → 6.4 mm (Type 1/2.8) — retrieved from the datasheet passages in the vector DB
>
> Q2 (routes to plate tool): When was the car with license plate K:SX:712 last seen?
> → Wed Oct 14 12:01:45 PM HKT 2026, alarm triggered by Motion area 3
> ```

---

## 🛠️ What are these technologies used for?

**RAG** closes the gap between what an LLM *knows* and what it *should use*: your
documents get chunked, embedded and stored in a vector DB, then the most relevant
passages are retrieved and handed to the model as context.

| Building block | What it is |
| :-- | :-- |
| **Document loader** 📄 | `PyPDFLoader` — text out of a PDF, one page per document. |
| **Text splitter** ✂️ | `RecursiveCharacterTextSplitter` (fixed size/overlap) and `SemanticChunker` (split on embedding similarity → fewer, larger, topic-coherent chunks). |
| **Embeddings** 🧮 | `OpenAIEmbeddings` pointed at a local Ollama (`qwen3-embedding:0.6b`) — turns text into vectors. |
| **Vector DB** 🗄️ | **Chroma** — stores embeddings, answers "what is semantically closest to this query?". |
| **Retrieval** 🔎 | `similarity_search(query, k=3)` — the *R* in RAG. |
| **Agent + tools** 🤖 | `create_agent` + `@tool` — the model *chooses* between the RAG lookup and a hard-coded DB lookup. |

**LangGraph** takes those linear chains and turns them into explicit **state
machines**: nodes are functions, edges define control flow, and state is typed —
so loops, branches, parallelism, memory and human approval become just wiring.

| Building block | What it is |
| :-- | :-- |
| **StateGraph** 🕸️ | Nodes + edges (`START`/`END`) compiled into an executable graph. |
| **State schema** 📐 | A `TypedDict` or Pydantic model that every node reads from / writes to — including `Annotated[..., add]` **reducers** that *append* messages instead of overwriting. |
| **Conditional edges** 🔀 | A routing function picks the next node — the backbone of routing, ReAct and generator/evaluator loops. |
| **Checkpointer** 🧠 | `InMemorySaver` / `MemorySaver` + `thread_id` — the graph *remembers* previous runs on the same thread. |
| **`interrupt()` / `Command`** 🙋 | Human-in-the-loop: pause the graph, wait for a real decision, resume on the same thread. |

This repo is that stack in miniature: **embed documents** → **retrieve from them**
→ **drive it all as a graph**.

---

## 📚 Research — a tour of the folders

**The learning ladder (suggested order):**

```
📄 rag_intro              PDF → chunks → embeddings → Chroma → retrieval → RAG agent   (hello RAG)
🕸️ langgraph_intro/01     one node, one edge, a TypedDict state                        (hello world)
📐 02..04                 Pydantic state, message reducers, prompt chains
🕵️ 05                     hand-rolled ReAct loop (agent → tools → agent)
⚡ 06..07                 parallel branches + structured-output routing
🏭 08..09                 orchestrator/worker + generator/evaluator loop
🧠 10..11                 memory (checkpointer) + human-in-the-loop
```

### 📁 rag_intro

| 📁 Module | What's inside | 🎯 Takeaway |
| :-- | :-- | :-- |
| `test_api_calls.py` 💬 | A raw `openai`-SDK `responses.create(...)` call against the local Ollama endpoint — **no LangChain at all**. | What's underneath everything: it's just an OpenAI-compatible HTTP API. |
| `test_manual_embedding.py` 🧮 | `PyPDFLoader` reads the SONY IMX415 datasheet, `RecursiveCharacterTextSplitter` (100 chars / 20 overlap) chunks it, `OpenAIEmbeddings` (Ollama `qwen3-embedding:0.6b`) turns every chunk into a vector — printed, not stored. | Watch each stage of the RAG pipeline by hand: docs → chunks → vectors. |
| `create_db.py` 🗄️ | The same pipeline, but the end is `Chroma.from_documents(...)`, persisted to `src/rag_intro/db/`. | One call from "embeddings in a list" to "a queryable vector DB". |
| `create_db_symantic.py` ✂️ | `SemanticChunker` (Ollama embeddings, percentile breakpoints) produces **~19 large, topic-coherent chunks** instead of hundreds of tiny ones. | Chunk size is a *quality* decision, not just a size decision. |
| `test_rag_retrieval.py` 🔎 | `similarity_search("What is the diagonal of the SONY IMX415…?", k=3)`, the top passages are stuffed into a prompt, the chat model answers from context. | The **RAG in a sentence**: retrieve → augment → generate. |
| `main.py` 🤖 | `create_agent` with two `@tool`s — `sensor_spec_search` (Chroma similarity search over the datasheet) and `license_plate_list` (hard-coded alarm records). Three questions: one routes to each tool plus an **agent-memory** question that reuses previous answers. | The model is the router: one agent, two knowledge sources, zero branching code. |

### 📁 langgraph_intro

| 📁 Module | What's inside | 🎯 Takeaway |
| :-- | :-- | :-- |
| `01_langgraph_basics.py` 🕸️ | A `TypedDict` state, one node, `START → welcome → END`, compiled and invoked. | The **hello world**: graph, node, edge, state — nothing more. |
| `02_langgraph_schema_control.py` 📐 | State is a **Pydantic model** (`topic / post / processed_post`) with field descriptions instead of a dict. | Type your state so nodes can't silently pass garbage to each other. |
| `03_langgraph_handling_messages.py` 💬 | Three message lists in one state: manual append **vs** `Annotated[list, add]` **reducer** auto-append. | Reducers *merge* updates instead of overwriting them — how conversations accumulate. |
| `04_langgraph_prompts_chains.py` 🔗 | Scratchpad: `ChatPromptTemplate` built from messages, LCEL pipe `prompt | client`. | Prompt templates belong *outside* the graph, where they're reusable. |
| `05_langgraph_react.py` 🕵️ | A **hand-rolled ReAct loop**: `agent_node` (one model call) → conditional edge → `tool_node` (executes tool calls, appends `ToolMessage`s) → back to the agent, until it stops calling tools. Tools: search the **INSTAR camera wiki** via its Elasticsearch API + a URL-restricted page scraper. Built on the `AgentState` subclass. | See the agent loop with no framework magic: `bind_tools`, `tool_calls`, `tool_call_id` — and why errors get fed *back* to the model instead of crashing. |
| `06_langgraph_parallelization.py` ⚡ | `START` fans out to **three nodes at once** (same topic → three platform-flavoured posts), results land in separate state keys. | One edge per branch = parallelism for free. |
| `07_langgraph_routing.py` 🔀 | A `with_structured_output(...)` **decider node** picks `ig / x / li` (a `Literal` field), a conditional edge routes to the matching post node. | Structured output **is** control flow when you wire it to conditional edges. |
| `08_langgraph_ochestrator_worker.py` 🏭 | Orchestrator decomposes a query into a `list[str]` of task prompts (structured output) → one **`ThreadPoolExecutor`** worker per task → collector summarizes all results. | Multi-agent is just: *fan out work, collect it, and have a state schema carry everything*. |
| `09_langgraph_generator_evaluator.py` 🎭 | Generator writes a joke, evaluator (structured `funny: bool` + `feedback`) grades it, generator revises — looping until "funny" or **5 iterations max**. | Self-critique loop with an **iteration cap** — the loop *must* terminate. |
| `10_langgraph_memory.py` 🧠 | Same graph compiled with `InMemorySaver`; two invocations on the same `thread_id` → the second one **remembers** the first. (State key must be named `messages` for the `add_messages` reducer to kick in.) | Memory is a **checkpointer + thread id**, not a magic flag. |
| `11_langgraph_human-in-the-loop.py` 🙋 | An `approval` node calls **`interrupt()`** — the graph *pauses* and hands the question to you via `input()`. Answer `y/n`, resume with `Command(resume=...)` on the same `thread_id`, and the graph branches accordingly. | The graph can **stop and wait for a human** — approval flows, not just chat. |

---

## 🧰 Run it

Every folder is its own `uv` **package** (**Python 3.14**, a `uv.lock` in each).
They all assume a **local Ollama** serving:

* **`qwen3.8:latest`** — the chat model
* **`qwen3-embedding:0.6b`** — the embedding model (`rag_intro`)

…on the OpenAI-compatible endpoint `http://localhost:11434/v1` — that's hardcoded
in each module, so to point anywhere else just edit those kwargs.

```sh
# once, so Ollama is answering on :11434
ollama serve              # if it isn't already running
ollama pull qwen3.8:latest
ollama pull qwen3-embedding:0.6b
```

### rag_intro (suggested order)

The `test_*` scripts use **CWD-relative paths** (`src/rag_intro/...`) — run them
**from inside the folder**.

```sh
cd rag_intro
uv sync                   # fills this folder's own .venv

uv run python src/rag_intro/test_api_calls.py         # 1. raw API call, no LangChain
uv run python src/rag_intro/test_manual_embedding.py  # 2. PDF → chunks → vectors (manual)
uv run python src/rag_intro/create_db.py              # 3. build the Chroma DB
#    …or: uv run python src/rag_intro/create_db_symantic.py   (semantic chunking variant)
uv run python src/rag_intro/test_rag_retrieval.py     # 4. retrieve + answer from context
uv run python src/rag_intro/main.py                   # 5. the agent: RAG + plate tools
```

### langgraph_intro

The modules are run as modules (01–04 & 10 also execute on import, so both styles
work).

```sh
cd langgraph_intro
uv sync

# in learning order:
uv run python -m langgraph_intro.01_langgraph_basics
uv run python -m langgraph_intro.02_langgraph_schema_control
uv run python -m langgraph_intro.03_langgraph_handling_messages
uv run python -m langgraph_intro.04_langgraph_prompts_chains
uv run python -m langgraph_intro.05_langgraph_react          # interactive: asks you a question
uv run python -m langgraph_intro.06_langgraph_parallelization
uv run python -m langgraph_intro.07_langgraph_routing
uv run python -m langgraph_intro.08_langgraph_ochestrator_worker
uv run python -m langgraph_intro.09_langgraph_generator_evaluator
uv run python -m langgraph_intro.10_langgraph_memory
uv run python -m langgraph_intro.11_langgraph_human-in-the-loop   # interactive: y/n prompt
```

> 💡 The `[project.scripts]` entry points (`rag_intro`, `langgraph-intro`) point at a
> leftover `main()` stub in `__init__.py` that only prints "Hello from…" — run the real
> experiments as shown above.

**Notes:**

* ⚠️ `05_langgraph_react` needs **real outbound internet** to the INSTAR wiki
  (`search.instar.wiki` + `wiki.instar.com`). The scraper only touches
  `https://wiki.instar.com/` URLs by construction.
* `create_db*.py` (re)writes `src/rag_intro/db/` — the persisted vector DB is the
  same directory `test_rag_retrieval.py` and `main.py` read from.
* The models are local and non-deterministic; "funny" is whatever model 09's
  evaluator says it is today.
