import os
from pathlib import Path

from dotenv import load_dotenv
from httpx import Timeout
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# Ollama's httpx default is timeout=None (unbounded). Without a connect
# timeout, an unreachable Ollama (firewalled, typo'd host, ...) stalls on the
# TCP handshake until the kernel gives up: ~2 minutes here, longer elsewhere.
# The read timeout is deliberately generous: Ollama sends no bytes until the
# generation is finished, so for a blocking /invoke call the read timeout is
# the cap on generation time. On a slow GPU, 600s of generation should still
# succeed.
MODEL_TIMEOUT = Timeout(connect=10, read=600, write=30, pool=10)

model = ChatOllama(
  model=os.getenv("OLLAMA_MODEL", "qwen3.8:latest"),
  base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
  temperature=0.6,
  client_kwargs={"timeout": MODEL_TIMEOUT},
)
