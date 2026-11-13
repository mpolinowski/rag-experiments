import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

model = ChatOllama(
  model=os.getenv("OLLAMA_MODEL", "qwen3.8:latest"),
  base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
  temperature=0.6,
)
