from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from .config import model

# load system prompt
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")

# define graph node
def chatbot(state: MessagesState):
  messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    *state["messages"],
  ]

  response = model.invoke(messages)

  return {"messages": [response]}

# build graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()
