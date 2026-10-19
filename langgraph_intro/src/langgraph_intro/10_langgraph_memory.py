from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama")
)

# Define the schema

class graph_shema(TypedDict):
    messages: Annotated[list[str], add_messages] # <- HAS TO BE CALLED MESSAGES, otherwise memory does not work...

# Create Node functions

def welcome(state: graph_shema) -> graph_shema:
    current_message = state['messages']

    resonse = client.invoke(current_message).content

    state['messages'] = [f"You asked {current_message}. Here is the response: {resonse}."]

    return state

# Create the State Graph with Memory Checkpoint

memory_graph = StateGraph(graph_shema)
checkpoint = InMemorySaver()

memory_graph.add_node("welcome", welcome)

memory_graph.add_edge(START, "welcome")
memory_graph.add_edge("welcome", END)
graph = memory_graph.compile(checkpointer=checkpoint)

# Run Graph with configuration

response = graph.invoke(
    {"messages": ["Hey there! My name is Mike. Who are you?"]},
    {"configurable":  {"thread_id": "this-is-a-memory-test"}}
)

for message in response['messages']:
    message.pretty_print()

response2 = graph.invoke(
    {"messages": ["Hey there! Do you know who I am?"]},
    {"configurable":  {"thread_id": "this-is-a-memory-test"}}
)

for message in response2['messages']:
    message.pretty_print()
