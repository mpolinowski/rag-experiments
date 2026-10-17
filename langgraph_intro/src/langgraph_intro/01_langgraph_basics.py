from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# # DEBUG
# print(client.invoke("Tell me an Haiku about the molecular structure of SiO2 (Quartz)").content)
# # 'Silicon holds four\noxygen bridges each pair—\nan endless lattice'

# Define the schema

class graph_shema(TypedDict):
    name: str
    message: str

# Create Node functions

def welcome(state: graph_shema) -> graph_shema:
    current_name = state['name']
    current_message = state['message']

    resonse = client.invoke(f"My name is {current_name}. {current_message}").content

    state['message'] = f"You asked {current_message}. Here is the response: {resonse}."

    return state

# Create the State Graph

## Adding Nodes
graph = StateGraph(graph_shema)
graph.add_node("welcome", welcome)
## Adding Edges
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)
graph = graph.compile()

# Run Graph

response = graph.invoke({"name": "Mike", "message": "Hey there!"}) # expects graph_shema!

print(response)
