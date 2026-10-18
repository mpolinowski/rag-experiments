from pydantic import BaseModel, Field
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
)

# Create structured Schema

class graph_shema(BaseModel):
    topic: str = Field(description="The topic of the post")
    post: str = Field(description="The social media post content")
    processed_post: str = Field(description="The curated social media posts content")

demo = graph_shema(
    topic="AI Coding Language Benchmarks",
    post="",
    processed_post=""
)

# Create Node functions

def generate_post(state: graph_shema) -> graph_shema:
    # optionally convert pydantic model to dictionary
    # state_dict = state.model_dump()
    # topic = state_dict["topic"]

    topic = state.topic
    # passing topic to model to generate post
    post = client.invoke(f"Write a social media post about {topic}").content

    # Update state with model response
    state.post = str(post)

    return state

def process_response(state: graph_shema) -> graph_shema:
    post = state.post

    processed_post = client.invoke(f"Re-write the following scial media post in a West-Coast Hip Hop tone: {post}").content

    state.processed_post = str(processed_post)

    return state

# Create the State Graph

## Adding Nodes
graph = StateGraph(graph_shema)
graph.add_node("generate_post", generate_post)
graph.add_node("process_response", process_response)
## Adding Edges
graph.add_edge(START, "generate_post")
graph.add_edge("generate_post", "process_response")
graph.add_edge("process_response", END)
graph = graph.compile()

# Run Graph

response = graph.invoke(demo) # expects graph_shema!

print(response)
