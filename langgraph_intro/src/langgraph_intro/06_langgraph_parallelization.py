from typing import TypedDict
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class graph_schema(TypedDict):
    topic: str
    ig: str
    x: str
    li: str

demo_query: graph_schema = {
    "topic": "a very concise description of the PC game Unreal Tournament from 1999 by the campany Epic",
    "ig": "",
    "x": "",
    "li": "",
}

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
)

def create_post_ig(state: graph_schema) -> dict:

    topic = state["topic"]
    post = client.invoke(f"Write a LinkedIn post about {topic}. Keep the tone Gen-z casual and engaging").content
    state["ig"] = str(post)

    return {"ig": post}

def create_post_x(state: graph_schema) -> dict:

    topic = state["topic"]
    post = client.invoke(f"Write a LinkedIn post about {topic}. Keep the tone like a mid-nineties action movie.").content
    state["x"] = str(post)

    return {"x": post}

def create_post_li(state: graph_schema) -> dict:

    topic = state["topic"]
    post = client.invoke(f"Write a LinkedIn post about {topic}. Keep the tone smart and business-like").content
    state["li"] = str(post)

    return {"li": post}

def build_state_graph():
    graph = StateGraph(graph_schema)

    graph.add_node("create_post_ig", create_post_ig)
    graph.add_node("create_post_x", create_post_x)
    graph.add_node("create_post_li", create_post_li)

    graph.add_edge(START, "create_post_ig")
    graph.add_edge(START, "create_post_x")
    graph.add_edge(START, "create_post_li")

    return graph.compile()


if __name__ == '__main__':
    parallel_graph = build_state_graph()
    response = parallel_graph.invoke(demo_query)

    print(response["ig"])
    print(response["x"])
    print(response["li"])
