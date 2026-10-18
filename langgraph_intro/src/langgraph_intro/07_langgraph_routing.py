from pydantic import BaseModel, Field, SecretStr
from typing import Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class model_schema(BaseModel):
    target: Literal['ig', 'x', 'li'] = Field(..., description="The best target social network to which to generate the post for based on the user query - Instagram, Twitter/X or LinkedIn")
    topic: str = Field(..., description="The topic for the post to generate.")

class graph_schema(TypedDict):
    target: str
    input: str
    topic: str
    post: str

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
)

client_with_structured_output = client.with_structured_output(model_schema)

# # DEBUG the decision model
# print(client_with_structured_output.invoke("Tell me about the the advantages of the Go programming language over Rust"))
# print(client_with_structured_output.invoke("Write a post about how sick 90s and early 2000s sitcoms were - contrast the differences between Love and a Mariage and the show Friends"))
# print(client_with_structured_output.invoke("A short post about me being on vacation sitting on the beach enjoying the sun"))

# # python -m langgraph_intro.07_langgraph_routing
# # target='x' topic='programming_languages'
# # target='li' topic='90s and early 2000s sitcoms were sick - contrasting Love and a Marriage with Friends'
# # target='x' topic='A short post about me being on vacation sitting on the beach enjoying the sun'

def decider_node(state: graph_schema) -> graph_schema:
    user_query = state["input"]

    target_decision = client_with_structured_output.invoke(user_query)

    state["target"] = target_decision.target
    state["topic"] = target_decision.topic

    return state

def condition(state: graph_schema) -> str:
    target = state["target"]

    if target == "ig":
        return 'trigger_create_post_ig'
    elif target == "x":
        return 'trigger_create_post_x'
    elif target == "li":
        return 'trigger_create_post_li'
    else:
        raise ValueError("Invalid target")

def create_post_ig(state: graph_schema) -> dict:

    input = state["input"]
    post = client.invoke(f"Write a LinkedIn post about {input}. Keep the tone Gen-z casual and engaging").content
    state["post"] = str(post)

    return {"post": post}

def create_post_x(state: graph_schema) -> dict:

    input = state["input"]
    post = client.invoke(f"Write a LinkedIn post about {input}. Keep the tone like a mid-nineties action movie.").content
    state["post"] = str(post)

    return {"post": post}

def create_post_li(state: graph_schema) -> dict:

    input = state["input"]
    post = client.invoke(f"Write a LinkedIn post about {input}. Keep the tone smart and business-like").content
    state["post"] = str(post)

    return {"post": post}

def build_state_graph():
    routed_graph = StateGraph(graph_schema)

    routed_graph.add_node("decider_node", decider_node)
    routed_graph.add_node("create_post_ig", create_post_ig)
    routed_graph.add_node("create_post_x", create_post_x)
    routed_graph.add_node("create_post_li", create_post_li)

    routed_graph.add_edge(START, "decider_node")
    routed_graph.add_conditional_edges("decider_node", condition, {
        "trigger_create_post_ig": "create_post_ig",
        "trigger_create_post_x": "create_post_x",
        "trigger_create_post_li": "create_post_li"
    })
    routed_graph.add_edge("create_post_ig", END)
    routed_graph.add_edge("create_post_x", END)
    routed_graph.add_edge("create_post_li", END)

    return routed_graph.compile()


if __name__ == '__main__':
    parallel_graph = build_state_graph()
    response = parallel_graph.invoke({
        "input": "Write a post about how sick 90s and early 2000s sitcoms were - contrast the differences between Love and a Mariage and the show Friends",
        "target": "",
        "topic": "",
        "post": ""
    })

    print(response["topic"])
    print(response["post"])
