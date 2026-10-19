from pydantic import BaseModel, Field, SecretStr
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


class model_schema(BaseModel):
    funny: bool = Field(..., description="True if the generated joke is funny, False if it is not.")
    feedback: str = Field(default="", description="Concrete feedback on how to make the joke funnier, or an empty string if it is already funny.")

class graph_schema(TypedDict):
    topic: str
    joke: str
    funny: bool
    feedback: str
    iterations: int

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
)

client_with_structured_output = client.with_structured_output(model_schema)


def generator_node(state: graph_schema) -> graph_schema:

    topic = state["topic"]

    if state['feedback']:
        response = client.invoke(f"Please modify the following joke: {state['joke']} based on the evaluators feedback: {state['feedback']}")
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a joke creator and will be evaluated on your joke quality. You will receive a topic and then generate a joke around it."),
            ("user", f"Here is the topic: {topic}. Please create a funny joke about it.")
        ])

        chain = prompt | client
        response = chain.invoke({"topic": topic})

    state["joke"] = str(response.content)

    return state

def evaluator_node(state: graph_schema) -> graph_schema:

    joke = state["joke"]
    iteration = state['iterations']
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a comedy critic. Your job is to evaluat the following joke and provide feedback on how to make it funnier."),
        ("user", f"Evaluate the following joke: {joke}. Respond with True if it is funny and False if it is not and provide feedback if it's not funny.")
    ])


    chain = prompt | client_with_structured_output
    evaluation = chain.invoke({"joke": joke})
    state["funny"] = evaluation.funny
    state["feedback"] = evaluation.feedback
    state["iterations"] = iteration + 1

    return state

def stop(state: graph_schema) -> str:
    if state["funny"] or state["iterations"] >=5:
        return "end"
    else:
        return "evaluator_node"


def build_state_graph():
    evaluator_graph = StateGraph(graph_schema)

    evaluator_graph.add_node("generator_node", generator_node)
    evaluator_graph.add_node("evaluator_node", evaluator_node)

    evaluator_graph.add_edge(START, "generator_node")
    evaluator_graph.add_conditional_edges("generator_node", stop, {
        "evaluator_node": "evaluator_node",
        "end": END
    })
    evaluator_graph.add_edge("evaluator_node", "generator_node")
    evaluator_graph.add_edge("generator_node", END)

    return evaluator_graph.compile()


if __name__ == '__main__':
    evaluator_graph = build_state_graph()
    response = evaluator_graph.invoke({
        "topic": "The Rust programming language",
        "joke": "",
        "funny": False,
        "feedback": "",
        "iterations": 0,
    })

    print(response["funny"])
    print(response["joke"])
