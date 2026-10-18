from pydantic import BaseModel, Field, SecretStr
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


class model_schema(BaseModel):
    tasks: list[str] = Field(..., description="Evaluate the user query to generate this list of invidiual tasks to be performed by the worker.")

class graph_schema(TypedDict):
    tasks: list[str]
    query: str
    results: list[str]
    summary: str

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
)

client_with_structured_output = client.with_structured_output(model_schema)


def orchestrator_node(state: graph_schema) -> graph_schema:
    user_query = state["query"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an orchestrator that breaks down a user query into tasks for the worker."),
        ("user", f"User query: {user_query}. Please generate one prompt per task for the worker to complete.")
    ])

    chain = prompt | client_with_structured_output

    task_list = chain.invoke({"query": user_query})

    state["tasks"] = task_list.tasks

    return state


def execute(query: str):
    response = client.invoke(f"Please execute this task: {query}")

    return response.content


def worker_node(state: graph_schema) -> graph_schema:

    tasks = state["tasks"]
    results = []

    # spawn as many workers as there are tasks
    with ThreadPoolExecutor(max_workers=len(tasks)) as executer:
        # and divide the lists of tasks on your workers
        results_futures = (executer.map(execute, tasks))
        # once resolved append to results
        for result in results_futures:
            results.append(result)

    state["results"] = results

    return state


def collector_node(state: graph_schema):

    results = state["results"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a collector that summarized the results from the worker."),
        ("user", f"Here are the results from the worker: {results}. Please provide an easy to read, concise and structured summary of the results.")
    ])

    chain = prompt | client

    summary = chain.invoke({"results": results})

    state["summary"] = str(summary.content)

    return state

def build_state_graph():
    orchestrator_graph = StateGraph(graph_schema)

    orchestrator_graph.add_node("orchestrator_node", orchestrator_node)
    orchestrator_graph.add_node("worker_node", worker_node)
    orchestrator_graph.add_node("collector_node", collector_node)

    orchestrator_graph.add_edge(START, "orchestrator_node")
    orchestrator_graph.add_edge("orchestrator_node", "worker_node")
    orchestrator_graph.add_edge("worker_node", "collector_node")
    orchestrator_graph.add_edge("collector_node", END)

    return orchestrator_graph.compile()


if __name__ == '__main__':
    orchestrator_graph = build_state_graph()
    response = orchestrator_graph.invoke({
        "query": "Give me the population numbers of the following cities: New York, Shanghai, Sydney, Hongkong, Berlin",
        "tasks": [],
        "results": [],
        "summary": ""
    })

    # # DEBUG
    # for chunk in orchestrator_graph.stream({
    #         "query": "Give me the population numbers of the following cities: New York, Shanghai, Sydney, Hongkong, Berlin",
    #         "tasks": [],
    #         "results": [],
    #         "summary": ""
    #     },
    #     stream_mode="updates"
    # )

    print(response["tasks"])
    print(response["summary"])
