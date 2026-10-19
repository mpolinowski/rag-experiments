from typing import Annotated, Literal, Optional, TypedDict
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

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

class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]

# Create Node functions

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
   decision = interrupt({
   "question": "Do you approve this action?",
   "details": state["action_details"],
   })

   return Command(goto="proceed" if decision else "cancel")

def proceed_node(state: ApprovalState):
    return {"status": "approved"}

def cancel_node(state: ApprovalState):
    return {"status": "rejected"}

approval_graph = StateGraph(ApprovalState)
checkpoint = MemorySaver()

approval_graph.add_node("approval", approval_node)
approval_graph.add_node("proceed", proceed_node)
approval_graph.add_node("cancel", cancel_node)

approval_graph.add_edge(START, "approval")
approval_graph.add_edge("proceed", END)
approval_graph.add_edge("cancel", END)

graph = approval_graph.compile(checkpointer=checkpoint)
config = {"configurable": {"thread_id": "approval-flow"}}

# Run Graph with configuration

initial = graph.invoke(
    {"action_details": "Send sumary by Email?", "status": "pending"},
    config=config,
)

# The graph is now PAUSED at the interrupt: nothing after "approval" has run.
# The pending question lives in state, keyed by "__interrupt__":
pending = initial["__interrupt__"][0].value
print(pending)  # {'question': 'Do you approve this action?', 'details': 'Send sumary by Email?'}

# --- THIS is the human in the loop: stop the process and wait for a real answer.
answer = input(f"{pending['question']} ({pending['details']}) [y/n] ").strip().lower()
resume_value = answer in ("y", "yes")

# Resuming on the SAME thread_id replays the interrupted node;
# interrupt() returns whatever we resume with, so the branch follows the answer.
resumed = graph.invoke(
    Command(resume=resume_value),
    config=config,
)

print(resumed["status"])
