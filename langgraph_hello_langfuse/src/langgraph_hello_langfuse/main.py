import uuid

from langchain_core.messages import HumanMessage
from langfuse import propagate_attributes
from langfuse._client.get_client import get_client
from langfuse.langchain import CallbackHandler

from .graph import graph

langfuse = get_client()
langfuse_handler = CallbackHandler()

def main():
  session_id = str(uuid.uuid4())
  print("-------------------------------")
  print("LOGIN")
  username = input("\nUser Name: ")
  print(f"\nHi {username}! Your chat session ID is {session_id}")
  print("-------------------------------")

  while True:
    question = input(f"\nAsk your question: ")

    if question.lower() in ["exit", "quit"]:
      break

    with propagate_attributes(
        trace_name="LangGraph Hello World",
        user_id=username,
        session_id=session_id,
        tags=["hello-world", "ollama", "langgraph"],
        metadata={
        "model": "local-ollama",
        "environment": "development",
        },
    ):

        response = graph.invoke(
            {
            "messages": [
                HumanMessage(content=question)
            ]
            },
            config = {
                "callbacks": [langfuse_handler],
                "run_name": "chatbot-execution",
            },
        )

    ai_message = response["messages"][-1]

    usage = ai_message.usage_metadata or {}
    model = ai_message.response_metadata.get('model', 'Unknown model')
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    print(f"\nAssistant: {ai_message.content}")

    # DEBUG
    print("----------------------------------")
    print(f"{model} token usage: {input_tokens}, {output_tokens}, {total_tokens}")
    print("----------------------------------\n")

  # send telemetry
  langfuse.flush()

if __name__ == "__main__":
  main()
