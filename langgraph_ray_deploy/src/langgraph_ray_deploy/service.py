from uuid import uuid4

from langchain_core.messages import HumanMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from .graph import graph


langfuse = get_client()
id = str(uuid4())
user = "Mike Polinowski"

async def execute_chat(
    question: str,
    username: str = user,
    session_id: str = id,
) -> dict:

    # Create a handler for this graph invocation
    langfuse_handler = CallbackHandler()

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

        response = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={
                "callbacks": [langfuse_handler],
                "run_name": "chatbot-execution",
                "configurable": {
                    "thread_id": session_id,
                },
            },
        )

    ai_message = response["messages"][-1]

    usage = ai_message.usage_metadata or {}

    return {
        "answer": ai_message.content,
        "model": ai_message.response_metadata.get(
            "model", "Unknown model"
        ),
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "session_id": session_id,
    }
