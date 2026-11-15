import json
import logging

from uuid import uuid4

from langchain_core.messages import HumanMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from .graph import build_graph

logger = logging.getLogger(__name__)

graph = build_graph()

langfuse = get_client()
user = "Mike Polinowski"

async def execute_chat(
    question: str,
    username: str = user,
    session_id: str | None = None,
) -> dict:

    session_id = session_id or str(uuid4())

    # Create a handler for this graph invocation
    langfuse_handler = CallbackHandler()

    with propagate_attributes(
        trace_name="LangGraph Hello World",
        user_id=username,
        session_id=session_id,
        tags=["hello-world", "ollama", "langgraph"],
        metadata={
            "model": "local-ollama",
            "environment": "production",
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

async def stream_chat(
    question: str,
    username: str = user,
    session_id: str | None = None,
):
    session_id = session_id or str(uuid4())

    langfuse_handler = CallbackHandler()

    try:
        with propagate_attributes(
            trace_name="LangGraph Hello World",
            user_id=username,
            session_id=session_id,
            tags=["hello-world", "ollama", "langgraph"],
            metadata={
                "model": "local-ollama",
                "environment": "production",
            },
        ):
            async for chunk, metadata in graph.astream(
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
                stream_mode="messages",
            ):
                # only process messages from chatbot node
                if metadata.get("langgraph_node") != "chatbot":
                    continue

                content = chunk.content

                if content:
                    yield (
                        "event: token\n"
                        f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                    )

        yield(
            "event: done\n"
            f"data: {json.dumps({'session_id': session_id})}\n\n"
        )

    except Exception:
        logger.exception(
            "Graph streaming failed for session %s",
            session_id,
        )

        yield(
            "event: error\n"
            f"data: {json.dumps({'error': 'Graph execution failed'})}\n\n"
        )
