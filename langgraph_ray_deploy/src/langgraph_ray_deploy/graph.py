from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from langfuse import get_client,  propagate_attributes

from .config import model

# load system prompt -> removed and replaced by dynamic prompt via Langfuse Prompt Management
# SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")
langfuse = get_client()


# define graph node
def build_graph(checkpointer=None):
    async def chatbot(state: MessagesState):

        langfuse_prompt = langfuse.get_prompt(
            name="langgraph-ray-serve-prompt",
            type="text",
            label="production",
            cache_ttl_seconds=300,
        )

        system_prompt = langfuse_prompt.get_langchain_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]

        # surface the prompt's experiment tag so it's filterable in the
        # Langfuse UI; a missing/non-dict config degrades to "default"
        prompt_config = langfuse_prompt.config if isinstance(langfuse_prompt.config, dict) else {}

        # associate the generation with the LangFuse prompt version
        with propagate_attributes(
            prompt=langfuse_prompt,
            metadata={"prompt_experiment": prompt_config.get("prompt_experiment", "default")},
        ):
            response = await model.ainvoke(messages)

        return {"messages": [response]}

    # build graph
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    return builder.compile(checkpointer=checkpointer)
