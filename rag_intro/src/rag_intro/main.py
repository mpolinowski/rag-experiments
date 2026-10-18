from pathlib import Path
from pydantic import SecretStr

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
)

# --- 1. Vector DB (RAG over your Chroma embeddings) -------------------------

db_path = Path(__file__).resolve().parent.parent / "rag_intro" / "db"
db_path.parent.mkdir(parents=True, exist_ok=True)

embed_model = OpenAIEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
    check_embedding_ctx_length=False
)

db_con = Chroma(persist_directory=str(db_path), embedding_function=embed_model)


# --- 2. Tools ----------------------------------------------------------------

@tool
def license_plate_list(plate: str) -> str:
    '''Use this tool when you need to answer a question about when a
    specified registration number / license plate was last detected.
    Do NOT use it for questions about camera or sensor specifications.
    Args:
        plate (str): The license plate number of the car to look up.
    Returns:
        str: A string containing information of the corresponding alarm
        event - containing the time stamp of the event and the alarm trigger.
    '''
    # Hard-coded info - simulating SQL retrieved data
    records = [{
        "plate": "K:FG:123",
        "timestamp": "Wed Oct 14 08:17:05 PM HKT 2026",
        "trigger": "Alarm triggered by Motion area 1"
    },{
        "plate": "K:SX:712",
        "timestamp": "Wed Oct 14 12:01:45 PM HKT 2026",
        "trigger": "Alarm triggered by Motion area 3"
    }]

    for record in records:
        if record["plate"].lower() == plate.lower():
            return (
                f"(License plate {record['plate']} last seen at "
                f"{record['timestamp']}. {record['trigger']}.)"
            )

    known = ", ".join(r["plate"] for r in records)
    return (
        f"(License plate {plate} was not found in recent alarm events. "
        f"Plates on file: {known}.)"
    )


@tool
def sensor_spec_search(query: str) -> str:
    '''Use this tool when a question is asked about the technical
    specifications of the SONY IMX415-AAQR STARVIS sensor, e.g. its size
    or diagonal, resolution, readout, frame rate, sensitivity, or lens
    mount. It searches the sensor datasheet in the Chroma vector database.
    Do NOT use it for questions about license plates or alarm events.
    Args:
        query (str): The specification question or the property the user
        is asking about.
    Returns:
        str: The relevant datasheet passages retrieved from the vector DB.
    '''
    chunks = db_con.similarity_search(query, k=3)

    if not chunks:
        return "(No relevant passages found in the sensor datasheet database.)"

    passages = [
        f"Passage {i + 1}:\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    ]
    return "(Datasheet passages retrieved from the vector DB:\n\n" + "\n\n".join(passages) + ")"


toolkit = [
    license_plate_list,
    sensor_spec_search,
]


# --- 3. Agent loop: execute tool calls -----------------------------
# bind_tools() only *announces* the tool to the model. create_agent wraps
# the model + tools in a loop: model -> tool_calls -> tool run -> ToolMessage
# -> final answer. Without this, a model tool call is never executed and
# its result never fed back.

agent = create_agent(client, toolkit)

agent_memory = []

def ask(question: str) -> str:
    result = agent.invoke({"messages": [("user", question)]})
    message = result["messages"][-1].content
    # # DEBUG
    # for m in result["messages"]:
    #     print(m)

    agent_memory.append(message)

    print("\nModel Response:")
    print(message)
    print("=" * 72)
    return message


print("Q1 (routes to Chroma tool): What is the diagonal of the SONY IMX415-AAQR STARVIS sensor?\n")
ask("What is the diagonal of the SONY IMX415-AAQR STARVIS sensor?")

print("\nQ2 (routes to plate tool): When was the car with license plate K:SX:712 last seen?\n")
ask("When was the car with license plate K:SX:712 last seen?")

print("\nQ3 (agent memory): What was the license plate of the car that triggered an alarm on Wednesday, October 14, 2026, at 12:01:45 PM HKT?\n")
ask(f"What was the license plate of the car that triggered an alarm on Wednesday, October 14, 2026, at 12:01:45 PM HKT? You can refer to the previous conversation: {agent_memory}")

print("\nQ4 (agent memory): Can you summarize the conversation?\n")
ask(" ".join(str(memory) for memory in agent_memory))
