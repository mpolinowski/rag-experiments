from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Create the Prompt Template

my_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps to create and curate posts for a social media platform."),
        ("human", "I want to create a post about {topic}.")
    ]
)


# DEBUG
chain = my_prompt | client
print(chain.invoke({"topic": "the movie Alien that reads like a joke and makes senior AI engineers laugh out loud"}).content)
