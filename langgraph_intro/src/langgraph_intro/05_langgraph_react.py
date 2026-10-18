import json
import re
from typing import Annotated, TypedDict
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

API_BASE = "https://search.instar.wiki/wiki_en/_search"
WIKI_BASE = "https://wiki.instar.com/en"

# The state holds every message type the loop produces (Human, AI, Tool),
# so we type it as AnyMessage instead of HumanMessage.
# The add_messages reducer is essential: without it create_agent
# compiles "messages" as a LastValue channel and each node's update REPLACES
# the whole list, so the ReAct loop loses context between steps.
class graph_schema(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

client = ChatOpenAI(
    model="qwen3.8:latest",
    temperature=0.6,
    top_p=0.95,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# The Prompt Template

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the INSTAR camera product line. "
    "Use the search tool to find relevant English wiki pages. If the descriptions are not "
    "detailed enough, use the scrape tool on one of the returned URLs to read the full page. "
    "Base your final answer ONLY on the information you retrieved, and cite the URLs you used."
    "Do not use any Markdown formatting in your response."
)


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping script/style blocks."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "template"}

    # Sets up a counter for how many "skip" blocks (script/style/etc.) we are
    # currently inside, plus the empty list that will collect visible text pieces.
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    # Called by HTMLParser for every opening tag; if it's one we want to skip
    # (like <script> or <style>), bump the counter so that block's text is ignored.
    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    # Called by HTMLParser for every closing tag; if it closes a "skip" block,
    # lower the counter again so normal text resumes (a counter, not a boolean,
    # also handles nested skip tags).
    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    # Called by HTMLParser for each chunk of plain text; if we're not inside a
    # skipped block, keep the trimmed text (empty chunks are dropped).
    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


def _http_get(url: str, timeout: int = 20) -> str:
    """Fetch a URL and return the body as text (our 'curl' call)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (INSTAR-Intro/0.5)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


@tool
def search_instar_wiki(query: str, limit: int = 5) -> str:
    """Search the English INSTAR camera wiki via the official Elasticsearch API.

    Takes a natural-language or keyword query and returns the top matching
    pages, each with: title, relevance score, a short description and the
    full page URL. Use the scrape_wiki_page tool on a URL if you need
    more detail than the description provides.
    """
    try:
        url = f"{API_BASE}?q={urllib.parse.quote(query)}"
        data = json.loads(_http_get(url))
        hits = data["hits"]["hits"][:limit]
        if not hits:
            return f"No wiki pages found for: {query}"

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            sublink = source.get("sublink1")
            results.append(
                {
                    "title": source.get("title"),
                    "score": round(hit.get("_score") or 0.0, 3),
                    "description": (source.get("description") or "")[:700],
                    # sublink1 already starts with "/"
                    "url": WIKI_BASE + sublink if sublink else None,
                }
            )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:  # surface a readable error to the agent
        return f"Search failed: {exc}"


@tool
def scrape_wiki_page(url: str) -> str:
    """Fetch a wiki.instar.com page and extract its readable text.

    Only use URLs that search_instar_wiki returned. Returns the page text
    with HTML stripped out (truncated if the page is very long).
    """
    if not url.startswith("https://wiki.instar.com/"):
        return "Refused: this tool only allows scraping pages from https://wiki.instar.com/"
    try:
        html = _http_get(url)
        extractor = _TextExtractor()
        extractor.feed(html)
        text = re.sub(r"\s{2,}", " ", " ".join(extractor.parts))
        return text[:5000] + ("... [truncated]" if len(text) > 5000 else "")
    except Exception as exc:
        return f"Scrape failed: {exc}"


if __name__ == '__main__':

    # ReAct agent: handles the tool loop (search -> maybe scrape -> answer)
    # and finishes with a final answer based on the retrieved information.
    agent = create_agent(
        client,
        tools=[search_instar_wiki, scrape_wiki_page],
        system_prompt=SYSTEM_PROMPT,
        state_schema=graph_schema,
    )
    print("INSTAR WIKI SEARCH AGENT\n")
    print("-----------------------------------------------------------------------------------\n")
    print("Please type your question below, e.g. 'How do I configure the INSTAR MQTT Broker?':\n")
    query = input()
    print("===================================================================================\n")
    # Wrap the raw string in a HumanMessage so the state only ever carries
    # typed message objects (Human/Tool/AI), as graph_schema promises.
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(result["messages"][-1].content)
