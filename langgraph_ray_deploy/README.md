# langgraph-ray-deploy

A LangGraph chatbot service deployed with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), using [Ollama](https://ollama.com) as the local LLM and [Langfuse](https://langfuse.com) for tracing and prompt management.

The Ray Serve deployment serves two endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check |
| `POST /invoke` | Send a message, get the model's answer plus token usage |
| `POST /stream` | Send a message, receive the answer token-by-token as Server-Sent Events |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for managing the Python environment
- [Ollama](https://ollama.com) running (defaults to `http://localhost:11434`) with the chat model pulled, e.g. `ollama pull qwen3.8:latest`
- [Langfuse](https://langfuse.com) (self-hosted or cloud) with a text prompt named `langgraph-ray-serve-prompt` under the label `production` (see [Troubleshooting](#troubleshooting))

## Setup

```bash
# 1. Create the venv and install all dependencies
uv sync

# 2. Create your local config from the example
cp example.env .env
#    then edit .env with your Ollama setup and Langfuse keys
```

`uv sync` creates `.venv` inside this folder and installs the project into it
in editable mode. That editable install is what makes
`import langgraph_ray_deploy` work — the project uses a `src/` layout, so the
package is **not** importable just by running from the project directory.
That's why everything below goes through the venv.

## Running

```bash
uv run python -m langgraph_ray_deploy.main
```

or equivalently, via the console entry point:

```bash
uv run langgraph-ray-deploy
```

Both start a local Ray Serve cluster (2 replicas) at
`http://localhost:8000` and keep running until you stop them with `Ctrl+C`.
Good for trying things locally; for a persistent service, run it in Docker
(below).

## Running in Docker (no systemd required)

The container *is* the service: the entrypoint is the blocking
`serve.run(...)`, so the container stays up while the app is up, and dies —
with docker restarting it (`restart: unless-stopped`) — if the app dies.
No supervisor, no systemd:

```bash
# build and start (needs the local python:3.14-slim image and
# .env created from example.env -- see Setup)
docker compose up -d

# day-to-day controls
docker compose logs -f api            # live logs
docker compose up -d --build          # pick up code changes
docker compose down                   # stop
```

Networking and firewall:

- The container reaches Ollama (`:11434`) and Langfuse (`:3000`) **on the
  host** via `host.docker.internal` (wired to `host-gateway` in the compose
  file). The `localhost` URLs in `.env` only apply outside the container.
- If ufw runs with its default drop-incoming policy, it will block that
  bridge→host traffic, and the app will 500 on every chat (connection
  refused to the LLM) even though `/health` passes. Allow the Docker bridge
  subnet to reach those two ports:

  ```bash
  sudo ufw allow from 172.16.0.0/12 to any port 11434,3000
  ```

- Or skip the bridge entirely: set `network_mode: host` on the service (and
  drop the `ports:` mapping) — then `localhost` in `.env` works as-is and no
  firewall rule is involved.

## Chatting from the terminal (`tui.py`)

Once the API is up — the Docker container or the foreground command above —
in any terminal:

```bash
uv run python -m langgraph_ray_deploy.tui
```

It posts to `http://localhost:8000/invoke`, keeps a session ID for the whole
conversation, and prints the answer plus token usage. Type `exit` or `quit`
to leave.

`tui_stream.py` is the same chat over the streaming endpoint: the answer
prints token-by-token as it's generated instead of after the full response:

```bash
uv run python -m langgraph_ray_deploy.tui_stream
```

## Trying it

```bash
curl -s http://localhost:8000/health

curl -s http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
        "username": "mike",
        "message": "Hello Ray Serve!",
        "session_id": "optional-session-id"
      }'
```

Each invocation is traced in Langfuse under the session ID you provide (or a
fresh one is generated when omitted).

## Streaming mode (`/stream`)

`/invoke` waits for the complete answer before replying. `/stream` instead
returns the answer **as it is generated**, using
[Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
(SSE). Same request body as `/invoke`:

```bash
curl -N -s http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{
        "username": "mike",
        "message": "Hello Ray Serve!",
        "session_id": "optional-session-id"
      }'
```

(`-N` disables curl's output buffering so tokens appear as they arrive.
Without it, curl may hold output in chunks that have little to do with the
actual token cadence.)

### Event protocol

The response body is a stream of SSE events: an `event:` line naming the
event, a `data:` line with its JSON payload, and a blank line separating
events. A successful stream looks like:

```text
event: token
data: {"content": "Hello"}

event: token
data: {"content": "! Ray"}

event: token
data: {"content": " Serve here."}

event: done
data: {"session_id": "3f2c...9a1d"}
```

| Event | Meaning |
|-------|---------|
| `token` | A chunk of the model's answer. Concatenate `data.content` from all `token` events, in order, to reconstruct the full answer. |
| `done` | Final event of a successful stream. `data.session_id` is the ID this conversation was traced under; send it back as `session_id` on the next request to continue the conversation. |
| `error` | Something failed while running the graph (e.g. Ollama unreachable). `data.error` carries the message. This is the last event. |

Notes:

- The server already filters the graph output — only chunks produced by the
  `chatbot` node become `token` events, so clients don't need to skip anything.
- A malformed body (e.g. missing `username`) is rejected with a plain
  `422` JSON response before the stream starts.
- Tokens arrive in small bursts with gaps between them. That is local
  Ollama batching, not buffering in the app — the stream is flushed per event.
- `tui_stream.py` (above) is a ready-made consumer of this protocol.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langgraph_ray_deploy'`

You're running the system Python, not the project's venv. Because of the
`src/` layout, nothing on the system Python's path can ever see the package —
it only exists inside `.venv`. Fix is to run through uv (or activate the
venv):

```bash
uv run python -m langgraph_ray_deploy.main
# or
source .venv/bin/activate
python -m langgraph_ray_deploy.main
```

### Prompt not found in Langfuse

The service fetches a prompt named `langgraph-ray-serve-prompt` under the
label `production` (both in `src/langgraph_ray_deploy/graph.py`) on every
chat request. If it's missing, `/health` still passes but `/invoke` will 500.
Create it under Prompt Management in Langfuse with that exact name and label.
