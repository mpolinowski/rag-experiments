from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ray import serve

from .service import execute_chat, stream_chat


app = FastAPI(title="LangGraph Hello World API")


class ChatRequest(BaseModel):
    username: str
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str | list
    session_id: str
    model: str
    usage: dict


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=10,
)
@serve.ingress(app)
class LangGraphService:

    @app.get("/health")
    async def health(self):
        return {"status": "ok"}

    @app.post("/invoke", response_model=ChatResponse)
    async def invoke(self, request: ChatRequest):

        session_id = request.session_id or str(uuid4())

        try:
            result = await execute_chat(
                question=request.message,
                username=request.username,
                session_id=session_id,
            )

            return ChatResponse(**result)

        except Exception as exc:
            # Log the full exception in production.
            raise HTTPException(
                status_code=500,
                detail="Graph execution failed",
            ) from exc

    @app.post("/stream")
    async def stream(self, request: ChatRequest):

        return StreamingResponse(
            stream_chat(
                question=request.message,
                username=request.username,
                session_id=request.session_id or str(uuid4()),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


deployment = LangGraphService.bind()


def run():
    # Serve 2.x defaults the proxy to "localhost" only, which works on a
    # plain server (curl 127.0.0.1) but hides port 8000 from Docker's
    # published port (connection refused on the container's bridge IP).
    # Cluster-scoped options are only applied when the controller is first
    # created, so this must run before serve.run().
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

    # Ray >= 2.4x serve.run() is non-blocking by default and returns once
    # the app is deployed -- passing blocking=True keeps the local cluster
    # alive for the life of this process.
    serve.run(deployment, blocking=True)


if __name__ == "__main__":
    run()
