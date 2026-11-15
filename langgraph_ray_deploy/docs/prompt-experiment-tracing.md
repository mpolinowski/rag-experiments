# Prompt config → trace metadata → Langfuse filter

How this service tags Langfuse traces with the prompt experiment that produced
them, so prompt A/B runs are filterable and comparable in the Langfuse UI.

## The pattern

1. The Langfuse prompt `langgraph-ray-serve-prompt` (label `production`) carries
   a `config` dict. By convention the experiment tag lives under the key
   `prompt_experiment`, e.g. `{"prompt_experiment": "v2-strict-tone"}`.
2. The `chatbot` node in `graph.py` reads that config and passes it into
   `propagate_attributes` as trace metadata, alongside the fixed metadata from
   `service.py` (`model`, `environment`):

   ```python
   prompt_config = langfuse_prompt.config if isinstance(langfuse_prompt.config, dict) else {}

   with propagate_attributes(
       prompt=langfuse_prompt,
       metadata={"prompt_experiment": prompt_config.get("prompt_experiment", "default")},
   ):
       response = await model.ainvoke(messages)
   ```

3. Metadata merges per-key across `propagate_attributes` calls (inner context
   wins), so `prompt_experiment` is **added** to the trace without clobbering
   `model` / `environment` set in the outer context in `service.py`.
4. In the Langfuse UI, filter traces by `metadata.prompt_experiment` to compare
   prompt runs.

## Why it's done inside the graph node

The tag must come from the prompt itself — the single source of truth. Editing
the prompt's `config` in Langfuse (bumping a value, e.g. `default` → `v2-tuned`)
retargets all future traces with **no code change and no deploy**. Missing or
non-dict `config` degrades to `"default"`, so traces are always filterable.

## Gotchas

- **Prompt fetching is cached** (`cache_ttl_seconds=300` in `graph.py`). A tag
  edited in the UI can take up to 5 minutes to appear on new traces per
  worker process.
- **Ray Serve does not hot-reload.** After changing Python code, the cluster
  keeps running the previously-deployed module until you redeploy/restart.
  (Prompt *config* changes are live; *code* changes are not.)
- **Read API in Langfuse v4 events_only mode:** the legacy
  `GET /api/public/traces` list endpoint is 404; read traces/observations via
  `GET /api/public/v2/observations` instead.

## Verifying

Run:

```bash
set -a && . ./.env && set +a
.venv/bin/python -c "
import asyncio
from langgraph_ray_deploy.service import execute_chat
res = asyncio.run(execute_chat(question='reply with the single word pong',
                               username='vi-metadata-check',
                               session_id='metadata-verify'))
print(res['answer'])
"
```

then check the trace for session `metadata-verify` in the Langfuse UI:
`metadata` should contain `prompt_experiment` (value from the prompt config,
e.g. `"default"`) alongside `model: local-ollama` and `environment: production`.
