# Nawa AI — Jordanian-Arabic Voice AI Backend

**نوا** is a real-time, Jordanian-Arabic (عامية أردنية) voice customer-service
backend that acts as a **Custom LLM for [Vapi.ai](https://vapi.ai)**. Vapi
handles telephony, speech-to-text, and text-to-speech; this service is the
brain: it receives the conversation, decides what to say, and streams the reply
back token-by-token.

The entire design is organized around one hard constraint:

> **Time-to-first-token (TTFT) must stay under 500ms. Everything streams.
> Nothing blocks the stream. No second LLM is ever called for guardrails.**

---

## How it works

Vapi calls this service exactly like the OpenAI `POST /chat/completions`
endpoint (it adds a `call` object). We reply with an OpenAI-format Server-Sent
Events stream. Per turn:

```
Vapi ──POST /chat/completions──▶  ┌─────────────────────────────────────────┐
                                  │ 1. Input guard   (regex, <1ms)           │
                                  │ 2. RAG           (embed+Qdrant, ~80ms)   │
                                  │ 3. Load history  (Redis GET, <10ms)      │
                                  │ 4. Stream reply, guarding each sentence  │
                                  │ 5. Persist turn  (after [DONE])          │
                                  └─────────────────────────────────────────┘
   ◀── data: {…delta…}\n\n  ×N  ──  data: [DONE]\n\n
```

### The guardrails (no LLM on the hot path)

- **Input guard** — `app/core/guards.py::scan_input`. Pure regex/blocklist,
  runs in well under a millisecond *before* the stream opens. Catches
  prompt-injection ("ignore your instructions" / "تجاهل تعليماتك"), unsafe
  requests, and off-topic abuse. When blocked, the caller streams a polite
  Jordanian redirect as a normal reply — the call never errors out.

- **Output guard** — `app/core/guards.py::check_output_sentence`, driven by
  `app/core/llm.py::stream_guarded`. Tokens are buffered only until an Arabic
  sentence boundary (`، . ؟ ! \n`), then a fast regex check runs (no fake
  guarantees, no leaked internal info, no invented numeric prices) and the
  sentence is flushed. **Latency cost = one sentence, not the whole reply.** On
  any blocked sentence, failed generation, or empty reply, a graceful spoken
  Arabic fallback is streamed followed by `[DONE]` — never dead air.

### RAG

`app/core/rag.py`. A cheap regex **intent gate** skips retrieval for
greetings/chitchat. Otherwise the query is embedded with
`text-embedding-3-small` (Redis-cached by normalized query), and the top-3
chunks are pulled from Qdrant with an optional metadata prefilter and score
threshold. Retrieved facts are injected into a **dedicated system block** that
instructs the model to answer *only* from them. RAG runs entirely *before* the
stream opens, budgeted at ~80ms.

### State

`app/core/state.py`. Conversation context is stored in Redis keyed by the Vapi
`call.id`. A single `GET` (<10ms) loads the last `HISTORY_TURNS` turns into the
prompt. The turn is persisted *after* `[DONE]`, so state I/O never touches TTFT.

### Prompt

`app/core/prompts.py`. A strict Jordanian-Arabic system prompt for the agent
**نوا**: Jordanian dialect only (MSA and other dialects banned), short spoken
sentences, answer only from retrieved context, offer a human transfer when
unknown (never invent prices/details), spoken numbers as words, no
markdown/symbols, and don't mention being an AI unless asked. `FEW_SHOT_EXAMPLES`
holds clearly-marked **placeholder** turns (`// TODO: replace with real call
excerpts`); the builder injects them if present and works fine if the list is
empty.

---

## Project structure

```
app/
  main.py            FastAPI app + lifespan warming LLM/Qdrant/Redis pools
  config.py          pydantic-settings; all env vars
  api/chat.py        POST /chat/completions (the Vapi Custom LLM contract)
  core/
    llm.py           one app-wide AsyncOpenAI client, streaming + output guard
    rag.py           intent gate, cached embed, top-3 retrieve, context build
    prompts.py       Jordanian-Arabic system prompt + FEW_SHOT_EXAMPLES
    guards.py        fast regex input filter + sentence-level output guard
    state.py         per-call state in Redis (Vapi call id), <10ms reads
    cache.py         Redis embedding cache (normalized-query keys)
  services/
    vector_db.py     async Qdrant client wrapper
  schemas/vapi.py    pydantic models for the request
  utils/
    logging.py       structured JSON logs tagged with call_id; TTFT per turn
    sse.py           OpenAI-format SSE chunk helpers (ensure_ascii=False)
scripts/ingest.py    embed + upsert business data into Qdrant
Dockerfile           slim python:3.12, uvicorn + uvloop + httptools, 1 worker
docker-compose.yml   app + qdrant + redis for local dev
requirements.txt
.env.example
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in the values. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — (required) | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model — swap without code changes |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_MAX_TOKENS` | `250` | Max tokens per streamed reply |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | RAG embedding model |
| `EMBEDDING_DIM` | `1536` | Embedding dimensionality |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `nawa_kb` | KB collection name |
| `RAG_TOP_K` | `3` | Chunks retrieved per query |
| `RAG_SCORE_THRESHOLD` | `0.30` | Min cosine score to keep a chunk |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis endpoint |
| `EMBEDDING_CACHE_TTL_S` | `604800` | Embedding cache TTL (7d) |
| `STATE_TTL_S` | `7200` | Per-call state TTL (2h) |
| `HISTORY_TURNS` | `6` | Recent turns loaded into the prompt |

---

## Run locally (docker-compose)

```bash
# 1. Configure
cp .env.example .env
#   edit .env and set OPENAI_API_KEY=...

# 2. Bring up app + Qdrant + Redis
docker compose up --build

# 3. Ingest the sample business knowledge base into Qdrant
docker compose exec app python -m scripts.ingest
#   (or ingest your own file: python -m scripts.ingest path/to/kb.json)

# 4. Health check
curl -s localhost:8000/health | jq
```

### Try a streamed turn

```bash
curl -N -X POST localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "gpt-4o-mini",
        "stream": true,
        "call": {"id": "test-call-1"},
        "messages": [
          {"role": "user", "content": "شو أوقات الدوام عندكم؟"}
        ]
      }'
```

You'll see `data: {…}` chunks in Jordanian Arabic ending with `data: [DONE]`.

### Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# start Qdrant + Redis however you like, then:
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --loop uvloop --http httptools --port 8000
```

---

## Wiring Vapi's Custom LLM

1. Deploy this service behind HTTPS (see Deployment) so it's reachable at, e.g.,
   `https://nawa.yourdomain.com`.
2. In the Vapi dashboard, on your Assistant's **Model** settings, choose
   **Custom LLM**.
3. Set the **URL** to your endpoint's base — Vapi appends `/chat/completions`:
   ```
   https://nawa.yourdomain.com
   ```
   (Point it at the base URL; the path served here is exactly `/chat/completions`.)
4. Set the model name to `gpt-4o-mini` (informational — the actual model is
   controlled by `LLM_MODEL` on the server).
5. Add any auth header Vapi supports if you front the service with a token — the
   endpoint itself is stateless and reads the `call` object from the body.
6. Save and place a test call. Watch the logs: each turn logs a `ttft_ms` line.

Vapi sends the full conversation plus the `call` object on every turn; we key
per-call state off `call.id`.

---

## Deployment (AWS ECS Fargate)

A live phone call is unforgiving: a cold start is **dead air**. Deploy for warm,
low-latency, always-on capacity.

- **ECS Fargate service**, `minimumHealthyPercent` tuned for zero-downtime
  rollouts. **Minimum 2 tasks** (never scale to zero) so there is always a warm
  task to answer a call.
- **Autoscaling** on CPU and/or request count (e.g. target-tracking at ~60% CPU),
  `min=2`, `max` sized to peak concurrent calls. **No scale-to-zero.**
- **Same region and VPC** as your Qdrant and Redis (ElastiCache) so RAG and state
  round-trips stay in single-digit milliseconds. Use private subnets + security
  groups; don't traverse the public internet for these hops.
- **Application Load Balancer** with **buffering disabled for SSE**: the target
  responds `text/event-stream` and sends `X-Accel-Buffering: no`. Increase the
  ALB **idle timeout** (e.g. 120s) so long turns aren't cut. (If you front with
  nginx/CloudFront instead, disable proxy buffering there too.)
- **Health checks**: point the target group at `GET /health`, which verifies
  Redis, Qdrant, and LLM reachability and returns `503` when any dependency is
  down, so unhealthy tasks are drained.
- **Warm pools**: the app lifespan issues throwaway LLM + embedding calls at
  startup, so a task is warm before it enters the load balancer rotation.
- **Single async worker per task** (uvloop + httptools). Scale out with more
  tasks, not more worker processes — this is an I/O-bound streaming workload.
- **Secrets** (`OPENAI_API_KEY`, `QDRANT_API_KEY`) via AWS Secrets Manager / SSM,
  injected as task-definition secrets — never baked into the image.

### Health endpoint

```
GET /health   →  200 {"status":"ok","redis":true,"qdrant":true,"llm":true}
              →  503 {"status":"degraded", ...}   if any dependency is down
```

---

## Design notes / guarantees

- **One OpenAI client** for the whole process — connection reuse is what keeps
  TTFT down; the client uses `max_retries=0` because a retry on the live path
  would blow the latency budget.
- **`ensure_ascii=False` everywhere** so Arabic travels as UTF-8, not `\uXXXX`.
- **Guards never call an LLM.** Every check is a precompiled regex.
- **Failures degrade gracefully**: cache/state/RAG errors are swallowed and the
  turn proceeds on the base prompt; generation failures stream the Arabic
  fallback then `[DONE]`.

## Offline local LLM smoke test

For a no-network/no-API-key Vapi contract test, run the backend with the built-in
OpenAI-compatible local stub. This mode streams Arabic chunks through the same
SSE endpoint and the same input/output guardrails, but disables RAG so it does
not need OpenAI embeddings or Qdrant.

```bash
LLM_PROVIDER=local_stub \
LLM_MODEL=local-restaurant-smoke \
RAG_ENABLED=false \
STRICT_DEPENDENCY_HEALTH=false \
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then verify streaming:

```bash
curl -N -X POST http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "local-restaurant-smoke",
        "stream": true,
        "call": {"id": "local-test-call"},
        "messages": [
          {"role": "user", "content": "مرحبا بدي احجز طاولة"}
        ]
      }'
```

Vapi runs in the cloud, so it cannot call `localhost` on your laptop. To test
this local backend from Vapi, expose it with an HTTPS tunnel and put the tunnel
base URL in the Vapi Custom LLM settings. The helper script below starts the
local stub backend and opens a temporary Cloudflare Tunnel:

```bash
./scripts/start_vapi_local.sh
```

When Cloudflare prints a URL like `https://example.trycloudflare.com`, put only
that base URL in Vapi. Vapi appends `/chat/completions` automatically:

```text
https://example.trycloudflare.com
```

Do **not** put `http://localhost:8000` in Vapi unless Vapi is running on the same
machine/network and can actually reach that host.
