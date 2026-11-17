# FAQ RAG Assistant

A retrieval-augmented generation (RAG) chat API that answers questions **only from documents you ingest**. When the answer isn't in the documents, it says so instead of guessing.

Distilled from a production RAG assistant I designed and built over several months; this is a generic, open version of its FAQ engine.

**Live demo:** open the chat button on [binilbaidya.com.np](https://www.binilbaidya.com.np) and ask about my work. It answers from [`data/faqs/about-binil.md`](data/faqs/about-binil.md).

**Stack:** FastAPI · LangChain · Qdrant · MongoDB (optional) · any OpenAI-compatible LLM (OpenAI, or free Google Gemini) · Docker

## How it works

```
 .md/.txt files ──► chunk by heading path (800 words, 100 overlap) ──► embed ──► Qdrant
                                                                                   │
 question ──► injection check ──► rate limit ──► AI switch ──► embed ──► hybrid search
                                                                           │ 72% vector + 28% keyword
                                                                           ▼
                              top 5 chunks ──► LLM (answer only from context) ──► SSE stream
                                                                           │
                                     signed-in users: both turns saved to MongoDB (titled, paged)
```

- **Heading-aware chunking.** Each Markdown section becomes a chunk that carries its full heading path (`## Orders > Returns`), so a subsection still names its topic. Long sections are split with overlap.
- **Hybrid scoring.** Cosine similarity is blended with a keyword score that ignores stopwords and plurals, so exact terms still rank well.
- **Grounded answers.** Retrieved text is treated as untrusted data. The model matches on meaning, gives partial answers honestly, and falls back to a fixed reply when the context is about something else.
- **Chat history (optional).** With `MONGO_URI` set, signed-in users (JWT) get conversations in MongoDB, auto-titled from the first message, with cursor-paged history. Anonymous visitors, and everyone when MongoDB isn't configured, chat without history.
- **AI on/off switch.** With MongoDB, admins can turn the assistant off at runtime (`PUT /v1/admin/ai`, cached for 15s). `AI_FORCE_DISABLED` works with or without MongoDB.
- **Guardrails.** Obvious prompt-injection phrases are rejected before reaching the LLM. A per-client rate limit plus a site-wide daily cap protect the LLM quota.

## Run locally

```bash
cp .env.example .env               # add OPENAI_API_KEY (or the Gemini settings) and a long JWT_SECRET_KEY
docker compose up -d               # API :8000, Qdrant :6333, MongoDB :27017
docker compose exec bot python -m scripts.ingest_faq_files data/faqs --recreate
curl -N -X POST localhost:8000/v1/chat/stream -H 'Content-Type: application/json' \
  -d '{"message": "What technologies does Binil work with?"}'
```

API docs: http://localhost:8000/docs (only when `ENVIRONMENT=development`).

For the admin endpoints, mint a token: `docker compose exec bot python -m scripts.create_admin_token`, then send `Authorization: Bearer <token>`.

### Use Gemini for free
Gemini has an OpenAI-compatible endpoint, so no code changes are needed:

```env
OPENAI_API_KEY=<key from aistudio.google.com>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_CHAT_MODEL=gemini-flash-lite-latest
OPENAI_EMBEDDING_MODEL=gemini-embedding-001
VECTOR_SIZE=3072
```

Changing the embedding model changes `VECTOR_SIZE`, so re-ingest with `--recreate`.

### Optional settings
`.env.example` lists only what you must set. These have working defaults:

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | Any other value hides `/docs`; use `production` when deployed |
| `APP_NAME` | `FAQ RAG Assistant` | Shown in `/health` and the docs |
| `LOG_LEVEL` | `INFO` | |
| `OPENAI_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint (see Gemini above) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `VECTOR_SIZE` | `1536` | Must match the embedding model |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-request limit for the LLM and embeddings (one retry) |
| `QDRANT_COLLECTION` | `faqs` | Use one per project |
| `STORE_ID` | `faq` | Knowledge-base namespace inside the collection |
| `ASSISTANT_PERSONA` | `polite` | `polite`, `fun` or `professional` (add more in `PERSONAS` in `app/agents/faq_agent.py`); answer and safety rules always apply |
| `NO_ANSWER_MESSAGE` | "I'm sorry, I don't have the answer…" | Exact reply when the FAQ doesn't cover a question |
| `MONGO_DB` | `faq_bot` | Database the app uses (always wins over a name in `MONGO_URI`); use one per project |
| `MONGO_CHAT_COLLECTION` / `MONGO_CONVERSATIONS_COLLECTION` | `chat_messages` / `chat_conversations` | |
| `AI_DEFAULT_ENABLED` | `true` | AI switch value before an admin sets it |
| `AI_FORCE_DISABLED` | `false` | Hard kill switch, overrides the admin setting |
| `CHAT_RATE_LIMIT` / `CHAT_RATE_WINDOW_SECONDS` | `20` / `60` | Questions per client per window |
| `CHAT_DAILY_LIMIT` | `500` | Questions per day across all visitors; `0` turns it off |
| `CORS_ALLOW_ALL` | `false` | Allow every origin |

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/chat` | optional JWT | `{"message", "conversation_id"?}` → answer + sources |
| `POST` | `/v1/chat/stream` | optional JWT | Same, as SSE: `token` events, then `done` with sources and title |
| `POST` | `/v1/chat/conversations` | JWT | Start (or reuse an empty) conversation |
| `GET` | `/v1/chat/conversations` | JWT | List titled conversations, newest first |
| `GET` | `/v1/chat/history?conversation_id=…&before=…` | JWT | Cursor-paged messages |
| `GET` / `PUT` | `/v1/admin/ai` | ADMIN | Read or set the AI on/off switch |
| `POST` | `/v1/admin/faqs` | ADMIN | Upload `.md`/`.txt` files (multipart `files`) or JSON documents |
| `GET` | `/v1/admin/faqs` | ADMIN | List ingested files and chunk counts |
| `DELETE` | `/v1/admin/faqs?document_id=…` | ADMIN | Remove one ingested file |
| `GET` | `/health`, `/v1/ai/status` | none | Health and AI switch status |

Chat endpoints are rate-limited. A signed-in chat turn must include `conversation_id`.

## Tests

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

The tests run offline: in-memory Qdrant, deterministic fake embeddings, a fake LLM and stub history.

## Deploy for free

| Part | Service |
|---|---|
| API | [Render](https://render.com) free web service from the `Dockerfile` (sleeps when idle; first request ~30s) |
| Vector DB | [Qdrant Cloud](https://cloud.qdrant.io) free cluster → `QDRANT_URL`, `QDRANT_API_KEY` |
| Chat history (optional) | [MongoDB Atlas](https://www.mongodb.com/atlas) free M0 cluster → `MONGO_URI`; skip it if visitors chat anonymously |
| LLM | Gemini free tier (above) |
| Website | Any static host (e.g. Netlify) calling the API |

1. Create the Qdrant cluster (and Atlas, if you want history) and a Render web service. Copy the variables from `.env.example` into Render, with `ENVIRONMENT=production` and `CORS_ALLOWED_ORIGINS=<your site URL>`.
2. Ingest once from your machine against the cloud Qdrant: `QDRANT_URL=… QDRANT_API_KEY=… python -m scripts.ingest_faq_files data/faqs --recreate`
3. Add `examples/ChatWidget.tsx` to your Next.js site and set `NEXT_PUBLIC_CHAT_API_URL` to the Render URL.

## Project layout

```
app/
  agents/faq_agent.py          retrieval + grounded answer chain
  agents/orchestrator.py       AI switch, history bookkeeping, streaming
  database/vector_store.py     Qdrant collection, upsert, hybrid search
  database/chat_history.py     conversations + messages in MongoDB
  database/ai_settings.py      runtime AI on/off switch
  ingestion/text_files.py      heading-path chunking
  api/routes/                  chat, admin (FAQs, AI switch), health
  auth.py, security.py, rate_limit.py
scripts/                       ingest a folder, mint an admin token
data/faqs/                     sample knowledge base
examples/ChatWidget.tsx        React widget for a website
```
