# Ali — Pakistani Real Estate Conversational AI

A fully local, CPU-optimised conversational AI system built for a Pakistani property agency.
No cloud APIs. No RAG. No tools. All intelligence from prompt design and context management.

---

## Architecture

```
Browser (index.html)
       │  WebSocket /ws/chat  &  REST /session
       ▼
FastAPI (backend/api/main.py)
       │  stream_response() async generator
       ▼
Conversation Manager (backend/Conversation/conversation.py)
  ├── Session store (in-memory dict, UUID keyed, 30-min TTL)
  ├── Stage machine  greeting → category_selection → subtype_selection → closing
  ├── State extraction  (selected_category, selected_subtype, selected_price)
  ├── Dynamic system prompt  CORE_IDENTITY + CONVERSATION STATE + stage hint
  └── Context window  sliding last-10-turns window
       │  ollama.AsyncClient.chat(..., stream=True)
       ▼
Ollama (local daemon, port 11434)
       │
       ▼
ali-realestate  (qwen3.5:2b, GGUF quantized, CPU inference)
```

---

## Project Structure

```
ali-realestate/
├── backend/
│   ├── api/
│   │   └── main.py                  # FastAPI + WebSocket server
│   ├── Conversation/
│   │   └── conversation.py          # Session mgmt, prompt orchestration, Ollama streaming
│   └── Ollama/
│       ├── Modelfile                # Custom ali-realestate model definition
│       └── ModelCreation.sh         # ollama create + run commands
├── frontend/
│   └── index.html                   # ChatGPT-style web UI (single file, no build step)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Ali_Chatbot.postman_collection.json
└── README.md
```

---

## Setup & Run

### Prerequisites
- Docker + Docker Compose  **or**  Python 3.10+ and Ollama installed locally
- Minimum 4 GB RAM (model is ~1.5 GB quantized)

### Option A — Docker Compose (recommended)

```bash
git clone <repo-url> && cd ali-realestate
docker compose up --build
```

On first start, Ollama will download `qwen3.5:2b` (~1.5 GB) and build the custom model.
Open **http://localhost:8000** — wait for the API health check to return `"status": "ok"`.

### Option B — Local (no Docker)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Create the custom model
ollama create ali-realestate -f backend/Ollama/Modelfile

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the API
python3 backend/api/main.py

# 5. Open the frontend
open http://0.0.0.0:8000   # or serve via any static file server
```

---

## API Reference

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness probe, returns active WS connection count |
| `POST` | `/session` | Create a session → `{"session_id": "<uuid>"}` |
| `GET`  | `/session/{id}` | Get stage + selection state for a session |
| `DELETE` | `/session/{id}` | Delete a session immediately |

### WebSocket — `ws://host/ws/chat`

**Client → Server** (JSON)
```json
{ "session_id": "<uuid>", "message": "I want to buy a house" }
```

**Server → Client** (JSON frames)
```json
{ "type": "token",           "data": "Sure! Here" }          // streamed tokens
{ "type": "done",            "data": "" }                    // end of turn
{ "type": "state",           "data": { ...session_info } }  // updated session state
{ "type": "session_created", "data": "<new-uuid>" }         // auto-created session
{ "type": "error",           "data": "..." }                 // error message
```

---

## Conversation Flow

```
Greeting
  ↓  user mentions "house" / "shop" / "apartment"
Category Selection
  ↓  user mentions size ("10 marla", "1 bedroom", etc.)
Subtype Selection
  ↓  user says "schedule" / "visit" / "book"
Closing
```

### Authorised Inventory

| Category | Subtype | Price |
|----------|---------|-------|
| Shops | 5 Marla | PKR 1.2 Crore |
| Shops | 8 Marla | PKR 2.1 Crore |
| Shops | 1 Kanal | PKR 3.8 Crore |
| Houses/Villas | 5 Marla | PKR 1.8 Crore |
| Houses/Villas | 7 Marla | PKR 2.6 Crore |
| Houses/Villas | 10 Marla | PKR 4.2 Crore |
| Houses/Villas | 1 Kanal Villa | PKR 8.5 Crore |
| Apartments | 1 Bedroom | PKR 55 Lac |
| Apartments | 2 Bedroom | PKR 95 Lac |
| Apartments | 3 Bedroom | PKR 1.5 Crore |

---

## Context & Memory Design

### Problem
Small 2B models cannot reliably re-infer user choices from raw history alone —
especially after an off-topic detour or after the context window trims old turns.

### Solution: Explicit State Injection

Every turn, the system prompt includes a `CONVERSATION STATE` block:

```
CONVERSATION STATE  (tracked by the system — treat as ground truth)
--------------------------------------------------------------------
Stage             : subtype_selection
Category chosen   : Houses/Villas
Subtype chosen    : 10 Marla House
Price confirmed   : PKR 4.2 Crore
--------------------------------------------------------------------
IMPORTANT: Do NOT ask the customer again about choices already made above.
```

This is computed deterministically in Python from keyword matching — the model
never has to infer it. The context window slides over the last 10 turn-pairs;
no greeting-pinning is used (it caused the model to re-ask already-answered questions).

---

## Performance Benchmarks

> Measured on: Intel Core i7-12th Gen, 16 GB RAM, no GPU.

| Metric | Value |
|--------|-------|
| Model | qwen3.5:2b (Q4_K_M GGUF) |
| Time to first token (TTFT) | ~1.8 s |
| Token throughput | ~12 tok/s |
| Peak RAM usage | ~2.1 GB |
| Concurrent sessions tested | 5 (sequential WS connections) |
| Session TTL | 30 minutes |
| Context window | Last 10 turn-pairs |

> Note: Benchmarks are approximate. Results vary by hardware and model quantization level.

---

## Known Limitations

- **Single process, in-memory sessions** — sessions are lost on restart. For production, replace `_sessions` dict with Redis.
- **Keyword-based stage machine** — complex phrasings ("I'd fancy something about 1000 sq ft") may not trigger transitions correctly. A small intent classifier would improve robustness.
- **English only** — the Modelfile and stage logic assume English input. Urdu/Roman Urdu support requires prompt additions.
- **CPU latency** — first token takes ~2 s on a laptop CPU. A GPU or Apple Silicon chip reduces this to <0.5 s.
- **Single worker** — `--workers 1` in uvicorn ensures the in-memory session store is consistent. Scaling to multiple workers requires an external session store.

---

## Honor Policy

All code is original work by the group. Generative tools were used to accelerate implementation; all generated code was reviewed, understood, and modified by group members.
