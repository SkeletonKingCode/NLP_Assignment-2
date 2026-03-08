"""
backend/api/main.py

FastAPI server for the Ali Real Estate chatbot.

Endpoints
---------
POST   /session              — create a new session
GET    /session/{id}         — get session state summary
DELETE /session/{id}         — delete a session
WS     /ws/chat              — streaming chat over WebSocket

WebSocket protocol
------------------
Client  → server:  {"session_id": "<uuid>", "message": "<text>"}
Server  → client:  {"type": "token",  "data": "<token>"}   (streamed)
                   {"type": "done",   "data": ""}           (end of turn)
                   {"type": "error",  "data": "<message>"}  (on failure)
                   {"type": "state",  "data": {session info}} (after done)
"""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path resolution — works regardless of where main.py is placed in the tree
#
#   python3 backend/main.py          (main.py sits beside Conversation/)
#   python3 backend/api/main.py      (main.py one level below Conversation/)
#   python3 -m backend.api.main      (package-style invocation)
#   uvicorn backend.api.main:app     (uvicorn from project root)
# ---------------------------------------------------------------------------

def _find_backend_root() -> Path:
    """Walk up from this file until we find a directory that contains
    a 'Conversation' sub-package, then return that directory."""
    candidate = Path(__file__).resolve().parent
    for _ in range(4):                          # look at most 4 levels up
        if (candidate / "Conversation" / "conversation.py").exists():
            return candidate
        candidate = candidate.parent
    # Fallback: return the directory of this file itself
    return Path(__file__).resolve().parent


_BACKEND_ROOT = _find_backend_root()
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# _PROJECT_ROOT is one level above _BACKEND_ROOT (contains frontend/)
_PROJECT_ROOT = _BACKEND_ROOT.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

from Conversation.conversation import (   # noqa: E402  (import after sys.path fix)
    create_session,
    get_session,
    delete_session,
    get_session_info,
    stream_response,
)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    print("Ali Real Estate API starting up...")
    yield
    print("Ali Real Estate API shutting down.")


app = FastAPI(
    title="Ali Real Estate Chatbot API",
    description="Local conversational AI for Pakistani property sales.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static files if the directory exists
if _FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")

@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI at the root URL."""
    index = _FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return {"message": "Ali Real Estate API", "docs": "/docs", "health": "/health"}

# ---------------------------------------------------------------------------
# Active WebSocket connection tracker (for concurrency metrics)
# ---------------------------------------------------------------------------

_active_connections: set[WebSocket] = set()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class SessionResponse(BaseModel):
    session_id: str
    message: str


@app.post("/session", response_model=SessionResponse, status_code=201)
async def create_new_session():
    """Create a new conversation session.

    Returns the session_id the client must include in every WebSocket message.
    """
    sid = create_session()
    return SessionResponse(session_id=sid, message="Session created successfully.")


@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Return the current state summary for a session (no full history)."""
    info = get_session_info(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return info


@app.delete("/session/{session_id}", status_code=200)
async def end_session(session_id: str):
    """Delete a session immediately."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    delete_session(session_id)
    return {"message": "Session deleted.", "session_id": session_id}


@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {
        "status": "ok",
        "active_connections": len(_active_connections),
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time streaming chat endpoint.

    Protocol
    --------
    1. Client connects.
    2. Client sends JSON: {"session_id": "...", "message": "..."}
    3. Server streams token frames: {"type": "token", "data": "<tok>"}
    4. Server sends end frame:      {"type": "done",  "data": ""}
    5. Server sends state frame:    {"type": "state", "data": {...}}
    6. Repeat from step 2 for next turn.

    The client should create a session via POST /session before connecting.
    If no session_id is supplied, a new session is created automatically.
    """
    await websocket.accept()
    _active_connections.add(websocket)

    try:
        while True:
            # ── Receive client message ────────────────────────────────────────
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, "error", "Invalid JSON payload.")
                continue

            session_id: str = payload.get("session_id", "")
            user_message: str = payload.get("message", "").strip()

            # Auto-create a session if the client forgot to
            if not session_id or get_session(session_id) is None:
                session_id = create_session()
                await _send(websocket, "session_created", session_id)

            if not user_message:
                await _send(websocket, "error", "Empty message received.")
                continue

            # ── Stream response tokens ────────────────────────────────────────
            try:
                async for token in stream_response(session_id, user_message):
                    if token.startswith("[ERROR]"):
                        await _send(websocket, "error", token)
                        break
                    await _send(websocket, "token", token)

            except Exception as exc:  # noqa: BLE001
                await _send(websocket, "error", f"Streaming error: {exc}")
                continue

            # ── Signal end of turn + send updated state ───────────────────────
            await _send(websocket, "done", "")
            state = get_session_info(session_id)
            if state:
                await websocket.send_text(
                    json.dumps({"type": "state", "data": state})
                )

    finally:
        _active_connections.discard(websocket)


async def _send(ws: WebSocket, msg_type: str, data: str) -> None:
    """Helper: send a typed JSON frame to the client."""
    try:
        await ws.send_text(json.dumps({"type": msg_type, "data": data}))
    except Exception:  # noqa: BLE001
        pass  # client already disconnected


# ---------------------------------------------------------------------------
# Entry point (dev only — use uvicorn in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Use import string relative to _BACKEND_ROOT so uvicorn resolves it correctly
    # whether the file is at backend/main.py or backend/api/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
