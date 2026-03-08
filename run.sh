#!/usr/bin/env bash
# run.sh — Start Ali Real Estate Chatbot
# Usage:  chmod +x run.sh && ./run.sh
#         ./run.sh --stop   ← kill everything

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.ali_pids"
MODEL_NAME="ali-realestate"

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping Ali..."
  if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
      kill "$pid" 2>/dev/null && echo "  killed $pid" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  echo "Done."
  exit 0
fi

> "$PID_FILE"   # clear stale pids

# ── 1. Start Ollama ───────────────────────────────────────────────────────────
echo "[1/3] Starting Ollama..."
if ! curl -sf http://localhost:11434 &>/dev/null; then
  ollama serve > /tmp/ollama.log 2>&1 &
  echo $! >> "$PID_FILE"
  sleep 3
  echo "      Ollama started."
else
  echo "      Ollama already running."
fi

# ── 2. Create model ───────────────────────────────────────────────────────────
echo "[2/3] Creating model $MODEL_NAME..."
ollama create "$MODEL_NAME" -f "$SCRIPT_DIR/backend/Ollama/Modelfile"
echo "      Model ready."

# ── 3. Start FastAPI ──────────────────────────────────────────────────────────
echo "[3/3] Starting FastAPI..."
cd "$SCRIPT_DIR/backend/api"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload > /tmp/ali_api.log 2>&1 &
echo $! >> "$PID_FILE"

sleep 3
echo ""
echo "  Ali is running!"
echo "  Chat UI  →  http://localhost:8000"
echo "  API Docs →  http://localhost:8000/docs"
echo "  Stop     →  ./run.sh --stop"
