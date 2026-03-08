"""
conversation.py — Conversation Manager for Ali Real Estate Chatbot
Phase III: Conversation Manager and Prompt Orchestration
"""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "ali-realestate"

SESSION_TIMEOUT_SECONDS = 30 * 60   # 30 minutes
MAX_HISTORY_TURNS = 10              # sliding window (user+assistant pairs)
STREAM_TIMEOUT_SECONDS = 120

REAL_ESTATE_KEYWORDS = [
    "shop", "house", "villa", "apartment", "flat", "property", "properties",
    "marla", "kanal", "price", "cost", "buy", "purchase", "rent", "lease",
    "bedroom", "visit", "agent", "schedule", "booking", "crore", "lac",
    "location", "area", "size", "floor", "plot", "real estate", "realestate",
    "ali", "hello", "hi", "hey", "thanks", "thank", "okay", "ok", "yes",
    "no", "sure", "please", "help", "want", "need", "show", "tell", "info",
    "details", "more", "option", "available", "list",
]

OFF_TOPIC_POLICY_REMINDER = (
    "The user has asked something off-topic. "
    "Politely acknowledge that you can only assist with real estate inquiries, "
    "then redirect the conversation back to property selection."
)

STAGES = ["greeting", "category_selection", "subtype_selection", "closing"]

STAGE_HINTS = {
    "greeting": (
        "You have just greeted the customer. "
        "Your next step is to ask what type of property they are interested in "
        "(Shops, Houses/Villas, or Apartments)."
    ),
    "category_selection": (
        "The customer is choosing a property category. "
        "Present the available subtypes for their chosen category with prices."
    ),
    "subtype_selection": (
        "The customer has selected a subtype. "
        "Share its price and key features warmly, "
        "then ask if they'd like to schedule a visit or speak to an agent."
    ),
    "closing": (
        "The customer is wrapping up. "
        "Thank them warmly and offer any final assistance."
    ),
}

CORE_IDENTITY = (
    "You are Ali, a friendly and professional real estate assistant for a "
    "property agency based in Pakistan. "
    "You only discuss real estate — Shops, Houses/Villas, and Apartments. "
    "Never discuss anything unrelated to real estate."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Holds all state for a single user session."""
    session_id: str
    history: list[dict] = field(default_factory=list)
    stage: str = "greeting"
    last_active: float = field(default_factory=time.time)
    greeting_preserved: Optional[dict] = None  # first assistant message


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Manages per-session dialogue state, prompt orchestration,
    conversation policy enforcement, and streaming LLM calls via Ollama.
    """

    def __init__(
        self,
        max_history_turns: int = MAX_HISTORY_TURNS,
        session_timeout: int = SESSION_TIMEOUT_SECONDS,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self.max_history_turns = max_history_turns
        self.session_timeout = session_timeout

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new session and return its unique session_id."""
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(session_id=sid)
        return sid

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve an existing session by ID.
        Returns None if the session does not exist or has expired.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.last_active > self.session_timeout:
            self.delete_session(session_id)
            return None
        return session

    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID."""
        self._sessions.pop(session_id, None)

    def _touch(self, session: Session) -> None:
        """Update the last-active timestamp for a session."""
        session.last_active = time.time()

    def purge_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        Returns the number of sessions removed.
        """
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self.session_timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    # ------------------------------------------------------------------
    # Dialogue History Management
    # ------------------------------------------------------------------

    def _append_turn(self, session: Session, role: str, content: str) -> None:
        """
        Append a single turn (user or assistant) to the session history.
        If this is the first assistant message, cache it as the preserved greeting.
        """
        turn = {"role": role, "content": content}
        session.history.append(turn)

        # Preserve the very first assistant message (greeting anchor)
        if role == "assistant" and session.greeting_preserved is None:
            session.greeting_preserved = turn

    def _trimmed_history(self, session: Session) -> list[dict]:
        """
        Return a context-window-limited slice of the history.

        Strategy:
        - Keep the preserved greeting at index 0 (if present).
        - Keep the last `max_history_turns` messages from the rest.
        - Deduplicate in case the greeting is already inside the window.
        """
        history = session.history
        if not history:
            return []

        # Last N messages (sliding window)
        window = history[-self.max_history_turns:]

        # Re-insert the greeting at the front if it was pushed out
        if (
            session.greeting_preserved is not None
            and session.greeting_preserved not in window
        ):
            window = [session.greeting_preserved] + window

        return window

    # ------------------------------------------------------------------
    # Conversation Policy Enforcement
    # ------------------------------------------------------------------

    def _is_off_topic(self, message: str) -> bool:
        """
        Heuristic check: returns True if the message appears off-topic
        (i.e., contains no real-estate-related keywords).
        """
        lowered = message.lower()
        return not any(kw in lowered for kw in REAL_ESTATE_KEYWORDS)

    def _advance_stage(self, session: Session, user_message: str) -> None:
        """
        Move the conversation stage forward based on simple keyword signals
        in the latest user message.
        """
        msg = user_message.lower()
        current = session.stage

        if current == "greeting":
            # Any property-type mention moves us to category selection
            if any(kw in msg for kw in ["shop", "house", "villa", "apartment", "flat"]):
                session.stage = "category_selection"

        elif current == "category_selection":
            # Size/bedroom mentions move us to subtype selection
            if any(kw in msg for kw in ["marla", "kanal", "bedroom", "1", "2", "3",
                                         "5", "7", "8", "10"]):
                session.stage = "subtype_selection"

        elif current == "subtype_selection":
            # Scheduling or closing signals
            if any(kw in msg for kw in ["visit", "schedule", "agent", "call",
                                         "thank", "thanks", "bye", "goodbye"]):
                session.stage = "closing"

    # ------------------------------------------------------------------
    # Prompt Orchestration
    # ------------------------------------------------------------------

    def _build_system_prompt(self, session: Session, policy_note: str = "") -> str:
        """
        Dynamically construct the system prompt by combining:
        - Core identity/role
        - Current stage hint
        - Any active policy reminder
        """
        parts = [CORE_IDENTITY]
        parts.append(f"[Stage hint — {session.stage}]: {STAGE_HINTS[session.stage]}")
        if policy_note:
            parts.append(f"[Policy reminder]: {policy_note}")
        return "\n\n".join(parts)

    def build_prompt_messages(
        self, session: Session, user_message: str
    ) -> tuple[list[dict], str]:
        """
        Build the full message list to send to Ollama's /api/chat endpoint.

        Returns:
            (messages, effective_user_content) where messages is:
            [system_prompt_dict] + [trimmed_history] + [current_user_turn]
        """
        # Determine if policy injection is needed
        policy_note = ""
        effective_user_content = user_message
        if self._is_off_topic(user_message):
            policy_note = OFF_TOPIC_POLICY_REMINDER
            effective_user_content = (
                f"{user_message}\n\n[System note: This appears off-topic. "
                "Redirect politely to real estate.]"
            )

        system_prompt = self._build_system_prompt(session, policy_note)
        system_msg = {"role": "system", "content": system_prompt}

        history_slice = self._trimmed_history(session)
        current_turn = {"role": "user", "content": effective_user_content}

        messages = [system_msg] + history_slice + [current_turn]
        return messages, effective_user_content

    # ------------------------------------------------------------------
    # Ollama Integration
    # ------------------------------------------------------------------

    async def stream_response(
        self, session_id: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Main entry point for a single conversational turn.

        1. Validates / retrieves the session.
        2. Applies policy checks and stage advancement.
        3. Builds the structured prompt.
        4. Streams tokens from Ollama via NDJSON.
        5. Appends completed turns to history.

        Yields individual token strings as they arrive.
        Raises ValueError if the session is not found.
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not found or has expired.")

        self._touch(session)

        # Advance stage before building prompt
        self._advance_stage(session, user_message)

        messages, effective_content = self.build_prompt_messages(session, user_message)

        # Append the (possibly policy-wrapped) user turn to history
        self._append_turn(session, "user", effective_content)

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
            },
            "think":    False, 
        }

        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=STREAM_TIMEOUT_SECONDS) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                            yield token

                        # Ollama signals end-of-stream with {"done": true}
                        if chunk.get("done", False):
                            break

        except httpx.ConnectError:
            error_msg = "[Error: Could not connect to Ollama. Is it running?]"
            yield error_msg
            full_response = error_msg
        except httpx.TimeoutException:
            error_msg = "[Error: Ollama response timed out.]"
            yield error_msg
            full_response = error_msg
        except httpx.HTTPStatusError as exc:
            error_msg = f"[Error: Ollama returned HTTP {exc.response.status_code}]"
            yield error_msg
            full_response = error_msg

        # Persist the completed assistant turn
        if full_response:
            self._append_turn(session, "assistant", full_response)


# ---------------------------------------------------------------------------
# Multi-turn Dialogue Test
# ---------------------------------------------------------------------------

async def _run_test_dialogue() -> None:
    """
    Hardcoded 6-turn test dialogue to validate the ConversationManager.
    Run with:  python conversation.py
    """
    manager = ConversationManager()
    sid = manager.create_session()
    print(f"=== Test session: {sid} ===\n")

    turns = [
        "Hi",
        "I want to buy a house",
        "Show me 10 marla options",
        "What's the weather today?",   # off-topic
        "I want to schedule a visit",
        "Thank you",
    ]

    for i, user_msg in enumerate(turns, 1):
        print(f"[Turn {i}] User: {user_msg}")
        print(f"[Turn {i}] Ali:  ", end="", flush=True)

        session = manager.get_session(sid)
        stage_before = session.stage if session else "unknown"

        async for token in manager.stream_response(sid, user_msg):
            print(token, end="", flush=True)

        session = manager.get_session(sid)
        stage_after = session.stage if session else "unknown"
        print(f"\n          [stage: {stage_before} → {stage_after}]\n")


if __name__ == "__main__":
    asyncio.run(_run_test_dialogue())