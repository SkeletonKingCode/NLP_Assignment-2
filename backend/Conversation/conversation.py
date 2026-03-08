"""
conversation.py
ConversationManager for Ali — Pakistani Real Estate Chatbot
Phase III: Conversation Manager and Prompt Orchestration
"""

import asyncio
import httpx
import uuid
import time
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "ali-realestate"

SESSION_TIMEOUT_SECONDS = 30 * 60   # 30 minutes
MAX_HISTORY_TURNS = 10              # sliding window (each turn = 1 message)

# Conversation stages in order
STAGES = ["greeting", "category_selection", "subtype_selection", "closing"]

# Keywords that indicate the user is ON-topic (real-estate related)
REAL_ESTATE_KEYWORDS = [
    "shop", "house", "villa", "apartment", "flat", "property", "marla",
    "kanal", "room", "bedroom", "price", "cost", "pkr", "crore", "lac",
    "buy", "purchase", "rent", "visit", "agent", "schedule", "real estate",
    "building", "plot", "commercial", "residential", "floor", "area",
    "location", "booking", "payment", "installment", "hi", "hello", "hey",
    "thank", "thanks", "bye", "goodbye", "yes", "no", "okay", "ok",
    "sure", "please", "want", "need", "show", "tell", "what", "how",
    "which", "available", "option", "choose", "select", "interest",
]

# Stage-transition trigger keywords
CATEGORY_KEYWORDS  = ["shop", "house", "villa", "apartment", "flat"]
SUBTYPE_KEYWORDS   = ["marla", "kanal", "bedroom", "1 bed", "2 bed", "3 bed"]
CLOSING_KEYWORDS   = ["visit", "schedule", "agent", "book", "thank", "bye",
                      "goodbye", "done", "okay", "ok", "sure"]

# System prompt template — stage hint and policy reminders are injected at runtime
SYSTEM_PROMPT_TEMPLATE = """\
You are Ali, a friendly and professional real estate assistant for a property \
agency based in Pakistan.

Your job is to help customers explore available properties across three categories:
- 🏪 Shops  (5 Marla – PKR 1.2 Crore | 8 Marla – PKR 2.1 Crore | 1 Kanal – PKR 3.8 Crore)
- 🏠 Houses / Villas  (5 Marla – PKR 1.8 Crore | 7 Marla – PKR 2.6 Crore | 10 Marla – PKR 4.2 Crore | 1 Kanal Villa – PKR 8.5 Crore)
- 🏢 Apartments  (1 Bed – PKR 55 Lac | 2 Bed – PKR 95 Lac | 3 Bed – PKR 1.5 Crore)

Conversation flow: greet → ask category → present subtypes → share price & features → offer visit/agent.
Never skip a stage. Never discuss anything outside real estate.
Use simple, warm English. Never invent properties or prices beyond the list above.

--- CURRENT STAGE HINT ---
{stage_hint}
{policy_reminder}\
"""

STAGE_HINTS = {
    "greeting":           "You are at the GREETING stage. Welcome the customer warmly and ask which property category interests them (Shops, Houses/Villas, or Apartments).",
    "category_selection": "You are at the CATEGORY SELECTION stage. The customer has shown interest. Present the available subcategories for their chosen category with prices.",
    "subtype_selection":  "You are at the SUBTYPE SELECTION stage. The customer has picked a subtype. Share its price and key features warmly, then ask if they'd like to schedule a visit or speak to an agent.",
    "closing":            "You are at the CLOSING stage. Help the customer schedule a visit or connect with an agent. Wrap up the conversation politely.",
}

OFF_TOPIC_REMINDER = """\

--- POLICY REMINDER ---
The customer's last message was off-topic. Politely acknowledge that you can \
only help with real estate matters, then steer the conversation back to where \
it left off.\
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Holds all state for a single user session."""
    session_id: str
    history: list[dict]          = field(default_factory=list)
    stage: str                   = "greeting"
    last_active: float           = field(default_factory=time.time)
    greeting_message: Optional[dict] = None   # preserved across window trims


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Manages multi-turn conversations with Ali, the real estate chatbot.

    Responsibilities
    ----------------
    - Session lifecycle (create / retrieve / expire / delete)
    - Dialogue history with sliding-window trimming
    - Conversation-stage tracking and enforcement
    - Off-topic detection and policy injection
    - Dynamic system-prompt construction
    - Async streaming calls to the local Ollama endpoint
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model_name: str = MODEL_NAME,
        max_history_turns: int = MAX_HISTORY_TURNS,
        session_timeout: int = SESSION_TIMEOUT_SECONDS,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.max_history_turns = max_history_turns
        self.session_timeout = session_timeout

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new session and return its unique session_id."""
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(session_id=sid)
        return sid

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.

        Returns None if the session does not exist or has expired.
        Expired sessions are removed from memory automatically.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.last_active > self.session_timeout:
            self.delete_session(session_id)
            return None
        return session

    def delete_session(self, session_id: str) -> None:
        """Remove a session from memory."""
        self._sessions.pop(session_id, None)

    def _touch(self, session: Session) -> None:
        """Update the last-active timestamp to reset the inactivity timer."""
        session.last_active = time.time()

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _append_message(self, session: Session, role: str, content: str) -> None:
        """
        Append a message to the session history.

        If this is the first assistant message, it is also stored separately
        as the 'greeting_message' so it can be preserved after window trimming.
        """
        msg = {"role": role, "content": content}
        session.history.append(msg)

        # Capture the greeting (first assistant turn) for permanent preservation
        if role == "assistant" and session.greeting_message is None:
            session.greeting_message = msg

    def _trim_history(self, session: Session) -> list[dict]:
        """
        Return a trimmed view of the history that fits within the context window.

        Rules
        -----
        - Keep at most `max_history_turns` messages from the *end* of history.
        - Always prepend the greeting message if it was trimmed out, ensuring
          the model never loses its opening context.
        """
        history = session.history
        if len(history) <= self.max_history_turns:
            return list(history)

        trimmed = history[-self.max_history_turns:]

        # Re-insert greeting if it was cut off
        if (
            session.greeting_message is not None
            and session.greeting_message not in trimmed
        ):
            trimmed = [session.greeting_message] + trimmed

        return trimmed

    # ------------------------------------------------------------------
    # Conversation policy
    # ------------------------------------------------------------------

    @staticmethod
    def _is_off_topic(user_message: str) -> bool:
        """
        Heuristically decide whether a user message is off-topic.

        Strategy: lowercase the message and check whether *any* real-estate
        keyword appears in it.  Short affirmative/social messages ("yes",
        "hi", "thanks") are also whitelisted via the keyword list.
        """
        lowered = user_message.lower()
        return not any(kw in lowered for kw in REAL_ESTATE_KEYWORDS)

    @staticmethod
    def _advance_stage(session: Session, user_message: str) -> None:
        """
        Attempt to advance the conversation stage based on user input.

        The stage machine only moves *forward*, never backward.
        """
        lowered = user_message.lower()
        current_idx = STAGES.index(session.stage)

        if current_idx == 0:  # greeting → category_selection
            if any(kw in lowered for kw in CATEGORY_KEYWORDS):
                session.stage = STAGES[1]
        elif current_idx == 1:  # category_selection → subtype_selection
            if any(kw in lowered for kw in SUBTYPE_KEYWORDS):
                session.stage = STAGES[2]
        elif current_idx == 2:  # subtype_selection → closing
            if any(kw in lowered for kw in CLOSING_KEYWORDS):
                session.stage = STAGES[3]
        # closing is terminal; no further advancement

    # ------------------------------------------------------------------
    # Prompt orchestration
    # ------------------------------------------------------------------

    def _build_system_prompt(self, session: Session, off_topic: bool) -> str:
        """
        Dynamically construct the system prompt.

        Incorporates:
        - Core identity and inventory
        - Current stage hint
        - Policy reminder (injected only when the last message was off-topic)
        """
        stage_hint      = STAGE_HINTS.get(session.stage, STAGE_HINTS["greeting"])
        policy_reminder = OFF_TOPIC_REMINDER if off_topic else ""
        return SYSTEM_PROMPT_TEMPLATE.format(
            stage_hint=stage_hint,
            policy_reminder=policy_reminder,
        )

    def _build_messages(
        self, session: Session, user_message: str, off_topic: bool
    ) -> list[dict]:
        """
        Assemble the final message list to send to Ollama.

        Structure
        ---------
        [system_prompt] + [trimmed_history] + [current_user_turn]

        When a message is off-topic the raw user text is still forwarded
        (the system prompt already instructs the model how to handle it).
        """
        system_prompt = self._build_system_prompt(session, off_topic)
        trimmed       = self._trim_history(session)

        messages = [{"role": "system", "content": system_prompt}]
        messages += trimmed
        messages += [{"role": "user", "content": user_message}]
        return messages

    # ------------------------------------------------------------------
    # Ollama integration
    # ------------------------------------------------------------------

    async def _stream_ollama(
        self, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        """
        Call the Ollama /api/chat endpoint with streaming enabled.

        Parses the NDJSON response line-by-line and yields individual
        content tokens as they arrive.

        Raises
        ------
        ConnectionError  — if Ollama is unreachable.
        TimeoutError     — if the request exceeds 60 seconds without a response.
        RuntimeError     — for unexpected HTTP errors.
        """
        payload = {
            "model":    self.model_name,
            "messages": messages,
            "stream":   True,
            "options":  {"temperature": 0.7},
            "think":    False, 
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", self.ollama_url, json=payload) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"Ollama returned HTTP {resp.status_code}"
                        )
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
                            yield token

                        if chunk.get("done", False):
                            break

        except httpx.ConnectError as exc:
            raise ConnectionError(
                "Could not connect to Ollama. Is it running on localhost:11434?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError("Ollama request timed out after 60 seconds.") from exc

    # ------------------------------------------------------------------
    # Public chat interface
    # ------------------------------------------------------------------

    async def chat(
        self, session_id: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Process one user turn and stream the assistant's reply.

        Steps
        -----
        1. Retrieve (or recreate) the session.
        2. Detect off-topic input.
        3. Advance the conversation stage.
        4. Build the orchestrated prompt.
        5. Stream tokens from Ollama; accumulate full response.
        6. Persist user + assistant turns to history.
        7. Yield tokens to the caller.

        Parameters
        ----------
        session_id   : ID returned by create_session().
        user_message : Raw text from the user.

        Yields
        ------
        str tokens as they stream from the model.
        """
        session = self.get_session(session_id)
        if session is None:
            # Silently recreate so callers don't crash on expired sessions
            self._sessions[session_id] = Session(session_id=session_id)
            session = self._sessions[session_id]

        self._touch(session)

        off_topic = self._is_off_topic(user_message)

        # Only advance stage for on-topic messages
        if not off_topic:
            _advance_stage = self._advance_stage   # local alias for clarity
            _advance_stage(session, user_message)

        messages = self._build_messages(session, user_message, off_topic)

        # Collect the full assistant response while streaming tokens out
        full_response_parts: list[str] = []

        async for token in self._stream_ollama(messages):
            full_response_parts.append(token)
            yield token

        full_response = "".join(full_response_parts)

        # Persist both turns to history *after* streaming completes
        self._append_message(session, "user",      user_message)
        self._append_message(session, "assistant", full_response)


# ---------------------------------------------------------------------------
# Multi-turn dialogue test
# ---------------------------------------------------------------------------

async def _run_test_dialogue() -> None:
    """
    Hardcoded 6-turn test to exercise the ConversationManager end-to-end.

    Covers:
    - Normal flow (greeting → category → subtype → closing)
    - Off-topic detection and redirection (Turn 4)
    - Streaming token output
    """
    manager    = ConversationManager()
    session_id = manager.create_session()

    test_turns = [
        "Hi",
        "I want to buy a house",
        "Show me 10 marla options",
        "What's the weather today?",   # off-topic
        "I want to schedule a visit",
        "Thank you",
    ]

    for turn_num, user_input in enumerate(test_turns, start=1):
        print(f"\n{'='*60}")
        print(f"Turn {turn_num} | Stage: {manager._sessions[session_id].stage}")
        print(f"User : {user_input}")
        print(f"Ali  : ", end="", flush=True)

        async for token in manager.chat(session_id, user_input):
            print(token, end="", flush=True)

        print()   # newline after streamed response

    print(f"\n{'='*60}")
    print("Test dialogue complete.")


if __name__ == "__main__":
    asyncio.run(_run_test_dialogue())