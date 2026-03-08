"""
backend/Conversation/conversation.py

Conversation manager for Ali — a Pakistani real estate assistant chatbot.
Handles session management, context window trimming, stage tracking,
off-topic policy enforcement, and streaming Ollama integration.

Context strategy
----------------
Small models (2B) cannot reliably re-infer what a user chose several turns
ago from raw chat history alone.  Instead, every turn we inject an explicit
CONVERSATION STATE block into the system prompt that names the stage, the
chosen category, and the chosen subtype as ground truth.  The model never
has to guess — we tell it exactly what has already been decided.

The old "pin the first greeting turn" approach is intentionally removed.
Re-inserting Ali's opening "What category would you like?" into a late-turn
context caused the model to think it still needed to ask that question.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator

import ollama

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "ali-realestate"
SESSION_TTL_SECONDS = 30 * 60   # 30-minute inactivity timeout
MAX_HISTORY_TURNS = 10           # sliding window: last N user+assistant pairs

# ── Inventory ───────────────────────────────────────────────────────────────
# Single source of truth.  Used in prompts AND in subtype extraction logic.

INVENTORY: dict[str, list[tuple[str, str]]] = {
    "Shops": [
        ("5 Marla Shop",   "PKR 1.2 Crore"),
        ("8 Marla Shop",   "PKR 2.1 Crore"),
        ("1 Kanal Shop",   "PKR 3.8 Crore"),
    ],
    "Houses/Villas": [
        ("5 Marla House",  "PKR 1.8 Crore"),
        ("7 Marla House",  "PKR 2.6 Crore"),
        ("10 Marla House", "PKR 4.2 Crore"),
        ("1 Kanal Villa",  "PKR 8.5 Crore"),
    ],
    "Apartments": [
        ("1 Bedroom Apt",  "PKR 55 Lac"),
        ("2 Bedroom Apt",  "PKR 95 Lac"),
        ("3 Bedroom Apt",  "PKR 1.5 Crore"),
    ],
}

def _inventory_block() -> str:
    """Render the full inventory as a formatted string for system prompts."""
    lines: list[str] = ["AUTHORISED INVENTORY — THE ONLY PROPERTIES THAT EXIST"]
    lines.append("=" * 55)
    for category, items in INVENTORY.items():
        lines.append(category.upper())
        for name, price in items:
            lines.append(f"  - {name:<20}: {price}")
        lines.append("")
    lines.append(
        "DO NOT invent locations, addresses, square footage, or prices.\n"
        "DO NOT modify or estimate any listed price.\n"
        "If asked for a size not listed, say it is unavailable and show what IS listed."
    )
    return "\n".join(lines)

CORE_IDENTITY = (
    "You are Ali, a friendly and professional real estate assistant for a "
    "property agency based in Pakistan.\n"
    "You ONLY discuss real estate: properties, prices, visits, and agent bookings.\n\n"
    + _inventory_block()
)

# ── Stage goal hints ─────────────────────────────────────────────────────────

STAGE_HINTS: dict[str, str] = {
    "greeting": (
        "CURRENT GOAL: Greet the customer warmly. "
        "Ask which category they want — Shops, Houses/Villas, or Apartments. "
        "Do NOT list prices yet."
    ),
    "category_selection": (
        "CURRENT GOAL: The customer has chosen a category (see CONVERSATION STATE). "
        "List ONLY the subtypes and exact PKR prices for that category from the "
        "AUTHORISED INVENTORY. Do NOT show subtypes from other categories."
    ),
    "subtype_selection": (
        "CURRENT GOAL: The customer has selected a specific subtype (see CONVERSATION STATE). "
        "State its exact price. Briefly describe it (great for families / good investment). "
        "Then ask: would they like to schedule a visit or speak to an agent? "
        "Do NOT offer other subtypes or re-list the category menu."
    ),
    "closing": (
        "CURRENT GOAL: Arrange a property visit or agent call for the chosen property "
        "(see CONVERSATION STATE). Be warm, confirm which property they selected, "
        "and offer clear next steps."
    ),
}

OFF_TOPIC_REMINDER = (
    "[POLICY] The customer's last message is not about real estate. "
    "Acknowledge it briefly and warmly, then redirect back to the current "
    "stage of the conversation. Do NOT answer the off-topic question."
)

# ---------------------------------------------------------------------------
# Stage-transition keyword tables  (checked on USER messages only)
# ---------------------------------------------------------------------------

# keyword → canonical category name in INVENTORY
_CATEGORY_MAP: list[tuple[str, str]] = [
    ("shop",      "Shops"),
    ("house",     "Houses/Villas"),
    ("villa",     "Houses/Villas"),
    ("apartment", "Apartments"),
    ("flat",      "Apartments"),
]

# (size_keyword, category) → canonical subtype name
# category=None means the keyword is unambiguous across all categories
_SUBTYPE_MAP: list[tuple[str, Optional[str], str]] = [
    # Shops
    ("5 marla",   "Shops",         "5 Marla Shop"),
    ("8 marla",   "Shops",         "8 Marla Shop"),
    ("1 kanal",   "Shops",         "1 Kanal Shop"),
    # Houses / Villas
    ("5 marla",   "Houses/Villas", "5 Marla House"),
    ("7 marla",   "Houses/Villas", "7 Marla House"),
    ("10 marla",  "Houses/Villas", "10 Marla House"),
    ("1 kanal",   "Houses/Villas", "1 Kanal Villa"),
    # Apartments (bedroom count is unambiguous — no category check needed)
    ("1 bedroom", None,            "1 Bedroom Apt"),
    ("2 bedroom", None,            "2 Bedroom Apt"),
    ("3 bedroom", None,            "3 Bedroom Apt"),
    ("1bed",      None,            "1 Bedroom Apt"),
    ("2bed",      None,            "2 Bedroom Apt"),
    ("3bed",      None,            "3 Bedroom Apt"),
]

# User explicitly requests a booking/visit → subtype_selection → closing
_CLOSING_KW: list[str] = [
    "schedule", "book a visit", "i'd like to visit", "i want to visit",
    "arrange a visit", "speak to an agent", "contact agent", "book agent",
    "i'd like to schedule",
]

# Broad set for off-topic detection
_REALESTATE_KW: list[str] = [
    "shop", "house", "villa", "apartment", "flat", "property", "properties",
    "marla", "kanal", "bedroom", "price", "pkr", "crore", "lac", "lakh",
    "buy", "purchase", "rent", "visit", "agent", "booking", "schedule",
    "real estate", "plot", "area", "size", "category",
    "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye",
    "yes", "no", "okay", "ok", "sure", "please", "show", "tell",
    "more", "info", "interested", "looking",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """All mutable state for one user conversation.

    Attributes
    ----------
    session_id        : Unique UUID for this session.
    history           : Ordered list of {role, content} message dicts.
    stage             : Current position in the conversation flow.
    selected_category : Canonical category name chosen by the user, or None.
    selected_subtype  : Canonical subtype name chosen by the user, or None.
    selected_price    : Price string for the chosen subtype, or None.
    last_active       : Unix timestamp of the last activity (for TTL).
    """

    session_id:        str
    history:           list[dict]     = field(default_factory=list)
    stage:             str            = "greeting"
    selected_category: Optional[str]  = None
    selected_subtype:  Optional[str]  = None
    selected_price:    Optional[str]  = None
    last_active:       float          = field(default_factory=time.time)

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def create_session() -> str:
    """Create a new session, store it, and return its UUID string."""
    sid = str(uuid.uuid4())
    _sessions[sid] = Session(session_id=sid)
    return sid


def get_session(session_id: str) -> Optional[Session]:
    """Return the Session for session_id, or None if expired / not found.

    Triggers a purge of all expired sessions as a side-effect.
    """
    _purge_expired_sessions()
    return _sessions.get(session_id)


def delete_session(session_id: str) -> None:
    """Remove a session from the store immediately."""
    _sessions.pop(session_id, None)

def get_session_info(session_id: str) -> Optional[dict]:
    """Return a JSON-safe summary of session state (no full history)."""
    session = get_session(session_id)
    if session is None:
        return None
    return {
        "session_id":        session.session_id,
        "stage":             session.stage,
        "selected_category": session.selected_category,
        "selected_subtype":  session.selected_subtype,
        "selected_price":    session.selected_price,
        "turn_count":        len(session.history) // 2,
    }

def _purge_expired_sessions() -> None:
    """Delete all sessions inactive longer than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s.last_active > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]

# ---------------------------------------------------------------------------
# Off-topic detection
# ---------------------------------------------------------------------------

def _is_off_topic(message: str) -> bool:
    """Return True when the user message contains no real-estate keywords.

    Messages of three words or fewer are always considered on-topic so that
    short acknowledgements like 'yes', 'ok', 'sure' are never flagged.
    """
    if len(message.strip().split()) <= 3:
        return False
    lower = message.lower()
    return not any(kw in lower for kw in _REALESTATE_KW)

# ---------------------------------------------------------------------------
# Stage tracking + state extraction  (USER messages only)
# ---------------------------------------------------------------------------

def _advance_stage_on_user(session: Session, user_message: str) -> None:
    """Advance session stage and extract chosen category / subtype from the
    USER's message.  The assistant's own text NEVER drives state changes.

    Side-effects
    ------------
    - session.stage             may be advanced one step
    - session.selected_category may be set when the user picks a category
    - session.selected_subtype  may be set when the user picks a size
    - session.selected_price    may be set alongside selected_subtype

    Stages flow strictly one way:
        greeting → category_selection → subtype_selection → closing
    """
    lower = user_message.lower()

    # ── greeting → category_selection ───────────────────────────────────────
    if session.stage == "greeting":
        for kw, canonical in _CATEGORY_MAP:
            if kw in lower:
                session.selected_category = canonical
                session.stage = "category_selection"
                break

    # ── category_selection → subtype_selection ───────────────────────────────
    elif session.stage == "category_selection":
        for size_kw, cat_filter, subtype_name in _SUBTYPE_MAP:
            if size_kw not in lower:
                continue
            # If the subtype is category-specific, only match within the
            # currently selected category so "5 marla" doesn't accidentally
            # match a Shop when the user picked Houses.
            if cat_filter is not None and session.selected_category != cat_filter:
                continue
            session.selected_subtype = subtype_name
            # Look up the price from INVENTORY
            category = cat_filter or session.selected_category or ""
            for name, price in INVENTORY.get(category, []):
                if name == subtype_name:
                    session.selected_price = price
                    break
            session.stage = "subtype_selection"
            break

    # ── subtype_selection → closing ──────────────────────────────────────────
    elif session.stage == "subtype_selection":
        if any(kw in lower for kw in _CLOSING_KW):
            session.stage = "closing"

    # "closing" is terminal

# ---------------------------------------------------------------------------
# Context window management
# ---------------------------------------------------------------------------

def _trimmed_history(session: Session) -> list[dict]:
    """Return at most MAX_HISTORY_TURNS user+assistant pairs from history.

    No greeting pinning is performed here.  Context is preserved through
    the explicit CONVERSATION STATE block in the system prompt instead,
    which is a far more reliable mechanism for small models.
    """
    max_entries = MAX_HISTORY_TURNS * 2   # each turn = one dict
    if len(session.history) > max_entries:
        return list(session.history[-max_entries:])
    return list(session.history)

# ---------------------------------------------------------------------------
# Prompt orchestration
# ---------------------------------------------------------------------------

def _build_conversation_state(session: Session) -> str:
    """Render the CONVERSATION STATE block injected into every system prompt.

    This gives the model explicit, authoritative ground truth about what has
    already been decided so it never needs to infer it from raw history.
    """
    lines = [
        "CONVERSATION STATE  (tracked by the system — treat as ground truth)",
        "-" * 60,
        f"Stage             : {session.stage}",
        f"Category chosen   : {session.selected_category or 'not yet chosen'}",
        f"Subtype chosen    : {session.selected_subtype  or 'not yet chosen'}",
        f"Price confirmed   : {session.selected_price    or 'not yet confirmed'}",
        "-" * 60,
        "IMPORTANT: Do NOT ask the customer again about choices already made above.",
        "           Focus only on the CURRENT GOAL for the current stage.",
    ]
    return "\n".join(lines)


def _build_system_prompt(session: Session, off_topic: bool) -> dict:
    """Build the dynamic system prompt dict sent as messages[0] to Ollama.

    Structure
    ---------
    1. CORE_IDENTITY  — inventory + hard rules            (always present)
    2. CONVERSATION STATE — explicit memory of choices    (always present)
    3. Stage hint     — the model's single goal this turn (always present)
    4. Off-topic reminder                                 (only if flagged)
    """
    parts = [
        CORE_IDENTITY,
        _build_conversation_state(session),
        STAGE_HINTS.get(session.stage, ""),
    ]
    if off_topic:
        parts.append(OFF_TOPIC_REMINDER)
    return {"role": "system", "content": "\n\n".join(parts)}

# ---------------------------------------------------------------------------
# Ollama streaming integration
# ---------------------------------------------------------------------------

async def stream_response(
    session_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Async generator that drives one complete conversational turn.

    Pipeline
    --------
    1.  Validate session.
    2.  Detect off-topic content.
    3.  Advance stage + extract state from USER message.
    4.  Append user turn to history.
    5.  Build: [dynamic_system_prompt] + [trimmed_history].
    6.  Stream Ollama; yield tokens as they arrive.
    7.  Append complete assistant turn to history.

    Yields
    ------
    str
        Individual content tokens, or a single [ERROR] string on failure.
    """
    session = get_session(session_id)
    if session is None:
        yield "[ERROR] Session not found or expired. Please start a new session."
        return

    session.last_active = time.time()

    # ── 1. Off-topic check ───────────────────────────────────────────────────
    off_topic = _is_off_topic(user_message)

    # ── 2. Stage + state extraction (USER only) ──────────────────────────────
    _advance_stage_on_user(session, user_message)

    # ── 3. Append user turn before building the context window ───────────────
    session.history.append({"role": "user", "content": user_message})

    # ── 4. Build final message list ──────────────────────────────────────────
    system_msg = _build_system_prompt(session, off_topic)
    messages = [system_msg] + _trimmed_history(session)

    # ── 5. Stream ─────────────────────────────────────────────────────────────
    client = ollama.AsyncClient()
    full_response: list[str] = []

    try:
        async for chunk in await client.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            think=False,
        ):
            token: str = chunk.message.content or ""
            if token:
                full_response.append(token)
                yield token

    except ollama.ResponseError as exc:
        yield f"\n[ERROR] Ollama ResponseError: {exc.error}"
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        return

    except Exception as exc:  # noqa: BLE001
        yield f"\n[ERROR] Could not reach Ollama: {exc}"
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        return

    # ── 6. Persist assistant turn ─────────────────────────────────────────────
    assistant_turn = {"role": "assistant", "content": "".join(full_response)}
    session.history.append(assistant_turn)
    session.last_active = time.time()

# ---------------------------------------------------------------------------
# Multi-turn smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    TEST_TURNS: list[tuple[str, str]] = [
        ("Turn 1", "Hi"),
        ("Turn 2", "I want to buy a house"),
        ("Turn 3", "Show me 10 marla options"),
        ("Turn 4", "What's the weather today?"),   # ← off-topic test
        ("Turn 5", "I'd like to schedule a visit"),
        ("Turn 6", "Thanks, goodbye"),
    ]

    async def run_test() -> None:
        """Simulate the 6-turn test dialogue on a single session."""
        sid = create_session()
        print("\n=== Ali Real Estate Chatbot — Smoke Test ===")
        print(f"Session ID: {sid}\n")

        for label, user_msg in TEST_TURNS:
            session = get_session(sid)
            if session:
                state_str = (
                    f"stage={session.stage} | "
                    f"category={session.selected_category} | "
                    f"subtype={session.selected_subtype} | "
                    f"price={session.selected_price}"
                )
            else:
                state_str = "session not found"

            print(f"[{label}] User  : {user_msg}")
            print(f"          State : {state_str}")
            print(f"          Ali   : ", end="", flush=True)

            async for token in stream_response(sid, user_msg):
                print(token, end="", flush=True)

            session = get_session(sid)
            if session:
                state_after = (
                    f"stage={session.stage} | "
                    f"category={session.selected_category} | "
                    f"subtype={session.selected_subtype} | "
                    f"price={session.selected_price}"
                )
            else:
                state_after = "session not found"

            print(f"\n          → {state_after}\n")

        print("=== Test Complete ===\n")

    asyncio.run(run_test())