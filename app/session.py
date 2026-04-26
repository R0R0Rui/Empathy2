from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import TypedDict
from uuid import uuid4


class Turn(TypedDict):
    role: str
    content: str


class InMemorySessionStore:
    """Thread-safe in-memory conversation history keyed by session ID."""

    def __init__(self, max_turns_per_session: int = 80) -> None:
        self._sessions: dict[str, list[Turn]] = {}
        self._max_turns = max_turns_per_session
        self._lock = RLock()

    def create_session(self) -> str:
        session_id = str(uuid4())
        with self._lock:
            self._sessions[session_id] = []
        return session_id

    def ensure_session(self, session_id: str | None = None) -> str:
        with self._lock:
            if session_id and session_id in self._sessions:
                return session_id
            if session_id:
                self._sessions[session_id] = []
                return session_id
        return self.create_session()

    def get_history(self, session_id: str, limit: int | None = None) -> list[Turn]:
        with self._lock:
            turns = self._sessions.get(session_id, [])
            selected = turns[-limit:] if limit is not None and limit > 0 else turns
            return deepcopy(selected)

    def append_exchange(self, session_id: str, user_message: str, assistant_reply: str) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_reply},
                ]
            )
            if self._max_turns > 0 and len(turns) > self._max_turns:
                del turns[: len(turns) - self._max_turns]

    def reset(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions[session_id] = []
            return existed

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            return count

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)

