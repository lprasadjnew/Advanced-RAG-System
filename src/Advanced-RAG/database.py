"""SQLite conversation memory store."""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Integer,
    create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///./conversations.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), index=True, nullable=False)
    role = Column(String(10), nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


# ── Session helpers ────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return str(uuid.uuid4())


def save_message(session_id: str, role: str, content: str) -> None:
    with SessionLocal() as db:
        db.add(Message(session_id=session_id, role=role, content=content))
        db.commit()


def get_conversation(session_id: str) -> list[dict]:
    """Return messages for a session as list of {role, content} dicts."""
    with SessionLocal() as db:
        rows = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at)
            .all()
        )
        return [{"role": r.role, "content": r.content} for r in rows]


def get_latest_session() -> str | None:
    """Return the most-recently-used session_id, or None if no history."""
    with SessionLocal() as db:
        row = (
            db.query(Message.session_id)
            .order_by(Message.created_at.desc())
            .first()
        )
        return row[0] if row else None


def clear_session(session_id: str) -> None:
    with SessionLocal() as db:
        db.query(Message).filter(Message.session_id == session_id).delete()
        db.commit()
