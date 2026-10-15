import json
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from src.infra.db_client import get_engine
from .model import SessionEventDTO, derive_messages

class SessionManager:
    def __init__(self):
        self.engine = get_engine()
        self.schema = os.getenv("AGENTIC_SYSTEM_SCHEMA", "agentic_system")

    def get_or_create_session(self, thread_id: str, user_id: str, title: Optional[str] = None):
        """Khởi tạo phiên chat nếu chưa có."""
        query = text(f"""
            INSERT INTO {self.schema}.sessions (thread_id, user_id, title)
            VALUES (:thread_id, :user_id, :title)
            ON CONFLICT (thread_id) DO NOTHING
        """)
        with self.engine.begin() as conn:
            conn.execute(query, {"thread_id": thread_id, "user_id": user_id, "title": title or "New Shopping Research"})

    def append_event(
        self,
        thread_id: str,
        turn_id: str,
        step_index: int,
        event_type: str,
        payload: Dict[str, Any]
    ):
        """Chèn 1 Event mới vào bảng session_events (Append-Only Log chuẩn DeepSeek)."""
        query = text(f"""
            INSERT INTO {self.schema}.session_events (thread_id, turn_id, step_index, event_type, payload)
            VALUES (:thread_id, :turn_id, :step_index, :event_type, :payload)
        """)
        with self.engine.begin() as conn:
            conn.execute(query, {
                "thread_id": thread_id,
                "turn_id": turn_id,
                "step_index": step_index,
                "event_type": event_type,
                "payload": json.dumps(payload, ensure_ascii=False)
            })

    def get_trajectory(self, thread_id: str) -> List[Dict[str, Any]]:
        """Đọc toàn bộ events từ DB và chiếu (project) thành messages cho LLM đọc."""
        query = text(f"""
            SELECT event_id, thread_id, turn_id, step_index, event_type, payload, created_at
            FROM {self.schema}.session_events
            WHERE thread_id = :thread_id
            ORDER BY event_id ASC
        """)
        events = []
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"thread_id": thread_id}).fetchall()
            for r in rows:
                raw_payload = r.payload if isinstance(r.payload, dict) else json.loads(r.payload)
                events.append(SessionEventDTO(
                    event_id=r.event_id,
                    thread_id=r.thread_id,
                    turn_id=r.turn_id,
                    step_index=r.step_index,
                    event_type=r.event_type,
                    payload=raw_payload,
                    created_at=r.created_at
                ))
        # Dùng hàm derive_messages bạn vừa viết để trả về format cho LLM!
        return derive_messages(events)
