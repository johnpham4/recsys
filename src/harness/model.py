from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class Message(BaseModel):
    thread_id: str
    user_id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None


class ToolCallDTO(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]


class SessionEventDTO(BaseModel):
    event_id: Optional[int] = None
    thread_id: str
    turn_id: str
    step_index: int = 0
    event_type: Literal[
        "user/message", 
        "assistant/tool_call", 
        "tool/result", 
        "assistant/message"
    ]
    payload: Dict[str, Any] 
    created_at: Optional[datetime] = None


class ChatMessageDTO(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


def derive_messages(events: List[SessionEventDTO]) -> List[Dict[str, Any]]:
    messages = []
    for e in events:
        if e.event_type == "user/message":
            messages.append({
                "role": "user",
                "content": e.payload.get("content")
            })
        
        elif e.event_type == "assistant/tool_call":
            messages.append({
                "role": "assistant",
                "content": e.payload.get("content"),
                "tool_calls": e.payload.get("tool_calls")
            })

        elif e.event_type == "tool/result":
            messages.append({
                "role": "tool",
                "tool_call_id": e.payload.get("tool_call_id"),
                "content": str(e.payload.get("output"))
            })

        elif e.event_type == "assistant/message":
            messages.append({"role": "assistant", "content": e.payload.get("content")})
            
    return messages
