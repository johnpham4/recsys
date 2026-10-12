# Đây là cách viết Endpoint chuẩn của công ty & DeerFlow bằng FastAPI:
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.agentic.graph import recsys_graph


router = APIRouter(prefix="/api/threads", tags=["Agent Threads"])


class SendMessageRequest(BaseModel):
    user_id: str
    message: str
@router.post("/{thread_id}/chat")
async def chat_in_thread(thread_id: str, request: SendMessageRequest):
    """
    Endpoint nhận tin nhắn từ web và đẩy vào đúng thread_id trong LangGraph!
    """
    # 1. Tra chìa khóa thread_id vào config của LangGraph
    config = {"configurable": {"thread_id": thread_id}}
    
    # 2. Gọi LangGraph chạy
    result = recsys_graph.invoke(
        {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id
        },
        config=config
    )
    
    # 3. Lấy tin nhắn cuối cùng của AI trả về cho Frontend
    ai_reply = result["messages"][-1].content
    return {
        "thread_id": thread_id,
        "reply": ai_reply
    }


@router.get("/{thread_id}")
async def get_thread_history(thread_id: str):
    """Lấy lại toàn bộ tin nhắn cũ của thread này để hiển thị lên màn hình chat."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Đọc trực tiếp từ Checkpointer của LangGraph:
    state = recsys_graph.get_state(config)
    
    if not state.values:
        raise HTTPException(status_code=404, detail="Thread không tồn tại")
        
    messages = state.values.get("messages", [])
    return {
        "thread_id": thread_id,
        "messages": [{"role": m.type, "content": m.content} for m in messages]
    }
