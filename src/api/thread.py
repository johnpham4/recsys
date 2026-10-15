# Đây là cách viết Endpoint chuẩn của công ty & DeerFlow bằng FastAPI:
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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

@router.post("/{thread_id}/stream")
async def chat_stream_endpoint(thread_id, message: str, user_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id
    }

    async def event_generator():
        # Listen event from langgraph
        async for event in recsys_graph.astream_event(inputs, config=config, version="v2"):
            event_type = event["event"]

            # Tool Event 
            if event_type == "on_tool_start":
                payload = {
                    "type": "tool_start",
                    "tool": event["name"],
                    "input": event["data"].get("input")
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        
            # Event LLM response
            elif event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    payload = {
                        "type": "token",
                        "content": chunk.content
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            # 3. Bắt sự kiện Tool chạy xong
            elif event_type == "on_tool_end":
                payload = {
                    "type": "tool_end",
                    "tool": event["name"]
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        # Bắn tín hiệu kết thúc
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")s


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
