"""
LangGraph ReAct Engine for Amazon RecSys SuperAgent.
Implements StateGraph with ToolNode, conditional routing, and checkpointer.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from .state import RecSysAgentState
from .prompts import SUPERAGENT_SYSTEM_PROMPT
from .tools import AGENT_TOOLS
# 1. Khởi tạo LLM và kết nối vào Tools
def get_llm():
    # Tự động nhận Groq API Key từ file .env (hoặc fallback sang OpenAI)
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROD_API_KEY") or os.getenv("OPENAI_API_KEY", "mock-key")
    is_groq = "gsk_" in api_key or bool(os.getenv("GROQ_API_KEY")) or bool(os.getenv("GROD_API_KEY"))
    base_url = "https://api.groq.com/openai/v1" if is_groq else None
    model_name = "llama-3.3-70b-versatile" if is_groq else "gpt-4o-mini"
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2
    )
    # Gắn danh sách 3 tools Redis vào LLM
    return llm.bind_tools(AGENT_TOOLS)


# 2. Node Agent: Nơi LLM suy nghĩ và ra quyết định gọi Tool
def agent_node(state: RecSysAgentState):
    model = get_llm()
    messages = list(state["messages"])
    
    # Nạp System Prompt vào đầu danh sách nếu chưa có
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SUPERAGENT_SYSTEM_PROMPT)] + messages
    response = model.invoke(messages)
    return {"messages": [response]}


# 3. Lắp ráp Đồ thị LangGraph
def create_recsys_graph(checkpointer=None):
    builder = StateGraph(RecSysAgentState)
    # Thêm 2 Node chính
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(AGENT_TOOLS))
    # Nối các Cạnh (Edges) tạo thành chu trình ReAct
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition) # Nếu LLM đòi gọi tool -> sang tools, nếu xong -> kết thúc
    builder.add_edge("tools", "agent")                      # Chạy tool xong quay lại cho LLM đọc kết quả
    # Gắn Checkpointer để tự động quản lý Thread
    cp = checkpointer or MemorySaver()
    return builder.compile(checkpointer=cp)
# Tạo sẵn 1 instance graph để sử dụng
recsys_graph = create_recsys_graph()
