"""
LangGraph ReAct Engine for Amazon RecSys SuperAgent.
Implements StateGraph with ToolNode, conditional routing, SQLite persistent checkpointer,
and DeerFlow Buying Dossier artifact generation.
"""

import os
import sqlite3
from pathlib import Path
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from .state import RecSysAgentState
from .prompts import SUPERAGENT_SYSTEM_PROMPT
from .tools import AGENT_TOOLS
from .artifacts import create_buying_dossier

# Toàn bộ Tools: Truy vấn RecSys/DB + Tạo Buying Dossier Artifact
ALL_TOOLS = list(AGENT_TOOLS) + [create_buying_dossier]
MAX_ITERATIONS = 5


def get_checkpointer():
    """Khởi tạo SQLite Checkpointer để lưu thread bền vững qua các lần restart server."""
    try:
        db_path = Path("data/agent_threads.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        logger.info(f"LangGraph Thread Checkpointer connected to {db_path}")
        return SqliteSaver(conn)
    except Exception as e:
        logger.warning(f"Failed to initialize SqliteSaver ({e}). Falling back to MemorySaver.")
        return MemorySaver()


# 1. Khởi tạo LLM và kết nối vào Tools
def get_llm():
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROD_API_KEY") or os.getenv("OPENAI_API_KEY", "mock-key")
    is_groq = "gsk_" in api_key or bool(os.getenv("GROQ_API_KEY")) or bool(os.getenv("GROD_API_KEY"))
    base_url = "https://api.groq.com/openai/v1" if is_groq else None

    # Tự động chọn mô hình Groq hỗ trợ Function Calling tốt nhất hiện có
    default_model = "openai/gpt-oss-120b" if is_groq else "gpt-4o-mini"
    model_name = os.getenv("LLM_MODEL_NAME", default_model)

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )
    return llm.bind_tools(ALL_TOOLS)


# 2. Node Agent: LLM tư duy và ra quyết định gọi Tool
def agent_node(state: RecSysAgentState):
    model = get_llm()
    messages = list(state["messages"])

    # Nạp System Prompt vào đầu danh sách nếu chưa có
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SUPERAGENT_SYSTEM_PROMPT)] + messages

    response = model.invoke(messages)
    return {"messages": [response]}

def event_loop(state: List[BaseMessage]) -> str:
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

# 3. Lắp ráp Đồ thị LangGraph
def create_recsys_graph(checkpointer=None):
    builder = StateGraph(RecSysAgentState)

    # Thêm Node Agent và Node Tools
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("planner", planner_node)

    # Cạnh điều hướng chu trình ReAct
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    # Gắn Checkpointer (Mặc định dùng SQLite bền vững)
    cp = checkpointer if checkpointer is not None else get_checkpointer()
    return builder.compile(checkpointer=cp)


# Tạo sẵn instance đồ thị để phục vụ API
recsys_graph = create_recsys_graph()
