"""
LangGraph State Definition for RecSys SuperAgent.
Defines the shared state passed between the Agent Node and Tool Nodes.
"""
from typing import Dict, Any, List
from langgraph.graph import MessagesState


class RecSysAgentState(MessagesState):
    """
    State (chiếc balo) truyền qua lại giữa các Node trong đồ thị LangGraph:
    - messages: Kế thừa sẵn từ MessagesState (tự động cộng dồn lịch sử chat).
    - user_id: ID khách hàng cần đề xuất sản phẩm.
    - context: Bộ nhớ tạm lưu các tiêu chí người dùng yêu cầu (giá, hãng, loại đồ).
    - artifacts: Danh sách báo cáo mua sắm (Buying Dossier) sinh ra.
    """
    user_id: str
    context: Dict[str, Any] = {}
    artifacts: List[Dict[str, Any]] = []
    plan: List[str] = []
