"""
Specialist Subagents for Multi-Agent RecSys Platform.
Implements the Specialist Worker Pattern (Recommendation Specialist & Context Specialist).
"""

import time
from typing import List, Optional
from loguru import logger

from .contracts import (
    RecommendedResult, 
    ContextResult, 
    SubagentExecution, 
    ProductEvidence, 
    UserProfileContext
)
from .tools import RecSysTools


class RecommendationSpecialist:
    """
    Subagent 1: Chuyên viên Gợi ý & Xếp hạng sản phẩm (Recommendation Specialist).
    Trách nhiệm: Nhận user_id -> truy vấn mô hình Recommendation -> trả về Top-K sản phẩm.
    Nguyên tắc: Giữ nguyên thứ tự rank và điểm số từ model backend (Zero-rerank).
    """

    def __init__(self, tools: Optional[RecSysTools] = None):
        self.tools = tools or RecSysTools()

    def execute(self, user_id: str, top_k: int = 5) -> tuple[RecommendedResult, SubagentExecution]:
        start_t = time.perf_counter()
        logger.info(f"[RecommendationSpecialist] Fetching Top-{top_k} recommendations for user: {user_id}")

        result = self.tools.get_recommendations(user_id=user_id, top_k=top_k)
        latency = (time.perf_counter() - start_t) * 1000

        trace = SubagentExecution(
            subagent_name="RecommendationSpecialist",
            action_taken=f"Retrieved {len(result.items)} items via {result.model_name}",
            output_summary={
                "item_count": len(result.items),
                "top_items": [p.item_id for p in result.items]
            },
            latency_ms=round(latency, 2)
        )

        return result, trace


class ContextSpecialist:
    """
    Subagent 2: Chuyên viên Ngữ cảnh & Bằng chứng (Context & Evidence Specialist).
    Trách nhiệm: Thu thập lịch sử hành vi khách hàng và thông tin sản phẩm để làm dẫn chứng giải thích.
    """

    def __init__(self, tools: Optional[RecSysTools] = None):
        self.tools = tools or RecSysTools()

    def execute(self, user_id: str, candidate_item_ids: List[str]) -> tuple[ContextResult, SubagentExecution]:
        start_t = time.perf_counter()
        logger.info(f"[ContextSpecialist] Gathering context for user {user_id} and items {candidate_item_ids}")

        # 1. Kéo profile & chuỗi tương tác gần nhất
        user_profile = self.tools.get_user_profile(user_id=user_id)

        # 2. Kéo thông tin chi tiết từng sản phẩm được đề xuất
        evidences = self.tools.get_product_evidences(item_ids=candidate_item_ids)

        latency = (time.perf_counter() - start_t) * 1000

        trace = SubagentExecution(
            subagent_name="ContextSpecialist",
            action_taken=f"Fetched user history ({len(user_profile.recent_item_sequence)} items) and {len(evidences)} item details",
            output_summary={
                "user_history_count": len(user_profile.recent_item_sequence),
                "evidence_count": len(evidences)
            },
            latency_ms=round(latency, 2)
        )

        context_res = ContextResult(
            user_id=user_id,
            user_profile=user_profile,
            evidences=evidences
        )

        return context_res, trace
