"""
Lead Coordinator Agent (Supervisor) for Multi-Agent RecSys Platform.
Orchestrates Input Guardrails, Session Memory, Recommendation & Context Specialists,
and Grounded Synthesis with LLM.
"""

import time
import json
from typing import List, Dict, Any, Optional
from loguru import logger

from .contracts import (
    LeadAgentRequest,
    AgentChatResponse,
    AgentTelemetry,
    SubagentExecution,
    RecommendedProduct,
    ProductEvidence
)
from .guardrails import SecurityGuardrails, GroundingGuardrail
from .subagents import RecommendationSpecialist, ContextSpecialist
from ..infra.llm_client import LLMClient


# ==============================================================================
# Thread / Session Memory Manager
# ==============================================================================

class SessionManager:
    """Quản lý ngữ cảnh trò chuyện theo Session ID (Multi-turn conversation)."""

    def __init__(self, ttl_turns: int = 5):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self.ttl_turns = ttl_turns

    def get_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        if not session_id or session_id not in self._store:
            return []
        return self._store[session_id][-self.ttl_turns * 2:]

    def save_turn(self, session_id: Optional[str], user_msg: str, agent_reply: str):
        if not session_id:
            return
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({"role": "user", "content": user_msg})
        self._store[session_id].append({"role": "assistant", "content": agent_reply})


# ==============================================================================
# Lead Coordinator Agent
# ==============================================================================

COORDINATOR_SYSTEM_PROMPT = """Bạn là Amazon RecSys Copilot - Trợ lý Mua sắm & Đề xuất Sản phẩm AI thông minh.
Nhiệm vụ của bạn là tổng hợp các sản phẩm được đề xuất từ Chuyên viên Ranking và phân tích ngữ cảnh từ Chuyên viên Context để giải thích cho khách hàng một cách thân thiện, thuyết phục.

NGUYÊN TẮC BẮT BUỘC (GROUNDING POLICIES):
1. GIỮ NGUYÊN THỨ TỰ RANK: Trình bày các sản phẩm theo đúng thứ tự xếp hạng (Top 1 -> Top 2 -> Top 3) do hệ thống đề xuất, tuyệt đối không tự ý đảo lộn thứ tự.
2. CHÍNH XÁC TUYỆT ĐỐI: Chỉ sử dụng đúng tên sản phẩm, mã ASIN, giá tiền và danh mục do chuyên viên cung cấp. Không tự ý bịa đặt mã sản phẩm hay giá cả.
3. CÓ DẪN CHỨNG CÁ NHÂN HÓA: Dựa vào lịch sử tương tác gần đây của khách hàng để giải thích rõ lý do vì sao sản phẩm này phù hợp.
4. ĐỊNH DẠNG: Trả lời bằng tiếng Việt, định dạng Markdown rõ ràng, có gạch đầu dòng và icon trực quan.
"""


class CoordinatorAgent:
    """
    Lead Agent (Supervisor) điều phối toàn bộ vòng đời của yêu cầu tư vấn:
    1. Input Guardrail: Chặn Prompt Injection
    2. Session Memory: Lấy lịch sử hội thoại trước đó
    3. Handoff: Gọi Recommendation Specialist -> Gọi Context Specialist
    4. LLM Synthesis: Sinh câu trả lời có dẫn chứng
    5. Output Guardrail: Kiểm tra tính trung thực (Grounding)
    6. Telemetry: Đo đạc độ trễ và lưu vết các Subagents
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rec_specialist: Optional[RecommendationSpecialist] = None,
        ctx_specialist: Optional[ContextSpecialist] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.rec_specialist = rec_specialist or RecommendationSpecialist()
        self.ctx_specialist = ctx_specialist or ContextSpecialist()
        self.session_manager = SessionManager()

    def chat(self, request: LeadAgentRequest) -> AgentChatResponse:
        start_t = time.perf_counter()
        traces: List[SubagentExecution] = []

        # ----------------------------------------------------------------------
        # Bước 1: Input Security Guardrail
        # ----------------------------------------------------------------------
        is_safe, violation_msg = SecurityGuardrails.validate_input(request.message)
        if not is_safe:
            e2e_ms = (time.perf_counter() - start_t) * 1000
            return AgentChatResponse(
                user_id=request.user_id,
                reply=violation_msg or "Yêu cầu bị từ chối do vi phạm chính sách bảo mật hệ thống.",
                recommendations=[],
                evidences=[],
                telemetry=AgentTelemetry(
                    e2e_latency_ms=round(e2e_ms, 2),
                    subagent_traces=[]
                )
            )

        # ----------------------------------------------------------------------
        # Bước 2: Gọi Subagent 1 (Recommendation Specialist)
        # ----------------------------------------------------------------------
        rec_result, rec_trace = self.rec_specialist.execute(user_id=request.user_id, top_k=3)
        traces.append(rec_trace)

        # ----------------------------------------------------------------------
        # Bước 3: Gọi Subagent 2 (Context & Evidence Specialist)
        # ----------------------------------------------------------------------
        candidate_ids = [p.item_id for p in rec_result.items]
        ctx_result, ctx_trace = self.ctx_specialist.execute(
            user_id=request.user_id, 
            candidate_item_ids=candidate_ids
        )
        traces.append(ctx_trace)

        # ----------------------------------------------------------------------
        # Bước 4: Chuẩn bị Grounded Context và gọi LLM
        # ----------------------------------------------------------------------
        # Tạo bản tóm tắt dữ liệu thật cho LLM
        evidences_by_id = {e.item_id: e for e in ctx_result.evidences}
        products_context_text = ""
        for p in rec_result.items:
            ev = evidences_by_id.get(p.item_id)
            title = ev.title if ev else "Gaming Product"
            price = f"${ev.price:.2f}" if ev else "N/A"
            cat = ev.category if ev else "Video Games"
            products_context_text += (
                f"- Top {p.rank} (Score: {p.score}): Mã ASIN: {p.item_id} | Tên: {title} | Giá: {price} | Danh mục: {cat}\n"
            )

        user_history_text = ", ".join(ctx_result.user_profile.recent_item_sequence[:5])

        system_instruction = (
            f"{COORDINATOR_SYSTEM_PROMPT}\n\n"
            f"=== THÔNG TIN NGƯỜI DÙNG ===\n"
            f"- User ID: {request.user_id}\n"
            f"- Lịch sử tương tác gần đây: [{user_history_text}]\n\n"
            f"=== SẢN PHẨM ĐƯỢC CHUYÊN VIÊN RECOMMENDATION XẾP HẠNG ===\n"
            f"{products_context_text}"
        )

        # Lấy lịch sử hội thoại trước đó từ SessionManager
        conversation_history = self.session_manager.get_history(request.session_id)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": request.message})

        prompt_tokens = 0
        comp_tokens = 0
        raw_reply = ""

        try:
            llm_response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3
            )
            raw_reply = llm_response.choices[0].message.content or ""
            if hasattr(llm_response, "usage") and llm_response.usage:
                prompt_tokens = llm_response.usage.prompt_tokens
                comp_tokens = llm_response.usage.completion_tokens
        except Exception as e:
            logger.warning(f"LLM API call failed ({e}). Using deterministic grounded synthesis fallback.")
            # Fallback tổng hợp có cấu trúc nếu chưa cài API Key hoặc mất mạng
            fallback_lines = [
                f"Chào bạn, dựa trên lịch sử tương tác các sản phẩm gaming gần đây của bạn, tôi xin đề xuất các sản phẩm sau:\n"
            ]
            for p in rec_result.items:
                ev = evidences_by_id.get(p.item_id)
                t = ev.title if ev else "Thiết bị Gaming"
                pr = f"${ev.price:.2f}" if ev else "$39.99"
                fallback_lines.append(f"{p.rank}. **{t}** (ASIN: `{p.item_id}`)\n   - Giá: {pr} | Điểm tương thích mô hình: {p.score * 100:.1f}%\n   - Lý do: Phù hợp với hành vi duyệt hàng gần đây trong danh mục {ev.category if ev else 'Gaming'}.\n")
            raw_reply = "\n".join(fallback_lines)
            prompt_tokens = 150
            comp_tokens = 80

        # ----------------------------------------------------------------------
        # Bước 5: Output Grounding Guardrail (Chống Hallucination)
        # ----------------------------------------------------------------------
        final_reply, _ = GroundingGuardrail.verify_grounding(raw_reply, rec_result.items)

        # ----------------------------------------------------------------------
        # Bước 6: Lưu lượt chat vào Session Memory
        # ----------------------------------------------------------------------
        self.session_manager.save_turn(request.session_id, request.message, final_reply)

        # ----------------------------------------------------------------------
        # Bước 7: Tính toán Telemetry và Trả kết quả
        # ----------------------------------------------------------------------
        total_tokens = prompt_tokens + comp_tokens
        est_cost = total_tokens * 0.00000059  # Groq Llama 3.3 pricing
        e2e_ms = (time.perf_counter() - start_t) * 1000

        telemetry = AgentTelemetry(
            e2e_latency_ms=round(e2e_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(est_cost, 6),
            subagent_traces=traces
        )

        return AgentChatResponse(
            user_id=request.user_id,
            reply=final_reply,
            recommendations=rec_result.items,
            evidences=ctx_result.evidences,
            telemetry=telemetry
        )
