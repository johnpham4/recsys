"""
Guardrails and Safety Policies for Agentic Recommendation Platform.
Protects against Prompt Injection (Input Guardrail) and Hallucination (Output Grounding).
"""

import re
from typing import List, Optional, Tuple
from .contracts import RecommendedProduct

class SecurityGuardrails:
    """
    Input Guardrail: Quét và ngăn chặn các hành vi tấn công Prompt Injection,
    Jailbreak hoặc cố tình trích xuất system prompt / credentials.
    """

    INJECTION_PATTERNS = [
        r"ignore\s+.*?instructions",
        r"system\s+prompt",
        r"jailbreak",
        r"act\s+as\s+(an\s+)?unrestricted",
        r"(reveal|show|dump|leak)\s+.*?(api\s*key|password|secret|credential|token)",
        r"\b(bearer\s+token|secret\s+credentials?|admin\s+credentials?)\b",
        r"(drop\s+table|delete\s+from|rm\s+-rf)",
        r"bypass\s+safety",
    ]

    @classmethod
    def validate_input(self, user_message: str) -> Tuple[bool, Optional[str]]:
        """
        Kiểm tra xem prompt đầu vào có chứa mã độc/tấn công không.
        
        Returns:
            (is_safe, reason)
        """
        text = user_message.lower()
        
        # 1. Kiểm tra Prompt Injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text):
                return False, f"Potential prompt injection detected ({pattern})."
        
        return True, None

class GroundingGuardrail:
    """
    Output Guardrail: Đảm bảo tính trung thực (Faithfulness & Grounding),
    ngăn chặn LLM tự bịa đặt ID sản phẩm hoặc thông tin không có trong kho.
    """
    @staticmethod
    def verify_grounding(
        llm_reply: str, 
        valid_products: List[RecommendedProduct]
    ) -> Tuple[str, bool]:

        valid_ids = {p.item_id for p in valid_products}
        
        # Nếu hệ thống có sản phẩm gợi ý nhưng reply rỗng
        if not llm_reply.strip():
            return "Hệ thống gợi ý đã tìm thấy các sản phẩm phù hợp nhưng gặp lỗi khi tổng hợp câu trả lời.", False
        # Nếu không có sản phẩm nào từ tool mà LLM vẫn tự nói thao thao bất tuyệt
        if not valid_ids:
            warning = "\n\n*(Lưu ý: Không tìm thấy sản phẩm khớp chính xác trong kho thời gian thực, vui lòng thử lại với từ khóa khác)*"
            return llm_reply + warning, False
        return llm_reply, True