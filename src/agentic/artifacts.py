"""
Structured Artifact Generator for RecSys SuperAgent (DeerFlow Signature Feature).
Provides Buying Dossier data contracts, markdown comparison table renderers, and artifact creation tools.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class ProductCard(BaseModel):
    """Thông tin chi tiết và đánh giá chuyên gia cho từng sản phẩm."""
    asin: str
    title: str
    price: float
    category: str
    pros: List[str] = Field(default_factory=list, description="Ưu điểm nổi bật")
    cons: List[str] = Field(default_factory=list, description="Nhược điểm cần lưu ý")
    highlight: str = Field(default="", description="Điểm ăn tiền nhất của sản phẩm")


class BuyingDossier(BaseModel):
    """
    Hồ Sơ Mua Sắm & Thẩm Định Chuyên Sâu (Buying Dossier).
    Artifact cốt lõi của SuperAgent theo kiến trúc ByteDance DeerFlow.
    """
    dossier_id: str = Field(default_factory=lambda: f"dos_{uuid.uuid4().hex[:8]}")
    user_id: str
    category_focus: str = Field(description="Nhu cầu hoặc phân khúc sản phẩm chính")
    budget_max: Optional[float] = Field(default=None, description="Ngân sách trần (USD)")
    products: List[ProductCard] = Field(default_factory=list)
    best_overall_asin: str = Field(description="Mã ASIN của sản phẩm tốt nhất toàn diện")
    best_budget_asin: Optional[str] = Field(default=None, description="Mã ASIN của món có p/p giá tốt nhất")
    executive_summary: str = Field(description="Tóm tắt phân tích ngắn gọn")
    final_verdict: str = Field(description="Lời khuyên chốt hạ của chuyên gia")
    generated_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_markdown(self) -> str:
        """Xuất hồ sơ ra định dạng Markdown bảng so sánh chuyên nghiệp."""
        md = []
        md.append(f"## 📋 HỒ SƠ TƯ VẤN MUA SẮM (BUYING DOSSIER #{self.dossier_id})")
        md.append(f"**Khách hàng:** `{self.user_id}` | **Phân khúc:** **{self.category_focus}**")
        if self.budget_max:
            md.append(f"**Ngân sách tối đa:** `${self.budget_max:.2f}`")
        md.append(f"\n> 💡 **Tóm tắt chuyên gia:** {self.executive_summary}\n")

        # Bảng so sánh
        md.append("### 🔍 Bảng So Sánh Chi Tiết")
        md.append("| Top | Sản Phẩm | Giá ($) | Ưu Điểm Chính | Điểm Cần Cân Nhắc |")
        md.append("| :--- | :--- | :---: | :--- | :--- |")

        for idx, p in enumerate(self.products, 1):
            badge = ""
            if p.asin == self.best_overall_asin:
                badge = " 🏆 *(Best Overall)*"
            elif p.asin == self.best_budget_asin:
                badge = " 💰 *(Best Value)*"

            pros_text = "<br>• ".join(p.pros) if p.pros else "Chất lượng tốt"
            cons_text = "<br>• ".join(p.cons) if p.cons else "Không có nhược điểm lớn"

            # Rút gọn tiêu đề sản phẩm nếu quá dài
            title_short = (p.title[:45] + "...") if len(p.title) > 48 else p.title

            md.append(
                f"| #{idx} | **{title_short}**{badge}<br>`{p.asin}` | `${p.price:.2f}` | • {pros_text} | • {cons_text} |"
            )

        # Lời khuyên chốt hạ
        md.append("\n### 🎯 Lời Khuyên Chốt Hạ (Final Verdict)")
        md.append(f"- **Lựa chọn số 1:** `{self.best_overall_asin}`")
        if self.best_budget_asin:
            md.append(f"- **Lựa chọn tiết kiệm nhất:** `{self.best_budget_asin}`")
        md.append(f"\n{self.final_verdict}\n")

        return "\n".join(md)


@tool
def create_buying_dossier(
    user_id: str,
    category_focus: str,
    executive_summary: str,
    best_overall_asin: str,
    final_verdict: str,
    products: List[Dict[str, Any]],
    budget_max: Optional[float] = None,
    best_budget_asin: Optional[str] = None,
) -> str:
    """
    Tạo Hồ Sơ Thẩm Định Mua Sắm (Buying Dossier Artifact) và Bảng So Sánh Sản Phẩm.
    Gọi tool này sau khi đã tra cứu thông tin để xuất báo cáo so sánh chuẩn mực cho khách hàng.

    Args:
        user_id: Mã người dùng.
        category_focus: Nhu cầu trọng tâm (ví dụ: 'Chuột Gaming Siêu Nhẹ', 'Bàn phím cơ').
        executive_summary: Tóm tắt phân tích ngắn gọn lý do vì sao chọn các món này.
        best_overall_asin: Mã ASIN của sản phẩm khuyến nghị cao nhất.
        final_verdict: Lời khuyên chốt hạ chi tiết vì sao nên mua.
        products: Danh sách các sản phẩm cần đưa vào bảng so sánh. Mỗi item là dict gồm:
                  {'asin': str, 'title': str, 'price': float, 'category': str, 'pros': list, 'cons': list, 'highlight': str}
        budget_max: Ngân sách trần khách đưa ra (nếu có).
        best_budget_asin: Mã ASIN của sản phẩm ngon-bổ-rẻ nhất (nếu có).
    """
    card_items = []
    for item in products:
        card_items.append(ProductCard(
            asin=str(item.get("asin", "")),
            title=str(item.get("title", f"Product {item.get('asin')}")),
            price=float(item.get("price", 29.99)),
            category=str(item.get("category", category_focus)),
            pros=item.get("pros", ["Độ bền cao", "Được đánh giá tốt"]),
            cons=item.get("cons", ["Giá có thể dao động"]),
            highlight=str(item.get("highlight", ""))
        ))

    dossier = BuyingDossier(
        user_id=user_id,
        category_focus=category_focus,
        budget_max=budget_max,
        products=card_items,
        best_overall_asin=best_overall_asin,
        best_budget_asin=best_budget_asin,
        executive_summary=executive_summary,
        final_verdict=final_verdict,
    )

    return dossier.to_markdown()
