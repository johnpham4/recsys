import os
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from loguru import logger
from src.infra.db_client import get_engine
from .catalog_service import CatalogService


class CartService:
    """Service xử lý Giỏ hàng và Tinh chỉnh Sở thích (User Preferences) trong schema agentic_system."""

    def __init__(self, catalog_service: Optional[CatalogService] = None):
        self.engine = get_engine()
        self.schema = os.getenv("AGENTIC_SYSTEM_SCHEMA", "agentic_system")
        self.catalog = catalog_service or CatalogService()

    def add_to_cart(self, user_id: str, asin: str, quantity: int = 1) -> Dict[str, Any]:
        """Thêm sản phẩm vào giỏ hàng hoặc tăng số lượng nếu đã có."""
        query = text(f"""
            INSERT INTO {self.schema}.user_cart (user_id, asin, quantity, updated_at)
            VALUES (:user_id, :asin, :quantity, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, asin) 
            DO UPDATE SET 
                quantity = {self.schema}.user_cart.quantity + :quantity,
                updated_at = CURRENT_TIMESTAMP
            RETURNING cart_id, quantity;
        """)

        try:
            with self.engine.begin() as conn:
                row = conn.execute(query, {
                    "user_id": user_id,
                    "asin": asin,
                    "quantity": max(1, quantity)
                }).fetchone()

            product_info = self.catalog.get_product_by_asin(asin)
            title = product_info.get("title", asin) if product_info else asin

            return {
                "status": "success",
                "message": f"Đã thêm '{title}' vào giỏ hàng thành công.",
                "asin": asin,
                "current_quantity": row[1] if row else quantity
            }
        except Exception as e:
            logger.error(f"Error adding to cart: {e}")
            return {
                "status": "error",
                "message": f"Không thể thêm vào giỏ hàng do lỗi hệ thống: {str(e)}"
            }

    def get_cart(self, user_id: str) -> List[Dict[str, Any]]:
        """Lấy danh sách các mặt hàng trong giỏ của user."""
        query = text(f"""
            SELECT asin, quantity, created_at
            FROM {self.schema}.user_cart
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
        """)
        items = []
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(query, {"user_id": user_id}).fetchall()
                asins = [r[0] for r in rows]
                products_map = {p["asin"]: p for p in self.catalog.get_products_by_asins(asins)}

                for r in rows:
                    asin = r[0]
                    p = products_map.get(asin, {})
                    items.append({
                        "asin": asin,
                        "title": p.get("title", f"Sản phẩm {asin}"),
                        "price": p.get("price", 0.0),
                        "quantity": r[1]
                    })
        except Exception as e:
            logger.error(f"Error fetching cart: {e}")
        return items

    def record_user_preference(
        self,
        user_id: str,
        pref_key: str,
        pref_value: str,
        sentiment: str = "positive"
    ) -> Dict[str, Any]:
        """Ghi nhận sở thích/yêu cầu loại trừ của người dùng qua đàm thoại (Dynamic Critiquing)."""
        query = text(f"""
            INSERT INTO {self.schema}.user_preferences (user_id, pref_key, pref_value, sentiment)
            VALUES (:user_id, :pref_key, :pref_value, :sentiment)
            RETURNING pref_id;
        """)
        try:
            with self.engine.begin() as conn:
                row = conn.execute(query, {
                    "user_id": user_id,
                    "pref_key": pref_key.lower().strip(),
                    "pref_value": pref_value.lower().strip(),
                    "sentiment": sentiment.lower().strip()
                }).fetchone()

            sentiment_text = "thích" if sentiment == "positive" else "muốn tránh"
            return {
                "status": "success",
                "message": f"Đã ghi nhớ bạn {sentiment_text} đặc điểm '{pref_key}: {pref_value}'."
            }
        except Exception as e:
            logger.error(f"Error recording preference: {e}")
            return {"status": "error", "message": str(e)}
