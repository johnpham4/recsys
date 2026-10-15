import os
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from loguru import logger
from src.infra.db_client import get_engine


class CatalogService:
    """Service xử lý dữ liệu sản phẩm từ PostgreSQL OLTP."""

    def __init__(self):
        self.engine = get_engine()
        self.schema = os.getenv("POSTGRES_OLTP_SCHEMA", "oltp")
        self.table = "amz_review_rating_raw"

    def get_products_by_asins(self, asins: List[str]) -> List[Dict[str, Any]]:
        """Lấy thông tin chi tiết danh sách sản phẩm theo mã ASIN."""
        if not asins:
            return []

        query = text(f"""
            SELECT parent_asin, title, price, main_category, description
            FROM {self.schema}.{self.table}
            WHERE parent_asin = ANY(:asins)
        """)

        results = []
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(query, {"asins": asins}).fetchall()
                for row in rows:
                    results.append({
                        "asin": row[0],
                        "title": row[1] or "Sản phẩm",
                        "price": self._parse_price(row[2]),
                        "category": row[3] or "Electronics",
                        "description": str(row[4])[:300] if row[4] else "Sản phẩm chất lượng cao."
                    })
        except Exception as e:
            logger.warning(f"Failed to query catalog from PostgreSQL ({e}). Using mock/fallback.")

        # Fallback cho các ASIN không tìm thấy trong DB để Agent không bị crash
        found_asins = {r["asin"] for r in results}
        for asin in asins:
            if asin not in found_asins:
                results.append({
                    "asin": asin,
                    "title": f"Sản phẩm {asin}",
                    "price": 39.99,
                    "category": "Electronics",
                    "description": "Chi tiết sản phẩm đang được cập nhật."
                })

        return results

    def get_product_by_asin(self, asin: str) -> Optional[Dict[str, Any]]:
        """Tra cứu 1 sản phẩm duy nhất."""
        products = self.get_products_by_asins([asin])
        return products[0] if products else None

    @staticmethod
    def _parse_price(raw_price: Any) -> float:
        if raw_price is None:
            return 29.99
        try:
            return round(float(str(raw_price).replace("$", "").strip()), 2)
        except (ValueError, TypeError):
            return 29.99
