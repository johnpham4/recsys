import json
from typing import List, Dict, Any, Optional
from loguru import logger
from src.infra.redis_client import get_redis_client
from .catalog_service import CatalogService


class RecSysService:
    """Service xử lý gợi ý sản phẩm (U2I, I2I, User Recent History) từ Redis Feature Store."""

    def __init__(self, catalog_service: Optional[CatalogService] = None):
        self.redis = get_redis_client()
        self.catalog = catalog_service

    def get_user_recent_interactions(self, user_id: str, limit: int = 5) -> List[str]:
        """Lấy danh sách mã ASIN mà user vừa xem hoặc tương tác gần đây nhất."""
        try:
            raw_seq = self.redis.get(f"feature:user:recent_items:{user_id}")
            if raw_seq:
                items = json.loads(raw_seq)
                return items[-limit:]
        except Exception as e:
            logger.warning(f"Error fetching user history from Redis ({e})")
        return [] 

    def get_i2i_candidates(self, item_id: str, top_k: int = 5) -> List[str]:
        """Lấy các sản phẩm liên quan từ ma trận Item-to-Item (Collaborative Filtering / Word2Vec)."""
        try:
            raw_i2i = self.redis.get(f"output:i2i:{item_id}")
            if raw_i2i:
                rec_ids = json.loads(raw_i2i).get("rec_item_ids", [])
                if rec_ids:
                    return rec_ids[:top_k]
        except Exception as e:
            logger.warning(f"Error fetching I2I from Redis ({e})")
        return []

    def get_popular_candidates(self, top_k: int = 10) -> List[str]:
        """Lấy sản phẩm xu hướng / bán chạy nhất làm fallback."""
        try:
            raw_popular = self.redis.get("output:popular")
            if raw_popular:
                rec_ids = json.loads(raw_popular).get("rec_item_ids", [])
                if rec_ids:
                    return rec_ids[:top_k]
        except Exception as e:
            logger.warning(f"Error fetching popular items from Redis ({e})")
        return []

    def get_personalized_recommendations(
        self,
        user_id: str,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Toàn bộ luồng RecSys:
        1. Lấy lịch sử tương tác gần nhất của User.
        2. Sinh candidate qua I2I từ món cuối cùng (hoặc Popular nếu khách mới).
        3. Ghép nối với CatalogService để lấy thông số chi tiết.
        4. Áp dụng Dynamic Critiquing (lọc theo ngân sách và danh mục).
        """
        # 1. Candidate Retrieval
        history = self.get_user_recent_interactions(user_id, limit=3)
        candidate_asins = []
        if history:
            last_item = history[-1]
            candidate_asins = self.get_i2i_candidates(last_item, top_k=15)

        if not candidate_asins:
            candidate_asins = self.get_popular_candidates(top_k=15)

        # 2. Enrich dữ liệu qua CatalogService
        products = self.catalog.get_products_by_asins(candidate_asins)

        # 3. Dynamic Critiquing / Filtering
        filtered = []
        for p in products:
            if max_price is not None and p["price"] > max_price:
                continue
            if category and category.lower() not in p["category"].lower():
                continue
            filtered.append(p)
            if len(filtered) >= top_k:
                break

        # Nếu lọc quá chặt không còn sản phẩm nào, trả về top candidates gốc
        return filtered if filtered else products[:top_k]
