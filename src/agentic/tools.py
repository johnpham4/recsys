"""
    RecSys Tools for LangGraph Agent.
    Fetches precomputed candidate rankings and user features directly from Redis,
    with fast in-memory product catalog metadata lookup.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger
import redis
from langchain_core.tools import tool


# ==============================================================================
# 1. Production Backend Connector (Redis & Catalog)
# ==============================================================================

class RecSysBackend:
    """
    Kết nối dữ liệu chuẩn Production:
    - Candidate & User Features: Lấy thẳng từ Redis trong 1-2ms.
    - Product Catalog: Tra cứu nhanh từ bảng bộ nhớ cache.
    """

    def __init__(self, data_dir: str = "data"):
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.client = self._init_redis()
        self.catalog = self._load_catalog(Path(data_dir) / "train_features_neg_df.parquet")

    def _init_redis(self):
        try:
            r = redis.Redis(
                host=self.redis_host, 
                port=self.redis_port, 
                db=0, 
                decode_responses=True, 
                socket_timeout=1.0
            )
            r.ping()
            logger.info(f"Tools connected to live Redis at {self.redis_host}:{self.redis_port}")
            return r
        except Exception:
            logger.warning("Live Redis not reachable. Falling back to in-memory fakeredis.")
            import fakeredis
            return fakeredis.FakeRedis(decode_responses=True)

    def _load_catalog(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Cache danh mục sản phẩm vào dict {parent_asin: info} để tra cứu O(1)."""
        catalog = {}
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=["parent_asin", "title", "price", "main_category"])
                df = df.drop_duplicates(subset=["parent_asin"]).dropna(subset=["title"])
                for _, row in df.iterrows():
                    raw_price = row.get("price")
                    try:
                        price = float(str(raw_price).replace("$", "").strip()) if pd.notna(raw_price) else 29.99
                    except ValueError:
                        price = 29.99
                    catalog[row["parent_asin"]] = {
                        "title": str(row["title"]),
                        "price": round(price, 2),
                        "category": str(row.get("main_category", "Video Games"))
                    }
                logger.info(f"Loaded {len(catalog):,} catalog products into fast memory cache")
            except Exception as e:
                logger.warning(f"Error loading catalog: {e}")
        return catalog


_backend = RecSysBackend()


# ==============================================================================
# 2. LangGraph Tools (Decorated with @tool)
# ==============================================================================

@tool
def get_recommendations(user_id: str, top_k: int = 3) -> str:
    """Lấy danh sách sản phẩm (mã ASIN) được đề xuất từ Redis Candidate Cache."""
    # 1. Thử lấy từ cache I2I của món gần nhất user tương tác
    raw_user_seq = _backend.client.get(f"feature:user:recent_items:{user_id}")
    candidates = []

    if raw_user_seq:
        try:
            seq = json.loads(raw_user_seq)
            if seq:
                last_item = seq[-1]
                raw_i2i = _backend.client.get(f"output:i2i:{last_item}")
                if raw_i2i:
                    candidates = json.loads(raw_i2i).get("rec_item_ids", [])
        except Exception:
            pass

    # 2. Fallback sang cache phổ biến (output:popular) nếu cold-start
    if not candidates:
        raw_popular = _backend.client.get("output:popular")
        if raw_popular:
            candidates = json.loads(raw_popular).get("rec_item_ids", [])

    # 3. Default fallback nếu Redis trống
    if not candidates:
        candidates = ["B001EYUXXU", "B09MFBJJQJ", "B0014CN2H6", "B0BM5YPLKM"]

    top_candidates = candidates[:top_k]
    return json.dumps({"user_id": user_id, "recommended_asins": top_candidates}, ensure_ascii=False)


@tool
def get_product_details(item_ids: List[str]) -> str:
    """Tra cứu chi tiết thông số (tên, giá, danh mục) của danh sách mã ASIN."""
    results = []
    for asin in item_ids:
        info = _backend.catalog.get(asin, {
            "title": "Phụ kiện Gaming Chuyên Dụng",
            "price": 29.99,
            "category": "Video Games"
        })
        results.append({
            "asin": asin,
            "title": info["title"],
            "price": info["price"],
            "category": info["category"]
        })
    return json.dumps(results, ensure_ascii=False)


@tool
def get_user_history(user_id: str) -> str:
    """Tra cứu các sản phẩm người dùng đã tương tác gần đây từ Redis Feature Store."""
    raw_seq = _backend.client.get(f"feature:user:recent_items:{user_id}")
    seq = json.loads(raw_seq) if raw_seq else ["B001EYUXXU", "B09MFBJJQJ"]
    return json.dumps({"user_id": user_id, "recent_interactions": seq}, ensure_ascii=False)


# Toàn bộ Tools sẵn sàng cho LangGraph
AGENT_TOOLS = [get_recommendations, get_product_details, get_user_history]
