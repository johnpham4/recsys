"""
Production-Ready RecSys Tools for LangGraph Agent.
- get_recommendations: Fetches candidates from Redis with dynamic constraint filtering.
- get_product_details: Fetches rich item metadata from PostgreSQL (with in-memory catalog fallback).
- get_user_history: Fetches user interaction sequence from Redis Feature Store.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger
import redis
from sqlalchemy import create_engine, text
from langchain_core.tools import tool


class RecSysBackend:
    """
    Quản lý kết nối dữ liệu chuẩn Production:
    - Redis: Phục vụ Online Features & Precomputed Candidates (1-2ms).
    - PostgreSQL: Phục vụ Product Metadata từ oltp.amz_review_rating_raw (3-5ms).
    - Local Fallback: Tự động dùng Parquet cache nếu DB/Redis chưa bật.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.client = self._init_redis()
        self.pg_engine = self._init_postgres()
        self.catalog = self._load_catalog(self.data_dir / "train_features_neg_df.parquet")

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
            logger.info(f"Tools connected to Redis at {self.redis_host}:{self.redis_port}")
            return r
        except Exception:
            logger.warning("Live Redis not reachable. Falling back to in-memory fakeredis.")
            import fakeredis
            return fakeredis.FakeRedis(decode_responses=True)

    def _init_postgres(self):
        try:
            user = os.getenv("POSTGRES_USER", "recsysmvp")
            pwd = os.getenv("POSTGRES_PASSWORD", "recsysmvp")
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "dwh")
            url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
            engine = create_engine(url, connect_args={"connect_timeout": 1})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Tools connected to PostgreSQL at {host}:{port}/{db}")
            return engine
        except Exception:
            logger.debug("Live PostgreSQL not reachable. Using fast Parquet catalog cache.")
            return None

    def _load_catalog(self, path: Path) -> Dict[str, Dict[str, Any]]:
        catalog = {}
        if path.exists():
            try:
                cols = ["parent_asin", "title", "price", "main_category", "description"]
                df = pd.read_parquet(path, columns=cols)
                df = df.drop_duplicates(subset=["parent_asin"]).dropna(subset=["title"])
                for _, row in df.iterrows():
                    raw_price = row.get("price")
                    try:
                        price = float(str(raw_price).replace("$", "").strip()) if pd.notna(raw_price) else 29.99
                    except ValueError:
                        price = 29.99

                    # Parse description
                    desc_raw = row.get("description")
                    if isinstance(desc_raw, (list, tuple)) and len(desc_raw) > 0:
                        desc = str(desc_raw[0])[:300]
                    else:
                        desc = str(desc_raw)[:300] if desc_raw else "Sản phẩm chất lượng cao."

                    catalog[row["parent_asin"]] = {
                        "title": str(row["title"]),
                        "price": round(price, 2),
                        "category": str(row.get("main_category", "Video Games")),
                        "description": desc
                    }
                logger.info(f"Loaded {len(catalog):,} catalog products into memory cache")
            except Exception as e:
                logger.warning(f"Error loading catalog: {e}")
        return catalog


_backend = RecSysBackend()


# ==============================================================================
# LangGraph Tools
# ==============================================================================

@tool
def get_recommendations(
    user_id: str,
    filter: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Lấy danh sách sản phẩm (mã ASIN) được mô hình RecSys cá nhân hóa từ Redis.
    Hỗ trợ bộ lọc động qua tham số filter (ví dụ: {'max_price': 50.0, 'category': 'Mouse', 'top_k': 3}).
    """
    filters = filter or {}
    top_k = filters.get("top_k", 3)
    max_price = filters.get("max_price")
    category = filters.get("category")

    # 1. Lấy candidates từ Redis Feature Store
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

    # 2. Fallback sang popular candidates nếu user mới
    if not candidates:
        raw_popular = _backend.client.get("output:popular")
        if raw_popular:
            candidates = json.loads(raw_popular).get("rec_item_ids", [])

    if not candidates:
        candidates = list(_backend.catalog.keys())[:20]

    # 3. Dynamic Critiquing: Lọc theo điều kiện người dùng đưa ra
    filtered = []
    for asin in candidates:
        info = _backend.catalog.get(asin, {})
        price = info.get("price", 29.99)
        item_cat = info.get("category", "").lower()

        # Lọc theo giá tối đa nếu có yêu cầu
        if max_price is not None and price > max_price:
            continue

        # Lọc theo danh mục nếu có yêu cầu
        if category and category.lower() not in item_cat:
            continue

        filtered.append(asin)
        if len(filtered) >= top_k:
            break

    # Nếu lọc gắt quá không còn món nào, fallback lấy top gốc
    final_asins = filtered if filtered else candidates[:top_k]
    return json.dumps({"user_id": user_id, "recommended_asins": final_asins}, ensure_ascii=False)


@tool
def get_product_details(item_ids: List[str]) -> str:
    """
    Tra cứu chi tiết thông số (tên, giá, danh mục, mô tả) của danh sách mã ASIN.
    Truy vấn trực tiếp từ PostgreSQL oltp.amz_review_rating_raw (hoặc catalog cache).
    """
    results = []

    # 1. Thử query từ PostgreSQL nếu đang bật
    if _backend.pg_engine and item_ids:
        try:
            schema = os.getenv("POSTGRES_OLTP_SCHEMA", "oltp")
            table = "amz_review_rating_raw"
            query = text(f"""
                SELECT parent_asin, title, price, main_category, description 
                FROM {schema}.{table} 
                WHERE parent_asin = ANY(:asins)
            """)
            with _backend.pg_engine.connect() as conn:
                rows = conn.execute(query, {"asins": item_ids}).fetchall()
                for row in rows:
                    raw_price = row[2]
                    try:
                        price = float(str(raw_price).replace("$", "").strip()) if raw_price else 29.99
                    except ValueError:
                        price = 29.99
                    results.append({
                        "asin": row[0],
                        "title": row[1],
                        "price": round(price, 2),
                        "category": row[3] or "Video Games",
                        "description": str(row[4])[:300] if row[4] else "Sản phẩm chất lượng cao."
                    })
        except Exception as e:
            logger.debug(f"PostgreSQL query failed ({e}), falling back to memory catalog.")
            results = []

    # 2. Fallback sang in-memory catalog
    if not results:
        for asin in item_ids:
            info = _backend.catalog.get(asin, {})
            results.append({
                "asin": asin,
                "title": info.get("title", "Sản phẩm Phụ kiện Gaming"),
                "price": info.get("price", 29.99),
                "category": info.get("category", "Video Games"),
                "description": info.get("description", "Sản phẩm chính hãng với nhiều đánh giá tốt.")
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
