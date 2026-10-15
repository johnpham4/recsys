import os
from functools import lru_cache
import redis
from loguru import logger


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """
    Trả về Redis client singleton có Connection Pool dùng chung cho toàn bộ app.
    Tự động fallback sang fakeredis nếu Redis server chưa bật (cho local test).
    """
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))

    try:
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=0,
            decode_responses=True,
            socket_timeout=1.5,
            max_connections=20
        )
        client = redis.Redis(connection_pool=pool)
        client.ping()
        logger.info(f"Connected to Redis Pool at {host}:{port}")
        return client
    except Exception as e:
        logger.warning(f"Live Redis unreachable ({e}). Falling back to in-memory FakeRedis.")
        try:
            import fakeredis
            return fakeredis.FakeRedis(decode_responses=True)
        except ImportError:
            logger.error("fakeredis is not installed. Returning standard disconnected client.")
            return redis.Redis(host=host, port=port, decode_responses=True)


def close_redis_client() -> None:
    """Đóng pool kết nối khi tắt server."""
    get_redis_client.cache_clear()
