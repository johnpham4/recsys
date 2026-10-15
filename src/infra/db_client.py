
import os
from functools import lru_cache
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine."""

    required_vars = ("POSTGRES_USER", "POSTGRES_PASSWORD")

    missing = [key for key in required_vars if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            f"Missing required database configuration: {', '.join(missing)}"
        )

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "dwh"),
    )

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_timeout=30,
        pool_recycle=1800,
    )


@lru_cache(maxsize=1)
def get_session_factory():
    return sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
    )


def dispose_engine() -> None:
    """Release pooled connections during application shutdown."""
    get_engine.cache_clear()
    get_session_factory.cache_clear()
