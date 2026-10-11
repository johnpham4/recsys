"""
Pure RecSys Load Testing & Performance Benchmark with Locust.
Tests only core RecSys serving endpoints (Zero Agent):
- Stage 1: Fast Redis Cache I2I Similar Products (/recs/i2i)
- Stage 1: Fast Redis Cache Popular / Trending Products (/recs/popular)
- Stage 2: User-to-Item Last Interacted Item Recs (/recs/u2i/last_item_i2i)
- Stage 2: Full Two-Stage Neural Sequential Rerank (/recs/u2i/rerank)
"""

import random
from locust import HttpUser, task, between, tag


SAMPLE_ITEMS = [
    "B001EYUXXU", "B09MFBJJQJ", "B0014CN2H6", "B0BM5YPLKM", "B003ZDOFF0",
    "B001EYUQDW", "B004HD55VK", "B07624RBWB", "B01GY35HKE", "B00CMQTVUA",
    "B00884S5N4", "B00DBLBMBQ", "B07PZ8NZSZ", "B01N3ASPNV", "B00503E8S2"
]


class PureRecSysUser(HttpUser):
    """
    Simulates real-world visitors hitting the core Recommendation Engine.
    Wait time: 0.1 - 0.5s for realistic heavy e-commerce load.
    """
    wait_time = between(0.1, 0.5)

    @tag("cache", "i2i")
    @task(5)
    def test_item_to_item_cache(self):
        """Stage 1: Redis Cache I2I Similar Products (Hot-path, ~50% traffic)."""
        item_id = random.choice(SAMPLE_ITEMS)
        with self.client.get(
            f"/recs/i2i?item_id={item_id}&count=10",
            name="/recs/i2i [Cache I2I]",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @tag("cache", "popular")
    @task(2)
    def test_popular_cache(self):
        """Stage 1: Redis Cache Trending / Popular Products (~20% traffic)."""
        with self.client.get(
            "/recs/popular?count=10",
            name="/recs/popular [Cache Popular]",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @tag("u2i", "last_item")
    @task(2)
    def test_u2i_last_item_i2i(self):
        """Stage 2: User Recent Item I2I (~20% traffic)."""
        user_id = f"user_{random.randint(1000, 1500)}"
        with self.client.get(
            f"/recs/u2i/last_item_i2i?user_id={user_id}&count=10",
            name="/recs/u2i/last_item_i2i [Recent Item]",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @tag("u2i", "rerank")
    @task(1)
    def test_twostage_rerank(self):
        """Stage 2: Full Two-Stage Neural Sequential Rerank (~10% heavy traffic)."""
        user_id = f"user_{random.randint(1000, 1500)}"
        with self.client.get(
            f"/recs/u2i/rerank?user_id={user_id}&top_k_retrieval=100&count=10",
            name="/recs/u2i/rerank [Two-Stage Neural]",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
