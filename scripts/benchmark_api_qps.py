"""
Pure RecSys Load Testing & QPS Benchmark Suite (Zero Agent).
Measures real Multi-User HTTP Concurrency, QPS, and Latency Percentiles (P50, P90, P95, P99)
strictly across RecSys endpoints:
1. Pure Redis Cache RecSys: GET /recs/i2i & GET /recs/popular (Stage 1 Fast Path)
2. Full Two-Stage RecSys: GET /recs/u2i/rerank (Redis Retrieval + Real PyTorch Forward Pass)
3. Realistic RecSys Production Traffic Mix (70% Cache Hit + 30% Deep Neural Rerank)
"""

import sys
import time
import asyncio
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from src.api.main import app, redis_client

# Top Amazon Video Games items for testing
SAMPLE_ITEMS = [
    "B001EYUXXU", "B09MFBJJQJ", "B0014CN2H6", "B0BM5YPLKM", "B003ZDOFF0",
    "B001EYUQDW", "B004HD55VK", "B07624RBWB", "B01GY35HKE", "B00CMQTVUA",
    "B00884S5N4", "B00DBLBMBQ", "B07PZ8NZSZ", "B01N3ASPNV", "B00503E8S2",
    "B0086VPUHI", "B007CM0K86", "B00DB2BI8M", "B0088TN73M", "B00DTY9B0O"
]


async def simulate_cache_i2i_request(
    client: httpx.AsyncClient, user_idx: int, req_idx: int
) -> Dict[str, Any]:
    """Tests Stage-1 Redis Cache Item-to-Item endpoint (Similar Products)."""
    item_id = SAMPLE_ITEMS[req_idx % len(SAMPLE_ITEMS)]
    t0 = time.perf_counter()
    try:
        response = await client.get(f"/recs/i2i?item_id={item_id}&count=10", timeout=5.0)
        return {
            "status_code": response.status_code,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "success": response.status_code == 200,
        }
    except Exception as e:
        return {
            "status_code": 500,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "success": False,
            "error": str(e),
        }


async def simulate_cache_popular_request(
    client: httpx.AsyncClient, user_idx: int, req_idx: int
) -> Dict[str, Any]:
    """Tests Stage-1 Redis Cache Popular/Trending endpoint."""
    t0 = time.perf_counter()
    try:
        response = await client.get("/recs/popular?count=10", timeout=5.0)
        return {
            "status_code": response.status_code,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "success": response.status_code == 200,
        }
    except Exception as e:
        return {
            "status_code": 500,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "success": False,
            "error": str(e),
        }


async def simulate_twostage_rerank_request(
    client: httpx.AsyncClient, user_idx: int, req_idx: int
) -> Dict[str, Any]:
    """Tests Full Two-Stage: Redis Retrieval (100 items) + Real PyTorch Neural Rerank."""
    user_id = f"user_{1000 + (user_idx % 500)}"
    t0 = time.perf_counter()
    try:
        response = await client.get(
            f"/recs/u2i/rerank?user_id={user_id}&top_k_retrieval=100&count=10",
            timeout=10.0,
        )
        return {
            "status_code": response.status_code,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "success": response.status_code == 200,
        }
    except Exception as e:
        return {
            "status_code": 500,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "success": False,
            "error": str(e),
        }


async def simulate_recsys_traffic_mix(
    client: httpx.AsyncClient, user_idx: int, req_idx: int
) -> Dict[str, Any]:
    """
    Realistic E-Commerce RecSys Traffic Mix:
    70% Cache Retrieval (I2I/Popular) + 30% Full Two-Stage Neural Rerank.
    """
    rand = random.random()
    if rand < 0.50:
        return await simulate_cache_i2i_request(client, user_idx, req_idx)
    elif rand < 0.70:
        return await simulate_cache_popular_request(client, user_idx, req_idx)
    else:
        return await simulate_twostage_rerank_request(client, user_idx, req_idx)


async def run_benchmark_suite(
    name: str,
    simulate_fn,
    concurrency: int = 20,
    total_requests: int = 200,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"🚀 RECSYS BENCHMARK: [{name.upper()}]")
    print(f"   Concurrency: {concurrency} Virtual Users | Total Requests: {total_requests}")
    print(f"   Live Redis: {type(redis_client).__name__} (Ping: {redis_client.ping()})")
    print("=" * 80)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Warmup
        for _ in range(10):
            await simulate_fn(client, 0, 0)

        semaphore = asyncio.Semaphore(concurrency)
        results: List[Dict[str, Any]] = []

        async def worker(idx: int):
            async with semaphore:
                res = await simulate_fn(client, idx % concurrency, idx)
                results.append(res)

        wall_start = time.perf_counter()
        tasks = [worker(i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        wall_duration = time.perf_counter() - wall_start

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    latencies = np.array([r["latency_ms"] for r in results])

    qps = len(results) / wall_duration
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))

    print("-" * 50)
    print(f"📊 KẾT QUẢ ĐO RECSYS [{name}]:")
    print(f"🔹 Thành công:            {success_count:,}/{len(results):,} ({success_count/len(results)*100:.1f}%)")
    print(f"🔹 Thời gian đo:          {wall_duration:.2f} giây")
    print(f"🚀 THỰC ĐO API QPS:       {qps:8.1f} req/s  <--- [QPS]")
    print(f"⏱️ P50 Latency:          {p50:8.2f} ms")
    print(f"⏱️ P95 Latency (SLA):    {p95:8.2f} ms")
    print(f"⏱️ P99 Latency:          {p99:8.2f} ms")
    print("-" * 50)

    return {
        "name": name,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "qps": round(qps, 1),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean_lat, 2),
        "success_rate": round(success_count / len(results) * 100, 1),
    }


async def main():
    print("\n" + "#" * 80)
    print("🎯 BENCHMARK PURE RECSYS (KHÔNG CÓ AGENT) TRÊN TOÀN BỘ HỆ THỐNG")
    print("#" * 80)

    results = []

    # 1. API 1: Pure Redis Cache RecSys (GET /recs/i2i) - Fast Path (Stage 1)
    r1 = await run_benchmark_suite(
        name="Stage-1 Redis Cache RecSys (/recs/i2i)",
        simulate_fn=simulate_cache_i2i_request,
        concurrency=20,
        total_requests=500,
    )
    results.append(r1)

    # 2. API 2: Full Two-Stage RecSys (Redis Retrieval + Real PyTorch Neural Rerank) - 20 Users
    r2 = await run_benchmark_suite(
        name="Full Two-Stage RecSys (Redis + Real PyTorch) - 20 Users",
        simulate_fn=simulate_twostage_rerank_request,
        concurrency=20,
        total_requests=200,
    )
    results.append(r2)

    # 3. API 2: Full Two-Stage RecSys - 50 Users (High Load)
    r3 = await run_benchmark_suite(
        name="Full Two-Stage RecSys (Redis + Real PyTorch) - 50 Users",
        simulate_fn=simulate_twostage_rerank_request,
        concurrency=50,
        total_requests=500,
    )
    results.append(r3)

    # 4. Realistic RecSys Traffic Mix (70% Cache + 30% Full Two-Stage PyTorch)
    r4 = await run_benchmark_suite(
        name="Production RecSys Traffic Mix (70% Cache + 30% Two-Stage)",
        simulate_fn=simulate_recsys_traffic_mix,
        concurrency=30,
        total_requests=500,
    )
    results.append(r4)

    out_file = PROJECT_ROOT / "benchmark_results_recsys_only.json"
    with open(out_file, "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Kết quả đã lưu vào: {out_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
