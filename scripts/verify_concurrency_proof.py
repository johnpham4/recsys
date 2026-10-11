import httpx
import asyncio
import time
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app

transport = httpx.ASGITransport(app=app)

async def test_single_user(client, user_id, idx):
    t_start = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    t0 = time.perf_counter()
    res = await client.get(f"/recs/u2i/rerank?user_id={user_id}&top_k_retrieval=100&count=3")
    dur_ms = (time.perf_counter() - t0) * 1000
    data = res.json()
    t_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    rec_id = data.get("metadata", {}).get("rec_id", "N/A")[:8]
    recs = data.get("recommendations", {}).get("rec_item_ids", [])
    print(f"[User #{idx:02d} | {user_id}] Bắn: {t_start} -> Trả: {t_end} (Mất: {dur_ms:5.1f}ms) | ReqID: {rec_id} | Top Recs: {recs}")

async def main():
    print("=" * 80)
    print("⚡ BẰNG CHỨNG THỰC TẾ: 5 USERS ẬP VÀO CÙNG 1 THỜI ĐIỂM")
    print("=" * 80)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        users = ["user_1001", "user_1002", "user_1003", "user_1004", "user_1005"]
        # asyncio.gather fires ALL 5 requests in parallel concurrently
        await asyncio.gather(*[test_single_user(client, u, i+1) for i, u in enumerate(users)])
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
