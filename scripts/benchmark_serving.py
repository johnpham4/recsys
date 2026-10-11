"""
Production Serving & Latency Benchmark for BERT4Rec & Recommendation Pipeline.
Measures P50, P90, P95, P99 Latency and Throughput (QPS) for CV / Technical Portfolio.
Supports both Local and Modal Cloud GPU environments.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.bert4rec.model import BERT4Rec
from model.bert4rec.inference import BERT4RecInference


def run_latency_benchmark(
    num_items: int = 5000,
    seq_len: int = 50,
    embed_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    n_iterations: int = 1000,
    warmup: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 70)
    print(f"⚡ RUNNING LATENCY & SLA BENCHMARK ({device.upper()})")
    print(f"   Model: BERT4Rec ({num_layers} layers, {num_heads} heads, embed_dim={embed_dim})")
    print(f"   Vocabulary: {num_items:,} items | Seq Length: {seq_len} | Iterations: {n_iterations:,}")
    print("=" * 70)

    # Initialize model
    model = BERT4Rec(
        num_users=10000,
        num_items=num_items,
        max_seq_len=seq_len,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=embed_dim * 4,
        padding_idx=num_items,
    )
    
    # Check if a trained checkpoint exists to load real weights
    ckpt_paths = list(Path("notebooks/data/003-bert4rec/checkpoints").glob("*.ckpt")) + \
                 list(Path("/checkpoints").glob("*.ckpt")) + \
                 list(Path("checkpoints").glob("*.ckpt"))
    
    if ckpt_paths:
        latest_ckpt = ckpt_paths[0]
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            # Filter lit_model prefix if present
            cleaned_state = {k.replace("model.", ""): v for k, v in state_dict.items() if not k.startswith("loss")}
            model.load_state_dict(cleaned_state, strict=False)
            print(f" Loaded weights from checkpoint: {latest_ckpt.name}")
        except Exception as e:
            print(f"ℹ️ Benchmarking with native initialized architecture: {e}")

    inferencer = BERT4RecInference(model=model, device=device, max_seq_len=seq_len, padding_idx=num_items)

    # Synthetic realistic user interaction sequence
    sample_seq = np.random.randint(1, num_items, size=min(15, seq_len)).tolist()

    # 1. Warm-up
    print(f"\n🔥 Warming up device ({warmup} iterations)...")
    for _ in range(warmup):
        _ = inferencer.predict_next_item(sequence=sample_seq, user_id=1, top_k=10)

    if device == "cuda":
        torch.cuda.synchronize()

    # 2. Benchmark Single-User Real-time Request (Batch Size = 1)
    print(f"🚀 Measuring Single-User Online Serving Latency ({n_iterations} requests)...")
    latencies_ms = []

    for _ in range(n_iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        _ = inferencer.predict_next_item(sequence=sample_seq, user_id=1, top_k=10)
        
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    p50 = np.percentile(latencies_ms, 50)
    p90 = np.percentile(latencies_ms, 90)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)
    mean_lat = np.mean(latencies_ms)
    throughput_qps = 1000.0 / mean_lat

    # 3. Benchmark Batch Serving (Batch Size = 32)
    print(f"📦 Measuring Batch Serving Latency (Batch Size = 32, 100 batches)...")
    batch_latencies_ms = []
    batch_user_ids = torch.randint(1, 10000, (32,), device=device)
    batch_input_ids = torch.randint(1, num_items, (32, seq_len), device=device)
    batch_attn = torch.ones((32, seq_len), device=device)

    for _ in range(100):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            _ = model.recommend(batch_user_ids, batch_input_ids, k=10, attention_mask=batch_attn)
            
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_latencies_ms.append((t1 - t0) * 1000.0)

    batch_p95 = np.percentile(batch_latencies_ms, 95)
    batch_throughput = (32 * 1000.0) / np.mean(batch_latencies_ms)

    # 4. Display Results
    print("\n" + "=" * 70)
    print("📊 SERVING PERFORMANCE & SLA REPORT")
    print("=" * 70)
    print(f"  Device:             {device.upper()}")
    print(f"  Mean Latency:       {mean_lat:.2f} ms")
    print(f"  P50 (Median):       {p50:.2f} ms")
    print(f"  P90:                {p90:.2f} ms")
    print(f"  P95 (SLA Target):   {p95:.2f} ms  <--- (Dùng cho CV)")
    print(f"  P99 (Tail Latency): {p99:.2f} ms")
    print(f"  Real-time QPS:      {throughput_qps:,.0f} req/s")
    print(f"  Batch (BS=32) P95:  {batch_p95:.2f} ms ({batch_throughput:,.0f} items/s)")
    print("=" * 70)

    return {
        "device": device,
        "mean_ms": round(mean_lat, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "qps": int(throughput_qps),
        "batch_p95_ms": round(batch_p95, 2),
        "batch_throughput": int(batch_throughput)
    }


if __name__ == "__main__":
    run_latency_benchmark()
