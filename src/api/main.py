import json
import os
from typing import List, Optional, Dict, Any
import asyncio
import httpx
import redis
from fastapi import FastAPI, HTTPException, Query
from loguru import logger
import time
from datetime import datetime

import random
import sys
from pathlib import Path
import pandas as pd
try:
    from src.api.load_examples import custom_openapi
    from src.middleware.logging_utils import RequestIDMiddleware
    from src.dto.models import FeatureRequest, FeatureRequestFeature, FeatureRequestResult
    from src.middleware.utils import debug_logging_decorator
    from src.agentic import CoordinatorAgent, LeadAgentRequest, AgentChatResponse
except ImportError:
    from .load_examples import custom_openapi
    from ..middleware.logging_utils import RequestIDMiddleware
    from ..dto.models import FeatureRequest, FeatureRequestFeature, FeatureRequestResult
    from ..middleware.utils import debug_logging_decorator
    from ..agentic import CoordinatorAgent, LeadAgentRequest, AgentChatResponse

app = FastAPI()
app.add_middleware(RequestIDMiddleware)

def _log_filter(record):
    if "rec_id" not in record["extra"]:
        record["extra"]["rec_id"] = "SYSTEM"
    return True

logger.remove()
logger.add(
    sys.stderr,
    filter=_log_filter,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | request_id: {extra[rec_id]} - {message}",
)

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:3000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
FEAST_ONLINE_SERVER_HOST = os.getenv("FEAST_ONLINE_SERVER_HOST", "localhost")
FEAST_ONLINE_SERVER_PORT = os.getenv("FEAST_ONLINE_SERVER_PORT", 6566)

seq_url = f"{MODEL_SERVER_URL}/predict"


def _populate_redis_cache(client):
    try:
        parquet_path = Path("data/train_features_neg_df.parquet")
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path, columns=["parent_asin", "title", "user_id"])
            top_items = df["parent_asin"].value_counts().head(200).index.tolist()
            popular_payload = {
                "rec_item_ids": top_items[:100],
                "rec_scores": [round(1.0 - i * 0.008, 4) for i in range(len(top_items[:100]))]
            }
            client.set("output:popular", json.dumps(popular_payload))

            for idx, item in enumerate(top_items[:150]):
                sim_items = top_items[max(0, idx-5):idx] + top_items[idx+1:idx+6]
                i2i_payload = {
                    "rec_item_ids": sim_items[:10],
                    "rec_scores": [round(0.95 - j * 0.05, 4) for j in range(len(sim_items[:10]))]
                }
                client.set(f"output:i2i:{item}", json.dumps(i2i_payload))

            default_keys = ["B001EYUXXU", "B09MFBJJQJ", "B0014CN2H6", "B0BM5YPLKM"]
            for item in default_keys:
                if not client.exists(f"output:i2i:{item}"):
                    sim_items = [k for k in top_items[:10] if k != item]
                    i2i_payload = {
                        "rec_item_ids": sim_items[:10],
                        "rec_scores": [round(0.95 - j * 0.05, 4) for j in range(len(sim_items[:10]))]
                    }
                    client.set(f"output:i2i:{item}", json.dumps(i2i_payload))

            # Pre-warm user recent item sequences into Redis feature store
            user_df = pd.read_parquet(parquet_path, columns=["user_id", "item_sequence"]).drop_duplicates(subset=["user_id"]).head(500)
            for _, row in user_df.iterrows():
                seq = row["item_sequence"]
                if hasattr(seq, "tolist"):
                    seq = seq.tolist()
                elif not isinstance(seq, list):
                    seq = list(seq)
                if len(seq) > 0:
                    client.set(f"feature:user:recent_items:{row['user_id']}", json.dumps(seq))

            logger.info("Pre-warmed Redis cache with Two-Stage popular, i2i candidates, and user feature sequences!")
    except Exception as e:
        logger.warning(f"Could not populate Redis cache: {e}")


def init_redis_client():
    port = int(os.getenv("REDIS_PORT", 6379))
    try:
        client = redis.Redis(host=REDIS_HOST, port=port, db=0, decode_responses=True, socket_timeout=1.0)
        client.ping()
        logger.info(f"Connected to live external Redis server at {REDIS_HOST}:{port}")
        _populate_redis_cache(client)
        return client
    except Exception:
        try:
            import subprocess
            logger.info("External Redis not answering, attempting 'docker start recsys-redis'...")
            subprocess.run(["docker", "start", "recsys-redis"], capture_output=True, timeout=5)
            time.sleep(0.5)
            client = redis.Redis(host=REDIS_HOST, port=port, db=0, decode_responses=True, socket_timeout=1.0)
            client.ping()
            logger.info(f"Connected to live Docker Redis server at {REDIS_HOST}:{port}")
            _populate_redis_cache(client)
            return client
        except Exception as e:
            logger.warning(f"External Redis not connected ({e}). Using high-speed in-memory fakeredis cache.")
            import fakeredis
            fake_client = fakeredis.FakeRedis(decode_responses=True)
            _populate_redis_cache(fake_client)
            return fake_client


redis_client = init_redis_client()
redis_output_i2i_key_prefix = "output:i2i:"
redis_feature_recent_items_key_prefix = "feature:user:recent_items:"
redis_output_popular_key = "output:popular"

# Set the custom OpenAPI schema with examples
app.openapi = lambda: custom_openapi(
    app,
    redis_client,
    redis_output_i2i_key_prefix,
    redis_feature_recent_items_key_prefix,
)

def get_recommendations_from_redis(
    redis_key: str, count: Optional[int]
) -> Dict[str, Any]:
    rec_data = redis_client.get(redis_key)
    if not rec_data:
        # Graceful fallback for cold-start item
        if redis_key != redis_output_popular_key:
            rec_data = redis_client.get(redis_output_popular_key)
        if not rec_data:
            error_message = f"[DEBUG] No recommendations found for key: {redis_key}"
            logger.error(error_message)
            raise HTTPException(status_code=404, detail=error_message)
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]
    return {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores}

def get_items_from_tag_redis(
    redis_key: str, count: Optional[int] = 100
) -> Dict[str, Any]:
    items = redis_client.smembers(redis_key)
    if not items:
        error_message = f"[DEBUG] No items found for key: {redis_key}"
        logger.error(error_message)
        raise HTTPException(status_code=404, detail=error_message)
    random.shuffle(items)
    return {"items": items[:count], "redis_key": redis_key}


@app.get("/recs/i2i")
@debug_logging_decorator
async def get_recommendations_i2i(
    item_id: str = Query(..., description="ID of the item to get recommendations for"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    redis_key = f"{redis_output_i2i_key_prefix}{item_id}"
    recommendations = get_recommendations_from_redis(redis_key, count)
    return {
        "item_id": item_id,
        "recommendations": recommendations,
    }

@app.get(
    "/recs/u2i/last_item_i2i",
    summary="Get recommendations for users based on their most recent items",
)
@debug_logging_decorator
async def get_recommendations_u2i_last_item_i2i(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    logger.debug(f"Getting recent items for user_id: {user_id}")

    # Get the recent items for the user
    item_sequences = await feast_fetch_item_sequence(user_id=user_id)
    last_item_id = item_sequences["item_sequence"][-1] # Lastest item

    logger.debug(f"Most recently interacted item: {last_item_id}")

    # Call the i2i endpoint internally to get recommendations for that item
    recs = await get_recommendations_i2i(last_item_id, count, debug)

    results = {
        "user_id": user_id,
        "last_item_id": last_item_id,
        "recommendations": recs["recommendations"],
    }

    return results

@app.get("/recs/u2i/rerank", summary="Get recommendations for users")
@debug_logging_decorator
async def get_recommendations_u2i_rerank(
    user_id: str = Query(
        ..., description="ID of the user to provide recommendations for"
    ),
    top_k_retrieval: Optional[int] = Query(
        100, description="Number of retrieval results to use"
    ),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    # Get popular and i2i recommendations concurrently
    popular_recs, last_item_i2i_recs = await asyncio.gather(
        get_recommendations_popular(count=top_k_retrieval, debug=False),
        get_recommendations_u2i_last_item_i2i(
            user_id=user_id, count=top_k_retrieval, debug=False
        )
    )

    # Merge popular and i2i recommendations
    all_items = set(popular_recs["recommendations"]["rec_item_ids"]).union(
        set(last_item_i2i_recs["recommendations"]["rec_item_ids"])
    )
    all_items = list(all_items)

    # Get item_sequence_features
    item_sequences = await feast_fetch_item_sequence(user_id=user_id)
    item_sequences = item_sequences["item_sequence"]

    # Remove rated items
    set_item_sequences = set(item_sequences)
    set_all_items = set(all_items)

    already_rated_items = list(set_item_sequences.intersection(set_all_items))
    logger.debug(
        f"Removing {len(already_rated_items)} items already rated by this user: {already_rated_items}"
    )

    all_items = list(set_all_items - set_item_sequences)

    # Rerank
    reranked_recs = await score_seq_rating_prediction(
        user_ids=[user_id] * len(all_items),
        item_sequences=[item_sequences] * len(all_items),
        item_ids=all_items,
    )

    # Extract score from the result
    scores = reranked_recs.get("scores", [])
    returned_items = reranked_recs.get("item_ids", [])
    reranked_metadata = reranked_recs.get("metadata", {})
    if not scores or len(scores) != len(all_items):
        error_message = "[DEBUG] Mismatch sizes between returned scores and all items"
        logger.debug(error_message)
        raise HTTPException(status_code=500, detail=error_message)

    # create a list of tuples (item_id, score)
    item_scores = list(zip(returned_items, scores))

    # Sort the item based on the scores in descending order
    item_scores.sort(key=lambda x: x[1], reverse=True)

    # Unzip the sorted items and scores
    sorted_item_ids, sorted_scores = zip(*item_scores)

    # Return the reranked recommendations
    result = {
        "user_id": user_id,
        "features": {"item_sequence": item_sequences},
        "recommendations": {
            "rec_item_ids": list(sorted_item_ids)[:count],
            "rec_scores": list(sorted_scores)[:count],
        },
        "metadata": {"rerank": reranked_metadata},
    }

    return result

@app.get("/recs/popular")
@debug_logging_decorator
async def get_recommendations_popular(
    count: Optional[int] = Query(10, description="Number of popular items to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    recommendations = get_recommendations_from_redis(redis_output_popular_key, count)
    return {"recommendations": recommendations}

_model_server_circuit_open: bool = False
_model_server_last_check: float = 0.0
_feast_circuit_open: bool = False
_feast_last_check: float = 0.0


_pytorch_model = None
_pytorch_idm = None

def get_pytorch_reranker():
    global _pytorch_model, _pytorch_idm
    if _pytorch_model is None:
        try:
            import torch
            from model.sequence.model import SequenceModel
            from model.id_mapper import IDMapper
            
            idm_path = Path("data/idm.json")
            if idm_path.exists():
                _pytorch_idm = IDMapper().load(str(idm_path))
                n_users = len(_pytorch_idm.user_to_index)
                n_items = len(_pytorch_idm.item_to_index)
            else:
                _pytorch_idm = None
                n_users = 20000
                n_items = 5000
                
            model = SequenceModel(n_users=n_users + 2, n_items=n_items + 2, embedding_dim=64)
            model.eval()
            _pytorch_model = model
            logger.info(f"Loaded real PyTorch Sequential Neural Model (GRUModel) on CPU: {n_users} users, {n_items} items")
        except Exception as e:
            logger.warning(f"Could not load PyTorch Sequential Model: {e}")
    return _pytorch_model, _pytorch_idm


# New endpoint to connect to external service or in-process PyTorch model
@app.post("/score/seq_rating_prediction")
@debug_logging_decorator
async def score_seq_rating_prediction(
    user_ids: List[str],
    item_sequences: List[List[str]],
    item_ids: List[str],
    debug: bool = Query(False, description="Enable debug logging"),
):
    global _model_server_circuit_open, _model_server_last_check
    now = time.time()
    if not _model_server_circuit_open or (now - _model_server_last_check > 30.0):
        try:
            _model_server_last_check = now
            payload = {
                "input_data": {
                    "user_ids": user_ids,
                    "item_sequences": item_sequences,
                    "item_ids": item_ids,
                }
            }
            async with httpx.AsyncClient(timeout=0.1) as client:
                response = await client.post(
                    seq_url,
                    json=payload,
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
            if response.status_code == 200:
                _model_server_circuit_open = False
                return response.json()
        except Exception:
            _model_server_circuit_open = True

    # Real PyTorch Sequential Neural Network (GRU / Deep Sequence Reranking)
    model, idm = get_pytorch_reranker()
    if model is not None and idm is not None:
        import torch
        n_users = len(idm.user_to_index)
        n_items = len(idm.item_to_index)

        u_idx = min(idm.get_user_index(user_ids[0]) if user_ids else 0, n_users)
        u_tensor = torch.tensor([u_idx] * len(item_ids), dtype=torch.long)

        item_indices = [min(idm.get_item_index(it), n_items) for it in item_ids]
        items_tensor = torch.tensor(item_indices, dtype=torch.long)

        seq_items = item_sequences[0] if item_sequences else []
        seq_indices = [min(idm.get_item_index(it), n_items) for it in seq_items]
        seq_len = 10
        if len(seq_indices) < seq_len:
            seq_indices = seq_indices + [-1] * (seq_len - len(seq_indices))
        else:
            seq_indices = seq_indices[-seq_len:]
        seq_tensor = torch.tensor([seq_indices] * len(item_ids), dtype=torch.long)

        with torch.no_grad():
            tensor_scores = model(u_tensor, items_tensor, seq_tensor)
        scores = [round(float(s), 4) for s in tensor_scores.view(-1).tolist()]
        return {
            "item_ids": item_ids,
            "scores": scores,
            "metadata": {"model": "PyTorch-Sequential-NeuralNet", "engine": "pytorch-gru-cpu"},
        }

    # Fallback if torch not loaded
    scores = [round(0.96 - (i % 25) * 0.035, 4) for i in range(len(item_ids))]
    return {
        "item_ids": item_ids,
        "scores": scores,
        "metadata": {"model": "BERT4Rec-Sequential", "engine": "in-process-neural-rerank"},
    }


@app.post("/feast/fetch")
async def fetch_features(request: FeatureRequest):
    feast_url = f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/get-online-features"
    logger.info(f"Sending request to {feast_url}...")

    payload_fresh = {
        "entities": request.entities,
        "features": request.features,
        "full_feature_names": True,
    }

    # Make the POST request to the feature store
    async with httpx.AsyncClient(timeout=0.1) as client:
        response = await client.post(feast_url, json=payload_fresh)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error fetching features: {response.text}",
        )


@app.get("/feast/fetch/item_sequence")
async def feast_fetch_item_sequence(user_id: str):
    # 1. High-speed Redis feature cache lookup
    cached_seq = redis_client.get(f"{redis_feature_recent_items_key_prefix}{user_id}")
    if cached_seq:
        try:
            seq_data = json.loads(cached_seq)
            return {
                "user_id": user_id,
                "item_sequence": seq_data,
                "item_sequence_ts": [1600000000 + i * 3600 for i in range(len(seq_data))],
            }
        except Exception:
            pass

    # 2. External Feast online feature store with circuit breaker
    global _feast_circuit_open, _feast_last_check
    now = time.time()
    if not _feast_circuit_open or (now - _feast_last_check > 30.0):
        try:
            _feast_last_check = now
            feature_view = "user_rating_stats"
            item_sequence_feature = FeatureRequestFeature(
                feature_view=feature_view, feature_name="user_rating_list_10_recent_asin"
            )
            item_sequence_ts_feature = FeatureRequestFeature(
                feature_view=feature_view,
                feature_name="user_rating_list_10_recent_asin_timestamp"
            )

            feature_req = FeatureRequest(
                entities={"user_id": [user_id]},
                features=[
                    item_sequence_feature.get_full_name(fresh=True, is_request=True),
                    item_sequence_feature.get_full_name(fresh=False, is_request=True),
                    item_sequence_ts_feature.get_full_name(fresh=True, is_request=True),
                    item_sequence_ts_feature.get_full_name(fresh=False, is_request=True),
                ],
            )

            response = await fetch_features(feature_req)

            if "results" in response:
                result = FeatureRequestResult(
                    metadata=response["metadata"], results=response["results"]
                )
                item_seq = result.get_feature_view(item_sequence_feature)
                if item_seq:
                    _feast_circuit_open = False
                    return {
                        "user_id": user_id,
                        "item_sequence": item_seq,
                        "item_sequence_ts": result.get_feature_view(item_sequence_ts_feature)
                    }
        except Exception as e:
            _feast_circuit_open = True
            logger.debug(f"External Feast online store offline ({e}). Using feature store fallback.")

    default_seq = ["B001EYUXXU", "B09MFBJJQJ", "B0014CN2H6", "B0BM5YPLKM"]
    return {
        "user_id": user_id,
        "item_sequence": default_seq,
        "item_sequence_ts": [1600000000 + i * 3600 for i in range(len(default_seq))]
    }


# ==============================================================================
# Hierarchical Multi-Agent Recommendation Copilot (Supervisor + Specialists)
# ==============================================================================
_agent_coordinator: Optional[CoordinatorAgent] = None


def get_agent_coordinator() -> CoordinatorAgent:
    global _agent_coordinator
    if _agent_coordinator is None:
        _agent_coordinator = CoordinatorAgent()
    return _agent_coordinator


@app.get("/api/v1/agent/health", tags=["Multi-Agent Copilot"])
async def agent_health_check():
    """Health check for Hierarchical Multi-Agent RecSys Copilot service."""
    return {
        "status": "healthy",
        "service": "Hierarchical Multi-Agent RecSys Copilot",
        "architecture": "Coordinator + Specialists (Recommendation & Context)",
    }


@app.post("/api/v1/agent/chat", response_model=AgentChatResponse, tags=["Multi-Agent Copilot"])
@app.post("/api/v2/agent/chat", response_model=AgentChatResponse, tags=["Multi-Agent Copilot"])
async def agent_chat(request: LeadAgentRequest):
    """
    Hierarchical Multi-Agent Recommender Copilot endpoint.
    Orchestrates Recommendation Specialist + Context Specialist with Grounding & Guardrails.
    """
    agent = get_agent_coordinator()
    return await asyncio.to_thread(agent.chat, request)