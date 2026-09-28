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
try:
    from src.api.load_examples import custom_openapi
    from src.middleware.logging_utils import RequestIDMiddleware
    from src.dto.models import FeatureRequest, FeatureRequestFeature, FeatureRequestResult
    from src.middleware.utils import debug_logging_decorator
except ImportError:
    from .load_examples import custom_openapi
    from ..middleware.logging_utils import RequestIDMiddleware
    from ..dto.models import FeatureRequest, FeatureRequestFeature, FeatureRequestResult
    from ..middleware.utils import debug_logging_decorator

app = FastAPI()
app.add_middleware(RequestIDMiddleware)

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | request_id: {extra[rec_id]} - {message}",
)

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:3000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
FEAST_ONLINE_SERVER_HOST = os.getenv("FEAST_ONLINE_SERVER_HOST", "localhost")
FEAST_ONLINE_SERVER_PORT = os.getenv("FEAST_ONLINE_SERVER_PORT", 6566)

seq_url = f"{MODEL_SERVER_URL}/predict"
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
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

# New endpoint to connect to external service
@app.post("/score/seq_rating_prediction")
@debug_logging_decorator
async def score_seq_rating_prediction(
    user_ids: List[str],
    item_sequences: List[List[str]],
    item_ids: List[str],
    debug: bool = Query(False, description="Enable debug logging"),
):
    logger.debug(
        f"Calling seq_rating_prediction with user_ids: {user_ids}, item_sequences: {item_sequences} and item_ids: {item_ids}"
    )

    # Prepare the payload for the model server
    payload = {
        "input_data": {
            "user_ids": user_ids,
            "item_sequences": item_sequences,
            "item_ids": item_ids,
        }
    }

    logger.debug(
        f"[COLLECT] Payload prepared: <features>{json.dumps(payload)}</features>"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                seq_url,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

        if response.status_code == 200:
            logger.debug(
                f"[COLLECT] Response from external service: <result>{json.dumps(response.json())}</result>"
            )
            result = response.json()
            return result
        else:
            error_message = (
                f"[DEBUG] External service returned an error: {response.text}"
            )
            logger.error(error_message)
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message,
            )

    except httpx.HTTPError as e:
        error_message = f"[DEBUG] Error connecting to external service: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)


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
    async with httpx.AsyncClient() as client:
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

    # Debug: Log response structure
    logger.debug(f"Feast response keys: {list(response.keys())}")
    logger.debug(f"Feast response: {response}")

    # Feast API always returns "results" key
    if "results" not in response:
        logger.error(f"No 'results' key found in response: {response}")
        raise HTTPException(status_code=500, detail="Invalid Feast API response format")

    logger.info(f"Metadata for result {response['metadata']} with user_id {user_id}")

    result = FeatureRequestResult(
        metadata=response["metadata"], results=response["results"]
    )
    item_sequence = result.get_feature_view(item_sequence_feature)
    item_sequece_ts = result.get_feature_view(item_sequence_ts_feature)

    return {
        "user_id": user_id,
        "item_sequence": item_sequence,
        "item_sequence_ts": item_sequece_ts
    }