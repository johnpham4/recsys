from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SubagentExecution(BaseModel):
    subagent_name: str
    action_taken: str
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class AgentTelemetry(BaseModel):
    e2e_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    subagent_traces: List[SubagentExecution] = Field(default_factory=list)


# Subagent 1: recommendation result
class RecommendedProduct(BaseModel):
    item_id: str
    score: float = 0.0
    rank: int

class RecommendedResult(BaseModel):
    user_id: str
    items: List[RecommendedProduct] = Field(default_factory=list)
    model_name: str = ""


# Subagent 2: context and evidence specialist contracts
class ProductEvidence(BaseModel):
    item_id: str
    title: str
    description: str = ""
    price: float = 0.0
    category: str = ""
    brand: str = ""


class UserProfileContext(BaseModel):
    user_id: str
    recent_item_sequence: List[str] = Field(default_factory=list)
    top_category: Optional[str] = None


class ContextResult(BaseModel):
    user_id: str
    user_profile: UserProfileContext
    evidences: List[ProductEvidence] = Field(default_factory=list)


# 4. Lead Agent (Coordinator) Request & Response Contracts
class LeadAgentRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None



class AgentChatResponse(BaseModel):
    user_id: str
    reply: str
    recommendations: List[RecommendedProduct] = Field(default_factory=list)
    evidences: List[ProductEvidence] = Field(default_factory=list)
    telemetry: AgentTelemetry


# Backward compatibility alias
AgentChatRequest = LeadAgentRequest

