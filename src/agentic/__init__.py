"""
Agentic Multi-Agent Recommendation Platform.
Hierarchical Supervisor Pattern with Coordinator and Specialist Subagents.
"""

from .contracts import (
    LeadAgentRequest,
    AgentChatResponse,
    AgentTelemetry,
    RecommendedProduct,
    ProductEvidence,
    RecommendedResult,
    ContextResult
)
from .coordinator import CoordinatorAgent
from .subagents import RecommendationSpecialist, ContextSpecialist

__all__ = [
    "CoordinatorAgent",
    "LeadAgentRequest",
    "AgentChatResponse",
    "AgentTelemetry",
    "RecommendedProduct",
    "ProductEvidence",
    "RecommendationSpecialist",
    "ContextSpecialist",
]
