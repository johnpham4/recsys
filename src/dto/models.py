from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel

class Entities(str, Enum):
    user = "user"
    parent_asin = "parent_asin"

class FeatureRequest(BaseModel):
    entities: Dict[str, List[str]]
    features: List[str]

class FeatureRequestFeature(BaseModel):
    feature_view: str
    feature_name: str

    def get_full_name(self, fresh: bool, is_request: bool):
        feature_view = self.feature_view
        delimiter = ":"
        if fresh:
            feature_view += "_fresh"
        if not is_request:
            delimiter = "__"
        return f"{feature_view}{delimiter}{self.feature_name}"

class FeatureRequestResult(BaseModel):
    class Metadata(BaseModel):
        feature_names: List[str]

    class Result(BaseModel):
        values: List[Optional[str]]
        statuses: List[str]
        event_timestamps: List[datetime]

    metadata: Metadata
    results: List[Result]

    def get_feature_view(self, feature: FeatureRequestFeature):
        # Try to find fresh feature first
        fresh_idx = None
        try:
            fresh_idx = self.metadata.feature_names.index(
                feature.get_full_name(fresh=True, is_request=False)
            )
        except ValueError:
            pass

        # Try to find common feature
        common_idx = None
        try:
            common_idx = self.metadata.feature_names.index(
                feature.get_full_name(fresh=False, is_request=False)  # Use response format
            )
        except ValueError:
            pass

        # Get the feature value
        feature_results = self.results
        get_feature_values = lambda idx: feature_results[idx].values[0] if idx is not None else None

        fresh_value = get_feature_values(fresh_idx)
        if fresh_value is not None:
            return fresh_value.split(",")

        common_value = get_feature_values(common_idx)
        if common_value is not None:
            return common_value.split(",")

        return []

    def get_feature_value_no_fresh(self, feature: FeatureRequestFeature):
        """Get normal feature not including fresh source"""
        common_idx = self.metadata.feature_names.index(
            feature.get_full_name(fresh=False, is_request=False)
        )

        feature_results = self.results
        get_feature_values = lambda idx: feature_results[idx].values[0]
        feature_value = get_feature_values(common_idx)
        return feature_value



