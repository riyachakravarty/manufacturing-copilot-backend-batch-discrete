from pydantic import BaseModel
from typing import Dict, List

class SHAPSummaryContext(BaseModel):
    feature_importance: Dict[str, float]
    top_features: List[str]

class SHAPDependenceContext(BaseModel):
    feature: str
    shap_values: List[float]
    feature_values: List[float]
    correlation: float
