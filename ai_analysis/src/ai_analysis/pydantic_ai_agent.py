"""
PydanticAI Structured Type-Safe Agent
Inspired by pydantic/pydantic-ai
Provides schema-validated, type-safe agent execution with structured response models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProductAnalysisResult(BaseModel):
    product_category: str = Field(..., description="Identified product category")
    total_records: int = Field(..., description="Number of rows analyzed")
    average_price: float = Field(..., description="Average calculated price")
    high_rating_ratio: float = Field(..., description="Percentage of products rated 4.0+")
    key_findings: List[str] = Field(default_factory=list, description="Top strategic insights")
    outlier_count: int = Field(default=0, description="Detected outlier price points")


class PydanticAIEngine:
    """
    Type-safe agent execution pipeline ensuring all outputs conform strictly to Pydantic schemas.
    """

    @staticmethod
    def validate_and_format_analysis(raw_data: Dict[str, Any]) -> ProductAnalysisResult:
        return ProductAnalysisResult(**raw_data)


if __name__ == "__main__":
    sample = {
        "product_category": "Footwear & Apparel",
        "total_records": 1250,
        "average_price": 49.99,
        "high_rating_ratio": 84.5,
        "key_findings": ["Strong positive correlation between rating and repeat purchases", "Mid-tier pricing drives 70% volume"],
        "outlier_count": 12,
    }
    validated = PydanticAIEngine.validate_and_format_analysis(sample)
    print("✅ PydanticAI Validated Schema:")
    print(validated.model_dump_json(indent=2))
