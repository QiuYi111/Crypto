"""Data models for LLM confidence vector system."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """Represents a news article with metadata."""
    
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content/summary")
    source: str = Field(..., description="News source")
    published_date: datetime = Field(..., description="Publication date")
    url: Optional[str] = Field(None, description="Original URL")
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Raw sentiment score")
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="Relevance to crypto trading")
    tags: List[str] = Field(default_factory=list, description="Article tags/categories")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConfidenceVector(BaseModel):
    """Represents a multi-dimensional confidence vector for trading decisions."""
    
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTCUSDT')")
    date: datetime = Field(..., description="Date of assessment")
    
    # Core confidence dimensions
    fundamentals: float = Field(..., ge=0, le=1, description="Fundamental analysis confidence")
    industry_condition: float = Field(..., ge=0, le=1, description="Industry/sector health confidence")
    geopolitics: float = Field(..., ge=0, le=1, description="Geopolitical risk impact confidence")
    macroeconomics: float = Field(..., ge=0, le=1, description="Macroeconomic environment confidence")
    
    # Additional dimensions
    technical_sentiment: float = Field(..., ge=0, le=1, description="Technical analysis sentiment")
    regulatory_impact: float = Field(..., ge=0, le=1, description="Regulatory environment impact")
    innovation_impact: float = Field(..., ge=0, le=1, description="Technological innovation impact")
    
    # Metadata
    news_sources: List[str] = Field(default_factory=list, description="Sources used for assessment")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence in this vector")
    reasoning: str = Field(..., description="Detailed reasoning for the scores")
    
    def to_array(self) -> List[float]:
        """Convert to numpy-compatible array format."""
        return [
            self.fundamentals,
            self.industry_condition, 
            self.geopolitics,
            self.macroeconomics,
            self.technical_sentiment,
            self.regulatory_impact,
            self.innovation_impact
        ]
    
    @classmethod
    def from_array(cls, symbol: str, date: datetime, vector: List[float], **kwargs) -> "ConfidenceVector":
        """Create from array format."""
        if len(vector) != 7:
            raise ValueError("Vector must have exactly 7 dimensions")
            
        return cls(
            symbol=symbol,
            date=date,
            fundamentals=vector[0],
            industry_condition=vector[1],
            geopolitics=vector[2],
            macroeconomics=vector[3],
            technical_sentiment=vector[4],
            regulatory_impact=vector[5],
            innovation_impact=vector[6],
            **kwargs
        )


class LLMResponse(BaseModel):
    """Represents a response from the LLM."""
    
    confidence_vector: List[float] = Field(..., description="Raw confidence vector from LLM")
    reasoning: str = Field(..., description="LLM's reasoning text")
    model_name: str = Field(..., description="Name of the LLM model used")
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    processing_time: float = Field(..., description="Processing time in seconds")


class SearchQuery(BaseModel):
    """Represents a search query for news retrieval."""
    
    symbol: str = Field(..., description="Trading symbol")
    date: datetime = Field(..., description="Target date for news")
    keywords: List[str] = Field(default_factory=list, description="Additional keywords")
    max_results: int = Field(default=10, description="Maximum number of articles")
    sources: List[str] = Field(default_factory=list, description="Preferred news sources")
    
    def to_search_string(self) -> str:
        """Convert to search string for external APIs."""
        base = f"{self.symbol.replace('USDT', '')} cryptocurrency"
        if self.keywords:
            base += " " + " ".join(self.keywords)
        return base