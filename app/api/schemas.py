from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    ticker: str
    company_name: str
    currency: str

    last_close: float
    predicted_close: float
    expected_move_percent: float
    confidence_lower: float
    confidence_upper: float
    direction_probability: float   # 0.0–1.0; >0.5 bullish

    risk_score: int                # 0–100
    risk_label: str                # Low / Moderate / High / Very High

    model_name: str
    model_version: str
    rmse: Optional[float]
    inference_time_ms: int

    top_features: list[str]
    generated_at: datetime


class TrainingStatusResponse(BaseModel):
    """
    Returned with HTTP 202 when /predict/{ticker} is hit for a ticker that
    has no cached and no persisted model yet. Training has been kicked off
    in the background — the client (Spring Boot) should poll /predict again
    after check_again_in_seconds instead of the request blocking for minutes.
    """
    ticker: str
    status: str                    # "training"
    message: str
    check_again_in_seconds: int


class BacktestResponse(BaseModel):
    ticker: str
    model_name: str
    model_version: str
    average_error: float
    direction_accuracy: float      # percentage
    max_error: float
    test_days: int
    ran_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    detail: str