"""FastAPI service for the credit scorecard model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from scorecard.inference import score_new_applicant

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    app.state.woe_tables = joblib.load(ARTIFACTS_DIR / "woe_tables.pkl")
    app.state.scorecard_table = joblib.load(ARTIFACTS_DIR / "scorecard_table.pkl")
    app.state.binning_artifacts = joblib.load(ARTIFACTS_DIR / "binning_artifacts.pkl")
    app.state.feature_order = joblib.load(ARTIFACTS_DIR / "keep_features.pkl")
    yield
    # no cleanup needed on shutdown


app = FastAPI(title="Credit Scorecard API", version="0.1.0", lifespan=lifespan)


class ApplicantRequest(BaseModel):
    checking_status: str
    credit_history: str
    purpose: str
    savings_status: str
    employment: str
    personal_status_sex: str
    other_parties: str
    property_magnitude: str
    other_payment_plans: str
    housing: str
    foreign_worker: str
    duration: int
    credit_amount: int
    installment_rate: int
    age: int


class ScoreResponse(BaseModel):
    pd: float
    points: float
    risk_band: Literal["low", "medium", "high"]


def _risk_band(pd_value: float) -> Literal["low", "medium", "high"]:
    if pd_value < 0.15:
        return "low"
    if pd_value < 0.40:
        return "medium"
    return "high"


@app.post("/credit-scorecard/score", response_model=ScoreResponse)
def score(applicant: ApplicantRequest) -> ScoreResponse:
    raw_features = applicant.model_dump()
    result = score_new_applicant(
        raw_features,
        app.state.binning_artifacts,
        app.state.woe_tables,
        app.state.scorecard_table,
        app.state.model,
        app.state.feature_order,
    )
    return ScoreResponse(
        pd=result["pd"], points=result["points"], risk_band=_risk_band(result["pd"])
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
