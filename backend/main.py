from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="Adaptive Tone Engine")

@app.get("/")
def home():
    return {
        "project": "Adaptive Tone Engine",
        "status": "Day 1",
        "docs": "/docs",
    }

@app.get("/status")
def status():
    return {
        "status": "ok",
        "service": "adaptive-tone-engine",
        "version": "0.1.0",
    }

# Schemas
class AnalyzeRequest(BaseModel):
    instrument: str = Field("guitar")
    style: str = Field("unknown")
    pickup_position: Optional[str] = None
    signal_level_db: Optional[float] = None
    brightness: Optional[float] = Field(None, ge=0, le=1)
    muddiness: Optional[float] = Field(None, ge=0, le=1)
    harshness: Optional[float] = Field(None, ge=0, le=1)

class AnalyzeResponse(BaseModel):
    status: str
    summary: str
    detected_traits: List[str]

class Recommendation(BaseModel):
    control: str
    action: str
    reason: str

class RecommendRequest(BaseModel):
    target_tone: str = Field("balanced")
    analysis: Optional[AnalyzeResponse] = None

class RecommendResponse(BaseModel):
    status: str
    target_tone: str
    recommendations: List[Recommendation]

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    traits: List[str] = []

    # Trait logic
    if req.brightness is not None:
        if req.brightness >= 0.7:
            traits.append("bright")
        if req.brightness <= 0.3:
            traits.append("dark")
    if req.muddiness is not None and req.muddiness >= 0.6:
        traits.append("muddy")
    if req.harshness is not None and req.harshness >= 0.6:
        traits.append("harsh")
    if req.signal_level_db is not None:
        if req.signal_level_db > -6:
            traits.append("hot input")
        if req.signal_level_db < -24:
            traits.append("low input")

    if not traits:
        traits = ["neutral"]

    summary = f"Detected {req.style} tone profile." if traits and traits != ["neutral"] else "No strong traits detected."
    return AnalyzeResponse(status="ok", summary=summary, detected_traits=traits)

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs: List[Recommendation] = []
    traits = req.analysis.detected_traits if req.analysis else []

    if "muddy" in traits:
        recs.append(Recommendation(control="bass", action="reduce", reason="Reduces low-mid muddiness."))
    if "harsh" in traits:
        recs.append(Recommendation(control="treble", action="reduce", reason="Smooths sharp upper-mid frequencies."))
    if "bright" in traits:
        recs.append(Recommendation(control="presence", action="reduce", reason="Lowers forward high-frequency energy."))
    if "low input" in traits:
        recs.append(Recommendation(control="input_gain", action="increase", reason="Raises input level for stronger signal."))

    if not recs:
        # fallback recommendation
        recs.append(Recommendation(control="midrange", action="increase slightly", reason="Adds body and presence."))

    return RecommendResponse(status="ok", target_tone=req.target_tone, recommendations=recs)
