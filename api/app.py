from __future__ import annotations
 
import re
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
 
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
 
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
 # ════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("car-recommender")
 
 
# ════════════════════════════════════════════════════════
# ARTIFACT PATHS  ← adjust to match your output directory
# ════════════════════════════════════════════════════════

 
ARTIFACT_DIR = "../data/processed"

PIPELINE_PATH = f"{ARTIFACT_DIR}/pipeline.joblib"
MATRIX_PATH   = f"{ARTIFACT_DIR}/feature_matrix.npz"
DATA_PATH     = f"{ARTIFACT_DIR}/cars_dataframe.parquet"
# ════════════════════════════════════════════════════════
# MODULE-LEVEL CACHE
# ════════════════════════════════════════════════════════
# Loaded once at startup; shared across all requests.
# Thread-safe for reads (GIL + immutable after load).
 
CACHE: dict = {}
 
 
def load_artifacts() -> None:
    """
    Load pipeline, matrix, and dataframe into CACHE.
    Called at startup and optionally via /reload.
    """
    log.info("Loading artifacts...")
    t0 = time.perf_counter()
 
    pipeline = joblib.load(PIPELINE_PATH)
 
    # Support both sparse (.npz via scipy) and dense (.npz via numpy)
    try:
        X = sparse.load_npz(MATRIX_PATH)               # scipy sparse
        log.info("Matrix loaded as sparse CSR")
    except Exception:
        npz = np.load(MATRIX_PATH, mmap_mode="r")      # numpy dense (memory-mapped)
        X   = npz["X"]
        log.info("Matrix loaded as dense (memory-mapped)  shape=%s", X.shape)
 
    # Support parquet or pickle
    if DATA_PATH.endswith(".parquet"):
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_pickle(DATA_PATH)
 
    # Pre-build search_text once at load time (not per request)
    df = _build_search_text(df)
 
    CACHE["pipeline"]     = pipeline
    CACHE["X"]            = X
    CACHE["df"]           = df
    CACHE["search_texts"] = df["search_text"].tolist()   # avoid repeated .tolist()
 
    elapsed = time.perf_counter() - t0
    log.info("✅ Artifacts loaded in %.3fs  |  rows=%d  matrix=%s",
             elapsed, len(df), getattr(X, "shape", "?"))
 
 
# ════════════════════════════════════════════════════════
# TEXT HELPERS
# ════════════════════════════════════════════════════════
 
def _normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def _build_search_text(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate key fields into one searchable string. Runs ONCE at load."""
    df = df.copy()
    df["search_text"] = (
        df["name"].astype(str)         + " " +
        df["brand"].astype(str)        + " " +
        df["fuel_type"].astype(str)    + " " +
        df["transmission"].astype(str) + " " +
        df["model"].astype(str)
    ).apply(_normalize)
    return df
 
 
# ════════════════════════════════════════════════════════
# FUZZY MATCH
# ════════════════════════════════════════════════════════
 
def _fuzzy_match(
    user_input: str,
    threshold: int = 60,
) -> tuple[pd.DataFrame | None, float]:
    """
    Match user query against pre-built search_text column.
 
    token_set_ratio handles partial / reordered queries well:
      "honda turbo"  ↔  "honda civic 1.5 rs turbo"  → high score
    """
    query   = _normalize(user_input)
    choices = CACHE["search_texts"]
    df      = CACHE["df"]
 
    result = process.extractOne(
        query,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,
    )
 
    if result is None:
        log.info("No fuzzy match for '%s' (threshold=%d)", user_input, threshold)
        return None, 0.0
 
    matched_text, score, idx = result
    matched_row = df.iloc[[idx]]
 
    log.info("Fuzzy match: '%s'  score=%.1f",
             matched_row["name"].values[0], score)
 
    return matched_row, float(score)
 
 
# ════════════════════════════════════════════════════════
# CORE RECOMMENDATION LOGIC
# ════════════════════════════════════════════════════════
 
def _recommend(
    user_input: str,
    top_n: int,
    threshold: int,
) -> list[dict]:
    """
    1. Fuzzy-match query → dataset row
    2. Transform row via pipeline
    3. Cosine similarity against full matrix
    4. Return top_n results (excluding self)
    """
    pipeline = CACHE["pipeline"]
    X        = CACHE["X"]
    df       = CACHE["df"]
 
    # ── Match ─────────────────────────────────────────────
    matched_row, score = _fuzzy_match(user_input, threshold=threshold)
 
    if matched_row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No car matched '{user_input}' "
                f"(fuzzy threshold={threshold}). "
                "Try a different name or lower the threshold."
            ),
        )
 
    # ── Transform ─────────────────────────────────────────
    user_vector = pipeline.transform(matched_row)
 
    # Ensure sparse for cosine_similarity consistency
    if not sparse.issparse(user_vector):
        user_vector = sparse.csr_matrix(user_vector)
 
    # ── Similarity ────────────────────────────────────────
    sims = cosine_similarity(user_vector, X)[0]   # shape: (n_cars,)
 
    # ── Rank & exclude self ───────────────────────────────
    ordered_indices = np.argsort(sims)[::-1]
    matched_name    = matched_row["name"].values[0]
    results         = []
 
    for idx in ordered_indices:
        if df.iloc[idx]["name"] == matched_name:
            continue
        results.append(idx)
        if len(results) == top_n:
            break
 
    # ── Build response ────────────────────────────────────
    rec_df = df.iloc[results].copy()
    rec_df["similarity_score"] = np.round(sims[results], 4)
    rec_df["match_score"]      = round(score, 2)
    rec_df["matched_input"]    = matched_name
 
    columns = [
        "name", "brand", "city", "price", "model",
        "kilometer", "fuel_type", "engine", "transmission",
        "similarity_score",
    ]
    # Keep only columns that exist in df
    columns = [c for c in columns if c in rec_df.columns]
    rec_df  = rec_df[columns + ["match_score", "matched_input"]]
 
    return rec_df.to_dict(orient="records")
 
 
# ════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ════════════════════════════════════════════════════════
 
class CarResult(BaseModel):
    name:             str
    brand:            Optional[str]  = None
    city:             Optional[str]  = None
    price:            Optional[float] = None
    model:            Optional[int]  = None
    kilometer:        Optional[float] = None
    fuel_type:        Optional[str]  = None
    engine:           Optional[float] = None
    transmission:     Optional[str]  = None
    similarity_score: float
    match_score:      float   = Field(description="Fuzzy match score (0–100)")
    matched_input:    str     = Field(description="Actual car name matched in dataset")
 
 
class RecommendResponse(BaseModel):
    query:           str
    top_n:           int
    threshold:       int
    results:         list[CarResult]
    latency_ms:      float
 
 
class HealthResponse(BaseModel):
    status:          str
    artifacts_loaded: bool
    rows:            Optional[int] = None
 
 
# ════════════════════════════════════════════════════════
# LIFESPAN  (replaces deprecated @app.on_event)
# ════════════════════════════════════════════════════════
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts once before the server starts accepting requests."""
    load_artifacts()
    yield
    # (cleanup on shutdown if needed)
    CACHE.clear()
    log.info("Cache cleared. Server shut down.")
 
 
# ════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════
 
app = FastAPI(
    title       = "Car Recommender API",
    description = "Content-based car recommendations via cosine similarity.",
    version     = "1.0.0",
    lifespan    = lifespan,
)
 
 
# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════
 
@app.get(
    "/recommend",
    response_model = RecommendResponse,
    summary        = "Get car recommendations",
    tags           = ["Recommend"],
)
def recommend(
    q:         str = Query(...,    description="Car name or description", example="honda civic turbo"),
    top_n:     int = Query(5,      description="Number of results",       ge=1, le=50),
    threshold: int = Query(60,     description="Fuzzy match threshold",   ge=0, le=100),
):
    """
    Returns the **top_n** most similar cars to the query.
 
    - **q**: free-text car name (e.g. `"honda civic turbo"`, `"toyota hybrid automatic"`)
    - **top_n**: how many recommendations to return (1–50)
    - **threshold**: minimum fuzzy match score; lower = more lenient (0–100)
    """
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded yet.")
 
    t0 = time.perf_counter()
 
    results = _recommend(user_input=q, top_n=top_n, threshold=threshold)
 
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("Query='%s'  top_n=%d  results=%d  latency=%.1fms",
             q, top_n, len(results), latency_ms)
 
    return RecommendResponse(
        query      = q,
        top_n      = top_n,
        threshold  = threshold,
        results    = results,
        latency_ms = latency_ms,
    )
 
 
@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health / liveness check",
    tags           = ["Ops"],
)
def health():
    """Liveness probe — use this for Docker HEALTHCHECK or k8s readiness probe."""
    loaded = bool(CACHE)
    return HealthResponse(
        status           = "ok" if loaded else "loading",
        artifacts_loaded = loaded,
        rows             = len(CACHE["df"]) if loaded else None,
    )
 
 
@app.post(
    "/reload",
    summary = "Hot-reload artifacts from disk",
    tags    = ["Ops"],
)
def reload_artifacts():
    """
    Reload pipeline, matrix, and dataframe from disk **without restarting**.
    Call this after retraining the model.
    """
    try:
        CACHE.clear()
        load_artifacts()
        return JSONResponse({"status": "reloaded", "rows": len(CACHE["df"])})
    except Exception as exc:
        log.error("Reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
 
 
# ════════════════════════════════════════════════════════
# DEV ENTRY POINT
# ════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)