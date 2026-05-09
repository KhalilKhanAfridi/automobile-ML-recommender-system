from __future__ import annotations
import re, time, logging, os
from contextlib import asynccontextmanager
from typing import Optional
import joblib, numpy as np, pandas as pd
import scipy.sparse as sparse
import dask.dataframe as dd
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

PIPELINE_PATH = os.path.join(BASE_DIR, "data", "processed", "pipeline.joblib")
MATRIX_PATH   = os.path.join(BASE_DIR, "data", "processed", "feature_matrix.npz")
PARQUET_PATH  = os.path.join(BASE_DIR, "data", "interim", "cars_cleaned.parquet")

CACHE = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("car-recommender")


def normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_artifacts():
    logger.info("Starting artifact load")
    pipeline = joblib.load(PIPELINE_PATH)

    try:
        X = sparse.load_npz(MATRIX_PATH)
    except Exception:
        X = np.load(MATRIX_PATH, mmap_mode="r")["X"]

    df = dd.read_parquet(
        PARQUET_PATH,
        blocksize="64MB",
        assume_missing=True
    ).compute()

    # CRITICAL: reset so that df.iloc[i] == matrix row i == search_texts[i]
    df = df.reset_index(drop=True)

    df["search_text"] = (
        df["name"].astype(str) + " " +
        df["brand"].astype(str) + " " +
        df["fuel_type"].astype(str) + " " +
        df["transmission"].astype(str) + " " +
        df["model"].astype(str)
    ).apply(normalize)

    assert X.shape[0] == len(df), (
        f"ALIGNMENT ERROR: matrix={X.shape[0]} rows, df={len(df)} rows. "
        "Re-run preprocessing."
    )

    CACHE["pipeline"]     = pipeline
    CACHE["X"]            = X
    CACHE["df"]           = df
    CACHE["search_texts"] = df["search_text"].tolist()

    null_links = df["link"].isna().sum() if "link" in df.columns else "N/A"
    logger.info(f"Artifacts loaded: {len(df)} rows, matrix={X.shape}, null_links={null_links}")
    logger.info("Sample alignment:\n%s", df[["name", "link"]].head(3).to_string())


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    CACHE.clear()
    logger.info("Cache cleared. Server shut down.")


app = FastAPI(
    title="Car Recommender API",
    description="Content-based car recommendations via cosine similarity.",
    version="1.0.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════

class CarResult(BaseModel):
    name:             str
    brand:            Optional[str]   = None
    city:             Optional[str]   = None
    fuel_type:        Optional[str]   = None
    transmission:     Optional[str]   = None
    link:             Optional[str]   = None
    price:            Optional[float] = None
    kilometer:        Optional[float] = None
    engine:           Optional[float] = None
    model:            Optional[int]   = None
    similarity_score: float
    match_score:      float
    matched_input:    str

class CarDetailResult(BaseModel):
    name:         str
    brand:        Optional[str]   = None
    city:         Optional[str]   = None
    fuel_type:    Optional[str]   = None
    transmission: Optional[str]   = None
    link:         Optional[str]   = None
    price:        Optional[float] = None
    kilometer:    Optional[float] = None
    engine:       Optional[float] = None
    model:        Optional[int]   = None
    match_score:  float
    message:      str

class RecommendResponse(BaseModel):
    query:      str
    top_n:      int
    threshold:  int
    latency_ms: float
    results:    list[CarResult]

class FilterResponse(BaseModel):
    filters_applied: dict
    total_found:     int
    results:         list[CarResult]

class SearchResponse(BaseModel):
    query:   str
    results: list[CarResult]

class HealthResponse(BaseModel):
    status:           str
    artifacts_loaded: bool
    rows:             Optional[int]       = None
    matrix_shape:     Optional[list[int]] = None


# ════════════════════════════════════════════════════════
# CORE HELPERS
# ════════════════════════════════════════════════════════

def fuzzy_match(user_input: str, threshold: int) -> tuple[pd.DataFrame | None, float]:
    query = normalize(user_input)
    result = process.extractOne(
        query,
        CACHE["search_texts"],
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,
    )
    if result is None:
        return None, 0.0

    matched_text, score, idx = result

    # iloc[idx] is safe: df was reset_index(drop=True) at load,
    # so list index == positional index always
    matched_row = CACHE["df"].iloc[[idx]]
    logger.info(
        "Fuzzy match: '%s'  link='%s'  score=%.1f",
        matched_row["name"].iloc[0],
        matched_row["link"].iloc[0] if "link" in matched_row.columns else "N/A",
        score,
    )
    return matched_row, float(score)


def recommend_cars(user_input: str, top_n: int, threshold: int) -> list[dict]:
    matched_row, score = fuzzy_match(user_input, threshold)
    if matched_row is None:
        return []

    user_vector = CACHE["pipeline"].transform(matched_row)
    if not sparse.issparse(user_vector):
        user_vector = sparse.csr_matrix(user_vector)

    # sims[i] corresponds to CACHE["df"].iloc[i] — guaranteed by reset_index at load
    sims = cosine_similarity(user_vector, CACHE["X"])[0]

    ordered = np.argsort(sims)[::-1]
    matched_name = matched_row["name"].iloc[0]

    # collect positional indices of top_n results (excluding matched car)
    result_indices = []
    for idx in ordered:
        if CACHE["df"].iloc[idx]["name"] != matched_name:
            result_indices.append(int(idx))
            if len(result_indices) >= top_n:
                break

    # ── THE FIX ───────────────────────────────────────────────────────────
    # Extract similarity scores BEFORE slicing df, using positional indices.
    # Never do sims[result_indices] after iloc — the df slice retains old
    # index labels which would misalign scores and therefore links.
    result_scores = sims[result_indices]           # numpy positional — correct
    rec_df = CACHE["df"].iloc[result_indices].copy()
    rec_df = rec_df.reset_index(drop=True)         # ← reset so new index is 0..N-1
    rec_df["similarity_score"] = result_scores     # ← assign by position, not label
    rec_df["match_score"]      = score
    rec_df["matched_input"]    = matched_name
    # ─────────────────────────────────────────────────────────────────────

    cols = [
        "name", "brand", "city", "price", "model", "kilometer",
        "fuel_type", "engine", "transmission", "link",
        "similarity_score", "match_score", "matched_input",
    ]
    cols = [c for c in cols if c in rec_df.columns]
    return rec_df[cols].to_dict("records")


# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

@app.get("/recommend", response_model=RecommendResponse,
         summary="Get ML-based car recommendations", tags=["recommendation"])
async def recommend(
    q:         str = Query(..., description="Car name or description"),
    top_n:     int = Query(10,  description="Number of recommendations"),
    threshold: int = Query(60,  description="Fuzzy match threshold (0-100)"),
):
    """
    Fuzzy-match the query to a car in the dataset, then return the
    top_n most similar cars using cosine similarity on the feature matrix.
    """
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")
    t0 = time.time()
    results = recommend_cars(q, top_n, threshold)
    latency_ms = (time.time() - t0) * 1000
    if not results:
        raise HTTPException(status_code=404, detail="No matching car found")
    logger.info("Recommend: query='%s'  results=%d  latency=%.1fms", q, len(results), latency_ms)
    return RecommendResponse(query=q, top_n=top_n, threshold=threshold,
                             latency_ms=latency_ms, results=results)


@app.get("/search", response_model=SearchResponse,
         summary="Fuzzy text search (no ML)", tags=["search"])
async def search(
    q:         str = Query(..., description="Search query"),
    top_n:     int = Query(10,  description="Max results"),
    threshold: int = Query(50,  description="Match threshold"),
):
    """
    Fuzzy text search without ML similarity scoring.
    Returns cars whose search text matches the query above threshold.
    """
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")
    query = normalize(q)
    raw = process.extract(
        query, CACHE["search_texts"],
        scorer=fuzz.token_set_ratio,
        limit=top_n,
        score_cutoff=threshold,
    )
    if not raw:
        raise HTTPException(status_code=404, detail="No results found")

    results = []
    for _, score, idx in sorted(raw, key=lambda x: x[1], reverse=True):
        row = CACHE["df"].iloc[idx]
        results.append({
            "name":             row["name"],
            "brand":            row.get("brand"),
            "city":             row.get("city"),
            "fuel_type":        row.get("fuel_type"),
            "transmission":     row.get("transmission"),
            "link":             row.get("link"),
            "price":            row.get("price"),
            "kilometer":        row.get("kilometer"),
            "engine":           row.get("engine"),
            "model":            row.get("model"),
            "similarity_score": 0.0,
            "match_score":      float(score),
            "matched_input":    row["name"],
        })
    logger.info("Search: query='%s'  results=%d", q, len(results))
    return SearchResponse(query=q, results=results)


@app.get("/car", response_model=CarDetailResult,
         summary="Get single car details by name", tags=["lookup"])
async def get_car(
    name:      str = Query(..., description="Car name to match"),
    threshold: int = Query(60,  description="Match threshold"),
):
    """Fuzzy-match a single car by name and return all its details."""
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")
    matched_row, score = fuzzy_match(name, threshold)
    if matched_row is None:
        raise HTTPException(status_code=404, detail="Car not found")
    row = matched_row.iloc[0]
    return CarDetailResult(
        name=row["name"], brand=row.get("brand"), city=row.get("city"),
        fuel_type=row.get("fuel_type"), transmission=row.get("transmission"),
        link=row.get("link"), price=row.get("price"), kilometer=row.get("kilometer"),
        engine=row.get("engine"), model=row.get("model"),
        match_score=score, message="Car found",
    )


@app.get("/filter", response_model=FilterResponse,
         summary="Filter cars by attributes (no ML)", tags=["filter"])
async def filter_cars(
    brand:        Optional[str]   = Query(None, description="Brand filter"),
    city:         Optional[str]   = Query(None, description="City filter"),
    fuel_type:    Optional[str]   = Query(None, description="Fuel type filter"),
    transmission: Optional[str]   = Query(None, description="Transmission filter"),
    year_min:     Optional[int]   = Query(None, description="Min model year"),
    year_max:     Optional[int]   = Query(None, description="Max model year"),
    price_min:    Optional[float] = Query(None, description="Min price"),
    price_max:    Optional[float] = Query(None, description="Max price"),
    km_max:       Optional[float] = Query(None, description="Max kilometers"),
    top_n:        int             = Query(20,   description="Max results"),
):
    """Filter cars by any combination of attributes. All filters are optional."""
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    df = CACHE["df"].copy()
    filters_applied = {}

    if brand:
        df = df[df["brand"].str.contains(brand, case=False, na=False)]
        filters_applied["brand"] = brand
    if city:
        df = df[df["city"].str.contains(city, case=False, na=False)]
        filters_applied["city"] = city
    if fuel_type:
        df = df[df["fuel_type"].str.contains(fuel_type, case=False, na=False)]
        filters_applied["fuel_type"] = fuel_type
    if transmission:
        df = df[df["transmission"].str.contains(transmission, case=False, na=False)]
        filters_applied["transmission"] = transmission
    if year_min is not None:
        df = df[df["model"] >= year_min]
        filters_applied["year_min"] = year_min
    if year_max is not None:
        df = df[df["model"] <= year_max]
        filters_applied["year_max"] = year_max
    if price_min is not None:
        df = df[df["price"] >= price_min]
        filters_applied["price_min"] = price_min
    if price_max is not None:
        df = df[df["price"] <= price_max]
        filters_applied["price_max"] = price_max
    if km_max is not None:
        df = df[df["kilometer"] <= km_max]
        filters_applied["km_max"] = km_max

    df = df.sort_values("price").head(top_n)

    if df.empty:
        raise HTTPException(status_code=404, detail="No cars match the filters")

    results = []
    for _, row in df.iterrows():
        results.append({
            "name":             row["name"],
            "brand":            row.get("brand"),
            "city":             row.get("city"),
            "fuel_type":        row.get("fuel_type"),
            "transmission":     row.get("transmission"),
            "link":             row.get("link"),
            "price":            row.get("price"),
            "kilometer":        row.get("kilometer"),
            "engine":           row.get("engine"),
            "model":            row.get("model"),
            "similarity_score": 0.0,
            "match_score":      0.0,
            "matched_input":    "",
        })

    return FilterResponse(filters_applied=filters_applied,
                          total_found=len(results), results=results)


@app.get("/brands", summary="Get unique brands", tags=["metadata"])
async def get_brands():
    """Get sorted list of unique car brands."""
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")
    return {"brands": sorted(CACHE["df"]["brand"].dropna().unique().tolist())}


@app.get("/cities", summary="Get unique cities", tags=["metadata"])
async def get_cities():
    """Get sorted list of unique cities."""
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")
    return {"cities": sorted(CACHE["df"]["city"].dropna().unique().tolist())}


@app.get("/fuel-types", summary="Get unique fuel types", tags=["metadata"])
async def get_fuel_types():
    """Get sorted list of unique fuel types."""
    if not CACHE:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")
    return {"fuel_types": sorted(CACHE["df"]["fuel_type"].dropna().unique().tolist())}


@app.get("/health", response_model=HealthResponse,
         summary="Health check", tags=["health"])
async def health():
    """Check if artifacts are loaded and return basic stats."""
    if CACHE:
        return HealthResponse(
            status="ok", artifacts_loaded=True,
            rows=len(CACHE["df"]),
            matrix_shape=list(CACHE["X"].shape),
        )
    return HealthResponse(status="loading", artifacts_loaded=False)


@app.post("/reload", summary="Hot-reload artifacts without restart", tags=["admin"])
async def reload():
    """Reload pipeline, matrix, and dataframe from disk without restarting."""
    try:
        CACHE.clear()
        load_artifacts()
        return {"status": "reloaded", "rows": len(CACHE["df"])}
    except Exception as exc:
        logger.error("Reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
