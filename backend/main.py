"""
backend/main.py — MarketLens Phase 5
====================================
FastAPI Application Entry Point.

Features:
  - CORS Middleware configuration for frontend dashboards
  - Health check and diagnostic endpoints (/health and /api/v1/health)
  - Interactive OpenAPI Swagger documentation (/docs and /redoc)
  - Lifespan context manager for warm model loading and startup verification
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.api.routes import router as api_router, health_check
from backend.intelligence.finalpipeline import get_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("marketlens.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models and verify connections on startup."""
    logger.info("Initializing MarketLens Orchestration Platform...")
    try:
        agent = get_agent()
        logger.info(
            "MarketLens Agent ready | FAISS vectors: %d | Model C features: %d",
            agent.evidence_engine.store.size,
            len(agent._feature_cols),
        )
    except Exception as e:
        logger.warning("Startup pre-warm encountered non-fatal notice: %s", e)
    yield
    logger.info("MarketLens Application shutdown complete.")


app = FastAPI(
    title="MarketLens AI Intelligence Engine",
    description=(
        "Institutional GenAI & Quantitative Systems API unifying event intelligence, "
        "FinBERT sentiment, ML abnormal return forecasting, and FAISS historical analogs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware (permits local dashboards, React/Next.js frontends, and API clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Diagnostic"])
async def root_health():
    """Root health check alias for cloud load balancers and orchestrators."""
    return await health_check()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Launching MarketLens API on %s:%d", host, port)
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
