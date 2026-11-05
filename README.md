# MarketLens

An Institutional GenAI & Quantitative Financial Intelligence Engine for Indian Equity Markets (NSE/BSE).

MarketLens continuously ingests financial news, extracts structured financial catalysts via local NLP, aligns pre-event technical market data, forecasts 3-day abnormal return outperformance ($AR_3 > +0.5\%$ vs NIFTY 50) using Machine Learning, retrieves historical semantic precedents via FAISS vector memory, and synthesizes institutional research memos.

---

## Architecture Overview

MarketLens executes an end-to-end 5-phase pipeline:

```
[ Financial News RSS Feeds ]
             │
             ▼
[ Phase 1: Data Foundation ] ───► PostgreSQL (companies, news, market_data, market_indices, events)
             │
             ▼
[ Phase 2: Event Intelligence ] ──► Local Qwen2.5-1.5B (Structured JSON) + FinBERT (Sentiment)
             │
             ▼
[ Phase 3: Financial ML Engine ] ──► 21-Feature Matrix (Momentum, Volatility, Volume Anomaly)
             │                       Model C Pipeline (AR3 > +0.5% Outperformance Classifier)
             ▼
[ Phase 4: Vector Evidence ] ────► FAISS IndexFlatIP (Cosine Similarity over all-MiniLM-L6-v2)
             │                       Empirical Base Rates (Win Rate, Median AR3, Top Precedents)
             ▼
[ Phase 5: GenAI Orchestrator ] ──► FastAPI REST Endpoints (`/api/v1/analyze`, `/api/v1/predict`, `/api/v1/evidence`)
                                     Institutional Research Briefing Memo (Ollama / Gemini / Fallback)
```

---

## Phase Breakdown

### Phase 1: Data Foundation
- Automated ingestion of historical daily market OHLCV data and NIFTY 50 benchmark index data.
- Relational PostgreSQL schema managed with SQLAlchemy & Alembic migrations.

### Phase 2: Event Intelligence
- **Structured Extraction:** Local `Qwen2.5-1.5B-Instruct` extracts event type, surprise, magnitude, and normalized ticker.
- **Continuous Sentiment:** `ProsusAI/finbert` generates positive, negative, and neutral probability distributions.

### Phase 3: Financial Machine Learning
- Strict chronological train/validation/test split to prevent lookahead bias.
- 21-feature matrix joining pre-event momentum (5d, 20d), 20-day realized volatility, volume anomaly ratios, and sentiment.
- Model C pipeline classifies whether an event generates positive 3-day abnormal return outperformance ($AR_3 > +0.5\%$).

### Phase 4: Historical Evidence Engine
- Local `sentence-transformers/all-MiniLM-L6-v2` encodes structured event context:
  `"[{ticker}] {event_type} | Surprise: {surprise} | {summary}"`
- `faiss.IndexFlatIP` vector store with Parquet metadata persistence.
- Calculates empirical historical base-rates: win rate, median $AR_3$, return spread, and top 3 analog narratives.

### Phase 5: GenAI Orchestrator & FastAPI Service
- Central `MarketLensAgent` orchestrating real-time extraction, technical alignment, ML inference, FAISS retrieval, and briefing synthesis.
- Multi-tier synthesis service supporting local Ollama (`qwen2.5:1.5b`), Google Gemini API, and a deterministic quantitative research fallback with strict hallucination guardrails.
- Production FastAPI application with OpenAPI documentation at `/docs`.

---

## Project Structure

```
MarketLens/
├── alembic/                      # Database migration scripts
│   ├── env.py
│   └── versions/
├── backend/
│   ├── intelligence/             # Phase 5: Orchestration & Synthesis
│   │   ├── finalpipeline.py
│   │   ├── prompt_templates.py
│   │   └── synthesizer.py
│   ├── api/                      # Phase 5: FastAPI REST API
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── memory/                   # Phase 4: Vector Store & Evidence Engine
│   │   ├── embedder.py
│   │   ├── evidence_engine.py
│   │   └── vector_store.py
│   ├── ml/                       # Phase 3: Financial Machine Learning
│   │   ├── dataset_builder.py
│   │   └── features.py
│   ├── services/                 # Phase 1 & 2: Ingestion & NLP Extraction
│   │   ├── event_extractor.py
│   │   ├── event_schemas.py
│   │   └── ingestion_service.py
│   └── main.py                   # FastAPI Application Entrypoint
├── database/                     # DB Connection & SQLAlchemy Models
│   ├── connection.py
│   └── models.py
├── notebooks/                    # Jupyter analysis & Model comparison
│   └── 03_model_comparison.ipynb
├── scripts/                      # CLI execution & verification scripts
│   ├── build_vector_index.py     # Batch embeds DB events into FAISS
│   ├── collect_market_data.py    # Downloads market and index data
│   ├── extract_events.py         # Batch extracts events from news
│   ├── run_phase3_training.py    # Trains and evaluates Model C
│   ├── test_agent.py             # End-to-end orchestration CLI test
│   ├── test_evidence_engine.py   # Tests historical analog retrieval
│   └── test_prediction.py        # Tests live Model C inference
├── .env.example                  # Environment configuration template
├── alembic.ini                   # Alembic configuration
└── requirements.txt              # Project dependencies
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- PostgreSQL database
- (Optional) [Ollama](https://ollama.com/) with `ollama run qwen2.5:1.5b` for fast local LLM synthesis

### 2. Installation
```bash
git clone https://github.com/Ashrafkkhan/MarketLens.git
cd MarketLens

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Setup
Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:
```bash
cp .env.example .env
```

Apply database migrations:
```bash
alembic upgrade head
```

### 4. Running the Verification Scripts

**Verify Live Inference Pipeline:**
```bash
python scripts/test_prediction.py
```

**Build Vector Memory Index:**
```bash
python scripts/build_vector_index.py
```

**Query Historical Analogs:**
```bash
python scripts/test_evidence_engine.py --query "Infosys reports 10% jump in profit, beating market consensus"
```

**Run End-to-End Orchestration Agent:**
```bash
python scripts/test_agent.py --headline "TCS bags mega $1.2B deal from European retail giant, revenue visibility strengthens" --ticker TCS
```

### 5. Launching the API Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`
* Health Check: `http://localhost:8000/health`
