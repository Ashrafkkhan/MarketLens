"""
backend/ml/dataset_builder.py — MarketLens Phase 3
====================================================
Builds a flat historical event matrix from the PostgreSQL database by joining:
  events -> news (published_at) -> market_data (stock prices) -> market_indices (NIFTY50)

Market-close alignment rules (IST = UTC+5:30):
  - If news published BEFORE 15:30 IST (10:00 UTC) on a trading day:
      Reaction day T0 = that calendar day; benchmark price P0 = prev day close.
  - If news published AFTER 15:30 IST or on a weekend/holiday:
      Reaction day T0 = next trading session; P0 = most recent close prior to T0.

Forward prices are aligned for T+1, T+3, T+5 trading days from T0.
Abnormal Return AR_k = (P_k - P0)/P0 - (M_k - M0)/M0  where M = NIFTY50 close.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from database.connection import SessionLocal
from database.models import Event, MarketData, MarketIndex, News

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NIFTY_SYMBOL = "^NSEI"
MARKET_CLOSE_UTC_HOUR = 10   # 15:30 IST = 10:00 UTC
FORWARD_HORIZONS = [1, 3, 5]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_market_data(session: Session) -> pd.DataFrame:
    """Load market_data table -> DataFrame with columns [ticker, date, close, volume]."""
    rows = session.query(
        MarketData.ticker,
        MarketData.timestamp,
        MarketData.close,
        MarketData.volume,
    ).all()
    df = pd.DataFrame(rows, columns=["ticker", "timestamp", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    return df.reset_index(drop=True)


def _load_index_data(session: Session, symbol: str = NIFTY_SYMBOL) -> pd.DataFrame:
    """Load market_indices for NIFTY50 -> DataFrame with columns [date, index_close]."""
    rows = session.query(
        MarketIndex.timestamp,
        MarketIndex.close,
    ).filter(MarketIndex.index_symbol == symbol).all()
    df = pd.DataFrame(rows, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["close"] = df["close"].astype(float)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return df[["date", "close"]].rename(columns={"close": "index_close"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Alignment logic
# ---------------------------------------------------------------------------

def _get_trading_days(mkt_df: pd.DataFrame, ticker: str) -> list:
    return sorted(mkt_df[mkt_df["ticker"] == ticker]["date"].unique().tolist())


def _find_t0_and_p0(
    published_at: datetime,
    ticker: str,
    mkt_df: pd.DataFrame,
    trading_days: list,
) -> tuple:
    """Return (T0, P0) for the event, or (None, None) if data unavailable."""
    if len(trading_days) < 2:
        return None, None

    pub_utc = pd.Timestamp(published_at)
    if pub_utc.tzinfo is None:
        pub_utc = pub_utc.tz_localize("UTC")
    else:
        pub_utc = pub_utc.tz_convert("UTC")

    pub_date = pub_utc.normalize()
    before_close = pub_utc.hour < MARKET_CLOSE_UTC_HOUR

    # Determine T0
    if before_close and pub_date in trading_days:
        t0 = pub_date
    else:
        future = [d for d in trading_days if d > pub_date]
        if not future:
            return None, None
        t0 = future[0]

    # Determine prev day for P0
    try:
        t0_idx = trading_days.index(t0)
    except ValueError:
        return None, None

    if t0_idx == 0:
        return None, None

    prev_day = trading_days[t0_idx - 1]
    row = mkt_df[(mkt_df["ticker"] == ticker) & (mkt_df["date"] == prev_day)]
    if row.empty:
        return None, None

    p0 = float(row.iloc[0]["close"])
    return t0, p0


def _forward_close(
    mkt_df: pd.DataFrame,
    ticker: str,
    t0: object,
    k: int,
    trading_days: list,
) -> Optional[float]:
    """Return close price k trading days after T0 for a given ticker."""
    try:
        t0_idx = trading_days.index(t0)
    except ValueError:
        return None
    fwd_idx = t0_idx + k
    if fwd_idx >= len(trading_days):
        return None
    fwd_day = trading_days[fwd_idx]
    row = mkt_df[(mkt_df["ticker"] == ticker) & (mkt_df["date"] == fwd_day)]
    if row.empty:
        return None
    return float(row.iloc[0]["close"])


def _index_on_day(idx_df: pd.DataFrame, day) -> Optional[float]:
    row = idx_df[idx_df["date"] == day]
    if row.empty:
        return None
    return float(row.iloc[0]["index_close"])


def _ar(pk: Optional[float], mk: Optional[float], p0: float, m0: Optional[float]) -> Optional[float]:
    """Compute Abnormal Return = stock_ret - mkt_ret."""
    if pk is None or mk is None or m0 is None or m0 == 0 or p0 == 0:
        return None
    return (pk - p0) / p0 - (mk - m0) / m0


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_event_matrix(session: Optional[Session] = None) -> pd.DataFrame:
    """
    Build and return the flat historical event matrix.

    Columns produced:
      event_id, news_id, ticker, published_at, event_date,
      p0, p1, p3, p5, m0, m1, m3, m5,
      ar_1, ar_3, ar_5, target_AR3_positive,
      event_type, gemini_sentiment, magnitude, surprise,
      finbert_positive, finbert_negative, finbert_neutral, summary
    """
    close_session = session is None
    if session is None:
        session = SessionLocal()

    try:
        logger.info("Loading market and index data...")
        mkt_df = _load_market_data(session)
        idx_df = _load_index_data(session)
        logger.info(
            "Loaded %d market rows (%d tickers) | %d index rows",
            len(mkt_df), mkt_df["ticker"].nunique(), len(idx_df),
        )

        # Load ticker-events + publication timestamp
        raw = (
            session.query(Event, News.published_at)
            .join(News, Event.news_id == News.id)
            .filter(Event.ticker.isnot(None))
            .all()
        )
        logger.info("Processing %d ticker-linked events...", len(raw))

        records = []
        skipped = 0

        for evt, published_at in raw:
            ticker = evt.ticker

            if ticker not in mkt_df["ticker"].values:
                skipped += 1
                continue

            trading_days = _get_trading_days(mkt_df, ticker)
            t0, p0 = _find_t0_and_p0(published_at, ticker, mkt_df, trading_days)

            if t0 is None or p0 is None or p0 == 0:
                skipped += 1
                continue

            # Forward stock prices
            p1 = _forward_close(mkt_df, ticker, t0, 1, trading_days)
            p3 = _forward_close(mkt_df, ticker, t0, 3, trading_days)
            p5 = _forward_close(mkt_df, ticker, t0, 5, trading_days)

            if p1 is None and p3 is None and p5 is None:
                skipped += 1
                continue

            # NIFTY50 benchmark — m0 = index close on the day before T0
            try:
                t0_idx = trading_days.index(t0)
            except ValueError:
                skipped += 1
                continue

            m0_day = trading_days[t0_idx - 1] if t0_idx > 0 else None
            m0 = _index_on_day(idx_df, m0_day) if m0_day else None

            # Forward index closes: use same T+k dates as the ticker's trading calendar
            m1 = _index_on_day(idx_df, trading_days[t0_idx + 1]) if t0_idx + 1 < len(trading_days) else None
            m3 = _index_on_day(idx_df, trading_days[t0_idx + 3]) if t0_idx + 3 < len(trading_days) else None
            m5 = _index_on_day(idx_df, trading_days[t0_idx + 5]) if t0_idx + 5 < len(trading_days) else None

            # Abnormal returns
            ar1 = _ar(p1, m1, p0, m0)
            ar3 = _ar(p3, m3, p0, m0)
            ar5 = _ar(p5, m5, p0, m0)

            if ar3 is None:
                skipped += 1
                continue

            records.append({
                "event_id":              evt.id,
                "news_id":               evt.news_id,
                "ticker":                ticker,
                "published_at":          published_at,
                "event_date":            t0,
                "p0":                    p0,
                "p1":                    p1,
                "p3":                    p3,
                "p5":                    p5,
                "m0":                    m0,
                "m1":                    m1,
                "m3":                    m3,
                "m5":                    m5,
                "ar_1":                  ar1,
                "ar_3":                  ar3,
                "ar_5":                  ar5,
                "target_AR3_positive":   int(ar3 > 0.005),
                "event_type":            evt.event_type,
                "gemini_sentiment":      evt.gemini_sentiment,
                "magnitude":             evt.magnitude,
                "surprise":              evt.surprise,
                "finbert_positive":      evt.finbert_positive,
                "finbert_negative":      evt.finbert_negative,
                "finbert_neutral":       evt.finbert_neutral,
                "summary":               evt.summary,
            })

        logger.info(
            "Event matrix: %d rows built | %d skipped (missing price/index data)",
            len(records), skipped,
        )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("event_date").reset_index(drop=True)
        return df

    finally:
        if close_session:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    df = build_event_matrix()
    print(f"\nEvent matrix shape: {df.shape}")
    if not df.empty:
        print("\nColumn dtypes:")
        print(df.dtypes)
        print("\nFirst 10 rows:")
        cols = ["ticker", "event_date", "p0", "ar_3", "target_AR3_positive", "event_type"]
        print(df[cols].head(10).to_string())
        print(f"\nTarget distribution:\n{df['target_AR3_positive'].value_counts()}")
        print(f"\nEvent types:\n{df['event_type'].value_counts()}")
