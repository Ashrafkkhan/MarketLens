"""
YFinance Market Data Provider for MarketLens.

Fetches historical daily OHLCV data for Indian equities (NSE)
and broad market indices via the yfinance library.

Validation rules applied:
  - Drop rows where Volume < 0.
  - Assert Low <= Open <= High and Low <= Close <= High.
  - Drop rows with NaN values in OHLCV price columns.
  - All timestamps converted to UTC before returning.
"""
import logging
import time
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import BaseMarketProvider, IndexRecord, MarketRecord

logger = logging.getLogger(__name__)

# Price columns that must be non-NaN for a row to be valid
_REQUIRED_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Default sleep between yfinance API calls (seconds)
DEFAULT_SLEEP_SECONDS = 1.0


def _ensure_nse_suffix(ticker: str) -> str:
    """
    Append '.NS' suffix for NSE-listed stocks if not already present.
    Index symbols (starting with '^') are passed through unchanged.
    """
    ticker = ticker.strip().upper()
    if ticker.startswith("^"):
        return ticker
    if "." in ticker:
        return ticker          # Already has a suffix (.NS, .BO, etc.)
    return f"{ticker}.NS"


def _base_ticker(ticker: str) -> str:
    """
    Return the base ticker symbol without any exchange suffix.
    Used to normalise the ticker stored in the database so it always
    matches the value in the `companies` table (e.g. 'INFY', not 'INFY.NS').

    Examples:
        'INFY.NS' → 'INFY'
        'INFY'    → 'INFY'
        '^NSEI'   → '^NSEI'  (index symbols unchanged)
    """
    ticker = ticker.strip().upper()
    if ticker.startswith("^"):
        return ticker
    return ticker.split(".")[0]


def _validate_ohlcv_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Apply business-logic validation to a raw yfinance OHLCV DataFrame.

    Steps:
      1. Flatten multi-level columns produced by yfinance >= 0.2 when
         downloading a single ticker.
      2. Drop rows with NaN in required price columns.
      3. Drop rows where Volume < 0.
      4. Drop rows violating OHLC price bounds (Low <= Open/Close <= High).

    Returns a cleaned DataFrame. Logs a warning for each dropped row.
    """
    # --- 1. Flatten MultiIndex columns (yfinance 0.2+ returns them) ----------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename 'Adj Close' → 'AdjClose' if present to avoid name confusion
    df = df.rename(columns={"Adj Close": "AdjClose"})

    # --- 2. Drop rows with NaN in required columns ---------------------------
    before = len(df)
    df = df.dropna(subset=_REQUIRED_PRICE_COLS)
    dropped_nan = before - len(df)
    if dropped_nan:
        logger.warning(
            "[%s] Dropped %d rows with NaN in OHLCV columns.", symbol, dropped_nan
        )

    # --- 3. Drop rows with negative volume -----------------------------------
    before = len(df)
    df = df[df["Volume"] >= 0]
    dropped_vol = before - len(df)
    if dropped_vol:
        logger.warning(
            "[%s] Dropped %d rows with negative Volume.", symbol, dropped_vol
        )

    # --- 4. Price-bound validation -------------------------------------------
    invalid_mask = (
        (df["Low"] > df["Open"])
        | (df["Open"] > df["High"])
        | (df["Low"] > df["Close"])
        | (df["Close"] > df["High"])
    )
    dropped_bounds = invalid_mask.sum()
    if dropped_bounds:
        logger.warning(
            "[%s] Dropped %d rows failing OHLC price-bound assertion "
            "(Low <= Open/Close <= High).",
            symbol,
            dropped_bounds,
        )
        df = df[~invalid_mask]

    return df


def _to_utc_datetime(ts: pd.Timestamp) -> datetime:
    """
    Convert a pandas Timestamp to a Python datetime that is explicitly
    timezone-aware in UTC.  Handles both tz-aware and tz-naive inputs.
    """
    if ts.tzinfo is not None:
        return ts.to_pydatetime().astimezone(timezone.utc)
    # tz-naive timestamps from yfinance are implicitly in the exchange's
    # local time (IST for NSE).  We treat daily bars as UTC midnight
    # since they are date-level granularity.
    return ts.to_pydatetime().replace(tzinfo=timezone.utc)


class YFinanceProvider(BaseMarketProvider):
    """
    Concrete market data provider backed by the yfinance library.

    Usage::

        provider = YFinanceProvider(sleep_seconds=1.0)
        records  = provider.fetch_ohlcv("INFY", date(2024, 1, 1), date(2024, 6, 30))
    """

    def __init__(self, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> None:
        """
        Args:
            sleep_seconds:  Seconds to sleep after each yfinance network call
                            to respect rate limits.
        """
        self.sleep_seconds = sleep_seconds

    # ------------------------------------------------------------------
    # BaseMarketProvider interface
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[MarketRecord]:
        """
        Fetch and validate daily OHLCV data for a single NSE-listed equity.

        The '.NS' suffix is automatically appended if missing.  All returned
        timestamps are UTC-aware.

        Args:
            ticker:  Ticker symbol (with or without '.NS' suffix).
            start:   Inclusive start date.
            end:     Inclusive end date.

        Returns:
            List of MarketRecord objects.  Empty list on error or no data.
        """
        # yf_symbol: used only for the yfinance API call (e.g. 'INFY.NS')
        # db_ticker:  stored in market_data.ticker — must match companies.ticker FK (e.g. 'INFY')
        yf_symbol = _ensure_nse_suffix(ticker)
        db_ticker  = _base_ticker(ticker)
        logger.info(
            "[%s] Fetching OHLCV data from %s to %s (yf_symbol=%s).",
            db_ticker, start, end, yf_symbol,
        )

        try:
            raw_df: pd.DataFrame = yf.download(
                yf_symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,   # Adjust for splits/dividends in-place
                actions=False,
                threads=False,
            )
        except Exception as exc:
            logger.error("[%s] yfinance download failed: %s", db_ticker, exc)
            return []
        finally:
            time.sleep(self.sleep_seconds)

        if raw_df.empty:
            logger.warning("[%s] No data returned by yfinance.", db_ticker)
            return []

        df = _validate_ohlcv_dataframe(raw_df.copy(), db_ticker)

        if df.empty:
            logger.warning("[%s] All rows dropped after validation.", db_ticker)
            return []

        records: list[MarketRecord] = []
        for ts, row in df.iterrows():
            try:
                records.append(
                    MarketRecord(
                        ticker=db_ticker,   # ← FK-safe: matches companies.ticker exactly
                        timestamp=_to_utc_datetime(ts),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row["Volume"]),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Skipping row at %s due to error: %s", db_ticker, ts, exc
                )

        logger.info("[%s] Fetched %d valid OHLCV records.", db_ticker, len(records))
        return records

    def fetch_index_ohlcv(
        self,
        index_symbol: str,
        index_name: str,
        start: date,
        end: date,
    ) -> list[IndexRecord]:
        """
        Fetch and validate daily OHLCV data for a broad market index.

        Args:
            index_symbol:  yfinance symbol (e.g. '^NSEI', '^NSEBANK').
            index_name:    Human-readable name stored in DB (e.g. 'NIFTY 50').
            start:         Inclusive start date.
            end:           Inclusive end date.

        Returns:
            List of IndexRecord objects.  Empty list on error or no data.
        """
        logger.info(
            "[%s] Fetching index OHLCV data from %s to %s.",
            index_symbol,
            start,
            end,
        )

        try:
            raw_df: pd.DataFrame = yf.download(
                index_symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,
                actions=False,
                threads=False,
            )
        except Exception as exc:
            logger.error("[%s] yfinance download failed: %s", index_symbol, exc)
            return []
        finally:
            time.sleep(self.sleep_seconds)

        if raw_df.empty:
            logger.warning("[%s] No data returned by yfinance.", index_symbol)
            return []

        df = _validate_ohlcv_dataframe(raw_df.copy(), index_symbol)

        if df.empty:
            logger.warning("[%s] All rows dropped after validation.", index_symbol)
            return []

        records: list[IndexRecord] = []
        for ts, row in df.iterrows():
            try:
                records.append(
                    IndexRecord(
                        index_name=index_name,
                        index_symbol=index_symbol,
                        timestamp=_to_utc_datetime(ts),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row["Volume"]),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Skipping row at %s due to error: %s",
                    index_symbol,
                    ts,
                    exc,
                )

        logger.info(
            "[%s] Fetched %d valid index OHLCV records.", index_symbol, len(records)
        )
        return records
