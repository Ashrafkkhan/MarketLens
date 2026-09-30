"""
Abstract Base Classes for MarketLens data providers.

All concrete providers must subclass these ABCs and implement
the required methods. This enforces a uniform interface across
different data sources (yfinance, Bloomberg, etc.).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class MarketRecord:
    """
    Validated, normalised OHLCV record ready for database insertion.
    All timestamps must be timezone-aware UTC datetimes.
    """
    ticker: str
    timestamp: datetime          # UTC, timezone-aware
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"MarketRecord.timestamp must be timezone-aware (ticker={self.ticker})"
            )


@dataclass
class IndexRecord:
    """
    Validated, normalised OHLCV record for a broad market index.
    All timestamps must be timezone-aware UTC datetimes.
    """
    index_name: str
    index_symbol: str
    timestamp: datetime          # UTC, timezone-aware
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"IndexRecord.timestamp must be timezone-aware "
                f"(symbol={self.index_symbol})"
            )


@dataclass
class NewsRecord:
    """
    Validated, normalised news article record ready for database insertion.
    All timestamps must be timezone-aware UTC datetimes.
    """
    headline: str
    url: str
    source: str
    published_at: datetime       # UTC, timezone-aware
    article: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            raise ValueError(
                f"NewsRecord.published_at must be timezone-aware (url={self.url})"
            )
        if not self.headline.strip():
            raise ValueError(f"NewsRecord.headline cannot be empty (url={self.url})")
        if not self.url.strip():
            raise ValueError("NewsRecord.url cannot be empty")


class BaseMarketProvider(ABC):
    """
    Abstract base class for market/index OHLCV data providers.

    Concrete implementations must handle:
    - Fetching raw data from the upstream source
    - Applying all business-logic validation
    - Returning clean, timezone-aware MarketRecord / IndexRecord lists
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[MarketRecord]:
        """
        Fetch and validate daily OHLCV data for a single ticker symbol.

        Args:
            ticker:  The ticker symbol (e.g. 'INFY.NS').
            start:   Inclusive start date.
            end:     Inclusive end date.

        Returns:
            List of validated MarketRecord objects with UTC timestamps.
            Returns an empty list if no data is available.
        """
        ...

    @abstractmethod
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
            index_symbol:  The index symbol (e.g. '^NSEI').
            index_name:    Human-readable name (e.g. 'NIFTY 50').
            start:         Inclusive start date.
            end:           Inclusive end date.

        Returns:
            List of validated IndexRecord objects with UTC timestamps.
            Returns an empty list if no data is available.
        """
        ...


class BaseNewsProvider(ABC):
    """
    Abstract base class for financial news providers.

    Concrete implementations must handle:
    - Fetching raw articles from the upstream source
    - Extracting article body text (best-effort)
    - Applying validation rules (skip missing timestamps, etc.)
    - Returning clean, timezone-aware NewsRecord lists
    """

    @abstractmethod
    def fetch_news(
        self,
        query: str,
        start: date,
        end: date,
    ) -> list[NewsRecord]:
        """
        Fetch news articles matching a query within a date range.

        Args:
            query:  Search query string (e.g. ticker + exchange suffix).
            start:  Inclusive start date (used to filter published_at).
            end:    Inclusive end date (used to filter published_at).

        Returns:
            List of validated NewsRecord objects with UTC timestamps.
            Returns an empty list if no articles are found.
        """
        ...
