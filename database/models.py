"""
MarketLens SQLAlchemy 2.0 Database Models — Phase 1 + Phase 2

Tables:
  Phase 1 (raw historical data layer):
    - companies      : NSE-listed company reference data
    - news           : raw financial news articles
    - market_data    : daily OHLCV for equities (FK → companies)
    - market_indices : daily OHLCV for broad market indices

  Phase 2 (event intelligence layer):
    - events         : structured financial events extracted from news
                       (Gemini classification + FinBERT sentiment scores)

All timestamps use DateTime(timezone=True) to prevent look-ahead bias
in downstream financial time-series analysis.
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 declarative models."""
    pass


# ===========================================================================
# Phase 1 — Raw Historical Data Layer
# ===========================================================================

class Company(Base):
    """
    Company metadata and reference data.
    Acts as the parent entity for historical ticker-level market data.
    """
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    market_data: Mapped[List["MarketData"]] = relationship(
        "MarketData",
        back_populates="company",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Company(id={self.id}, ticker='{self.ticker}', name='{self.company_name}')>"


class News(Base):
    """
    Raw historical financial news and press releases.
    Strictly raw unstructured text and publication metadata — no NLP features here.
    """
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    article: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
    )
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # 1:1 relationship to the Event extracted from this article (Phase 2)
    event: Mapped[Optional["Event"]] = relationship(
        "Event",
        back_populates="news",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<News(id={self.id}, source='{self.source}', published_at='{self.published_at}')>"


class MarketData(Base):
    """
    Raw historical OHLCV market pricing data for companies.
    High-frequency time-series table indexed on (ticker, timestamp).
    """
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(
        String(20),
        ForeignKey("companies.ticker", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    open: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    company: Mapped["Company"] = relationship("Company", back_populates="market_data")

    __table_args__ = (
        UniqueConstraint("ticker", "timestamp", name="uq_market_data_ticker_timestamp"),
        Index("ix_market_data_ticker_timestamp", "ticker", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<MarketData(ticker='{self.ticker}', timestamp='{self.timestamp}', close={self.close})>"


class MarketIndex(Base):
    """
    Raw historical OHLCV market data for broader market indices (NIFTY 50, NIFTY Bank, NIFTY IT).
    Indexed on (index_symbol, timestamp).
    """
    __tablename__ = "market_indices"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(100), nullable=False)
    index_symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    open: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("index_symbol", "timestamp", name="uq_market_index_symbol_timestamp"),
        Index("ix_market_index_symbol_timestamp", "index_symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<MarketIndex(symbol='{self.index_symbol}', timestamp='{self.timestamp}', close={self.close})>"


# ===========================================================================
# Phase 2 — Event Intelligence Layer
# ===========================================================================

class Event(Base):
    """
    Structured financial event intelligence extracted from a News article.

    Each News article maps to at most one Event (unique FK on news_id).
    Combines:
      - Gemini 2.5 Flash structured fields: event_type, gemini_sentiment,
        magnitude, surprise, summary
      - FinBERT continuous sentiment probabilities: finbert_positive,
        finbert_negative, finbert_neutral

    ticker is nullable for macro/economy-wide events not tied to a specific company.
    """
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Source article — unique constraint enforces 1:1 with news
    news_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("news.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Nullable: NULL for macro/economy-wide events (RBI policy, GDP data, etc.)
    ticker: Mapped[Optional[str]] = mapped_column(
        String(20),
        ForeignKey("companies.ticker", ondelete="SET NULL"),
        nullable=True,
    )

    # --- Gemini-extracted structured fields -----------------------------------
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    gemini_sentiment: Mapped[str] = mapped_column(String(20), nullable=False)
    magnitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    surprise: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    # --- FinBERT continuous sentiment probabilities ---------------------------
    finbert_positive: Mapped[float] = mapped_column(Float, nullable=False)
    finbert_negative: Mapped[float] = mapped_column(Float, nullable=False)
    finbert_neutral: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    news: Mapped["News"] = relationship("News", back_populates="event")

    __table_args__ = (
        Index("ix_events_ticker_event_type", "ticker", "event_type"),
        Index("ix_events_event_type", "event_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<Event(id={self.id}, news_id={self.news_id}, ticker='{self.ticker}', "
            f"event_type='{self.event_type}', sentiment='{self.gemini_sentiment}')>"
        )
