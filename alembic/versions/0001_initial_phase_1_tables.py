"""create initial phase 1 tables for MarketLens

Revision ID: 0001_initial_phase_1
Revises: 
Create Date: 2026-09-29 20:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_phase_1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Companies Table
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("sector", sa.String(length=100), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("country", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", name="uq_companies_ticker"),
    )
    op.create_index("ix_companies_ticker", "companies", ["ticker"])
    op.create_index("ix_companies_sector", "companies", ["sector"])

    # 2. News Table
    op.create_table(
        "news",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("article", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url", name="uq_news_url"),
    )
    op.create_index("ix_news_source", "news", ["source"])
    op.create_index("ix_news_published_at", "news", ["published_at"])

    # 3. MarketData Table
    op.create_table(
        "market_data",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("high", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("low", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("close", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["ticker"], ["companies.ticker"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", "timestamp", name="uq_market_data_ticker_timestamp"),
    )
    op.create_index("ix_market_data_ticker_timestamp", "market_data", ["ticker", "timestamp"])

    # 4. MarketIndex Table
    op.create_table(
        "market_indices",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("index_name", sa.String(length=100), nullable=False),
        sa.Column("index_symbol", sa.String(length=50), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("high", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("low", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("close", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("index_symbol", "timestamp", name="uq_market_index_symbol_timestamp"),
    )
    op.create_index("ix_market_index_symbol_timestamp", "market_indices", ["index_symbol", "timestamp"])


def downgrade() -> None:
    op.drop_index("ix_market_index_symbol_timestamp", table_name="market_indices")
    op.drop_table("market_indices")

    op.drop_index("ix_market_data_ticker_timestamp", table_name="market_data")
    op.drop_table("market_data")

    op.drop_index("ix_news_published_at", table_name="news")
    op.drop_index("ix_news_source", table_name="news")
    op.drop_table("news")

    op.drop_index("ix_companies_sector", table_name="companies")
    op.drop_index("ix_companies_ticker", table_name="companies")
    op.drop_table("companies")
