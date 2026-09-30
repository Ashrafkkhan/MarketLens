"""
Phase 2 Alembic migration: create `events` table.

Revision ID: 0002_add_events_table
Revises: 0001_initial_phase_1
Create Date: 2026-09-30

NOTE: This migration was already applied to the database on 2026-09-30.
      It is kept here as the source of truth for the schema.
      Running `alembic upgrade head` will be a no-op if already at this revision.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
revision: str = "0002_add_events_table"
down_revision: Union[str, None] = "0001_initial_phase_1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
# ---------------------------------------------------------------------------


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        # FK to news — unique constraint enforces 1 event per article
        sa.Column("news_id", sa.Integer(), nullable=False),
        # Nullable FK to companies.ticker — NULL for macro/economy-wide events
        sa.Column("ticker", sa.String(length=20), nullable=True),
        # Gemini-extracted structured fields
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("gemini_sentiment", sa.String(length=20), nullable=False),
        sa.Column("magnitude", sa.Float(), nullable=True),
        sa.Column("surprise", sa.String(length=20), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        # FinBERT continuous sentiment probabilities
        sa.Column("finbert_positive", sa.Float(), nullable=False),
        sa.Column("finbert_negative", sa.Float(), nullable=False),
        sa.Column("finbert_neutral", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["news_id"], ["news.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["ticker"], ["companies.ticker"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("news_id", name="uq_events_news_id"),
    )
    # Composite index: query events by company + event category
    op.create_index("ix_events_ticker_event_type", "events", ["ticker", "event_type"])
    # Single-column index: query all events of a given type across companies
    op.create_index("ix_events_event_type", "events", ["event_type"])


def downgrade() -> None:
    op.drop_index("ix_events_event_type", table_name="events")
    op.drop_index("ix_events_ticker_event_type", table_name="events")
    op.drop_table("events")
