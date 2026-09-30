"""
Pydantic schemas for Gemini structured event extraction.

Used as the `response_schema` in `GenerateContentConfig` so Gemini
returns a guaranteed-valid JSON object that is automatically validated
against this model — no hand-rolled JSON parsing needed.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FinancialEventExtraction(BaseModel):
    """
    Structured financial event extracted from a news headline/article
    within an Indian market (NSE/BSE/RBI/SEBI) context.

    All fields map 1:1 to columns in the `events` table except `sentiment`,
    which is stored as `gemini_sentiment` to distinguish it from FinBERT scores.
    """

    ticker: Optional[str] = Field(
        default=None,
        description=(
            "The primary NSE stock ticker affected, WITHOUT the '.NS' suffix "
            "(e.g., 'INFY', 'TCS', 'RELIANCE'). "
            "Set to null if this is a general macro, monetary policy, or "
            "economy-wide event not tied to a specific listed company."
        ),
    )
    event_type: Literal[
        "EARNINGS",
        "MONETARY_POLICY",
        "REGULATION",
        "CONTRACT_ORDER",
        "MERGER_ACQUISITION",
        "MANAGEMENT_CHANGE",
        "MACRO_DATA",
        "OTHER",
    ] = Field(
        description=(
            "Primary financial event category. "
            "EARNINGS: quarterly/annual results, revenue, profit. "
            "MONETARY_POLICY: RBI rate decisions, CRR, SLR changes. "
            "REGULATION: SEBI, MCA, government policy affecting listed companies. "
            "CONTRACT_ORDER: new deal wins, order books. "
            "MERGER_ACQUISITION: M&A, stake sales, demergers. "
            "MANAGEMENT_CHANGE: CEO/CFO/board-level changes. "
            "MACRO_DATA: GDP, inflation, IIP, PMI, trade balance. "
            "OTHER: anything else."
        )
    )
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description=(
            "Market sentiment of this event for the company or general market. "
            "POSITIVE: likely to push price or market up. "
            "NEGATIVE: likely to push price or market down. "
            "NEUTRAL: informational, no clear directional bias."
        )
    )
    magnitude: Optional[float] = Field(
        default=None,
        description=(
            "Numeric scale or size of the event where applicable. "
            "Examples: 50 for a 50 bps rate cut, 15 for 15% revenue growth, "
            "5000 for a Rs 5,000 crore order. Null if no clear numeric magnitude."
        ),
    )
    surprise: Literal["BEAT", "MISS", "IN_LINE", "NOT_APPLICABLE"] = Field(
        default="NOT_APPLICABLE",
        description=(
            "Surprise factor relative to analyst consensus or prior expectations. "
            "BEAT: result/announcement exceeded expectations. "
            "MISS: fell short of expectations. "
            "IN_LINE: met expectations. "
            "NOT_APPLICABLE: no consensus expectation to compare against."
        ),
    )
    summary: str = Field(
        description=(
            "A single concise sentence (max 120 characters) stating the core "
            "financial event that occurred. "
            "Example: 'Infosys Q2 FY25 net profit of Rs 6,506 crore beats estimates by 3%.'"
        )
    )
