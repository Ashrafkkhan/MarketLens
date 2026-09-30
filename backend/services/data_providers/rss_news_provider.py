"""
Google News RSS Provider for MarketLens.

Fetches financial news articles using the Google News RSS feed and
extracts article body text via trafilatura (with newspaper3k as fallback).

Validation rules applied:
  - Skip entries with missing or unparseable publication timestamps.
  - Skip entries with empty headlines or URLs.
  - All timestamps converted to UTC before returning.
  - Article body extraction is best-effort; None is acceptable.

RSS URL pattern:
  https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en
"""
import logging
import time
from datetime import date, datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus

import feedparser

from .base import BaseNewsProvider, NewsRecord

logger = logging.getLogger(__name__)

# Seconds to sleep between requests to Google News RSS
DEFAULT_SLEEP_SECONDS = 1.5

# Maximum number of articles to process per query to avoid runaway fetching
MAX_ARTICLES_PER_QUERY = 100

# Timeout in seconds for article body HTTP fetches
ARTICLE_FETCH_TIMEOUT = 10


def _rss_url(query: str, days: int) -> str:
    """Build the Google News RSS search URL for a given query and lookback window."""
    encoded = quote_plus(query)
    return (
        f"https://news.google.com/rss/search"
        f"?q={encoded}+when:{days}d"
        f"&hl=en-IN&gl=IN&ceid=IN:en"
    )


def _parse_rss_datetime(raw_date: str) -> Optional[datetime]:
    """
    Parse an RFC-2822 date string from RSS into a UTC-aware datetime.
    Returns None if parsing fails.
    """
    if not raw_date:
        return None
    try:
        dt = parsedate_to_datetime(raw_date)
        return dt.astimezone(timezone.utc)
    except Exception:
        try:
            # Some feeds use ISO 8601 format
            dt = datetime.fromisoformat(raw_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None


def _extract_article_body(url: str) -> Optional[str]:
    """
    Attempt to extract the full article body text from a URL.

    Strategy:
      1. Try trafilatura (faster, more accurate)
      2. Fall back to newspaper3k Article if trafilatura returns nothing
      3. Return None on any failure (missing article body is acceptable)
    """
    # --- Attempt 1: trafilatura -------------------------------------------
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and text.strip():
                return text.strip()
    except Exception as exc:
        logger.debug("trafilatura failed for %s: %s", url, exc)

    # --- Attempt 2: newspaper3k ------------------------------------------
    try:
        from newspaper import Article  # type: ignore[import]

        article = Article(url)
        article.download()
        article.parse()
        if article.text and article.text.strip():
            return article.text.strip()
    except Exception as exc:
        logger.debug("newspaper3k failed for %s: %s", url, exc)

    return None


def _date_range_to_days(start: date, end: date) -> int:
    """
    Convert a [start, end] date range into a 'days back from end' integer
    suitable for the Google News RSS 'when:{N}d' parameter.
    Minimum 1, maximum 30 (Google News RSS limit for the when: filter).
    """
    delta = (end - start).days + 1          # +1 to be inclusive
    return max(1, min(delta, 30))


class RSSNewsProvider(BaseNewsProvider):
    """
    Concrete news provider backed by the Google News RSS feed.

    Fetches up to MAX_ARTICLES_PER_QUERY articles per call.  Article body
    extraction is performed using trafilatura or newspaper3k.

    Usage::

        provider = RSSNewsProvider(sleep_seconds=1.5, fetch_article_body=True)
        records  = provider.fetch_news("INFY NSE", date(2024, 6, 1), date(2024, 6, 30))
    """

    def __init__(
        self,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        fetch_article_body: bool = True,
    ) -> None:
        """
        Args:
            sleep_seconds:       Seconds to sleep between HTTP requests.
            fetch_article_body:  Whether to scrape full article text.
                                 Set False to only ingest headlines + metadata.
        """
        self.sleep_seconds = sleep_seconds
        self.fetch_article_body = fetch_article_body

    # ------------------------------------------------------------------
    # BaseNewsProvider interface
    # ------------------------------------------------------------------

    def fetch_news(
        self,
        query: str,
        start: date,
        end: date,
    ) -> list[NewsRecord]:
        """
        Fetch news articles from Google News RSS matching *query*
        within the [start, end] date range.

        The Google News RSS 'when:{N}d' filter counts back from today,
        so we compute the required lookback window and then post-filter
        by published_at against [start, end].

        Args:
            query:   Search string (e.g. 'INFY NSE' or 'Infosys NSE').
            start:   Inclusive start date filter.
            end:     Inclusive end date filter.

        Returns:
            List of validated NewsRecord objects with UTC timestamps.
        """
        days = _date_range_to_days(start, end)
        url = _rss_url(query, days)

        logger.info(
            "Fetching Google News RSS for query=%r, days=%d (url=%s)",
            query,
            days,
            url,
        )

        try:
            feed = feedparser.parse(url)
        except Exception as exc:
            logger.error("feedparser failed for query=%r: %s", query, exc)
            return []
        finally:
            time.sleep(self.sleep_seconds)

        entries = feed.entries[:MAX_ARTICLES_PER_QUERY]
        logger.info("Google News returned %d entries for query=%r.", len(entries), query)

        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=timezone.utc)

        records: list[NewsRecord] = []
        skipped_no_date = 0
        skipped_out_of_range = 0
        skipped_bad_data = 0

        for entry in entries:
            # --- 1. Parse publication date -----------------------------------
            raw_date = entry.get("published") or entry.get("updated") or ""
            published_at = _parse_rss_datetime(raw_date)

            if published_at is None:
                logger.debug(
                    "Skipping entry (no parseable date): %s",
                    entry.get("link", "N/A"),
                )
                skipped_no_date += 1
                continue

            # --- 2. Filter by date range -------------------------------------
            if not (start_dt <= published_at <= end_dt):
                skipped_out_of_range += 1
                continue

            # --- 3. Extract headline and URL ---------------------------------
            headline: str = entry.get("title", "").strip()
            article_url: str = entry.get("link", "").strip()

            if not headline or not article_url:
                logger.debug("Skipping entry with empty headline or URL.")
                skipped_bad_data += 1
                continue

            # --- 4. Source name from feed source or domain ------------------
            source: str = (
                entry.get("source", {}).get("title", "")
                or feed.feed.get("title", "Google News")
            ).strip()
            if len(source) > 100:
                source = source[:100]

            # --- 5. Optional article body extraction -------------------------
            article_body: Optional[str] = None
            if self.fetch_article_body:
                article_body = _extract_article_body(article_url)
                time.sleep(self.sleep_seconds)

            # --- 6. Build and validate NewsRecord ---------------------------
            try:
                record = NewsRecord(
                    headline=headline,
                    url=article_url,
                    source=source,
                    published_at=published_at,
                    article=article_body,
                )
                records.append(record)
            except ValueError as exc:
                logger.warning("Skipping invalid NewsRecord: %s", exc)
                skipped_bad_data += 1

        logger.info(
            "Query=%r: %d valid records, %d skipped (no_date=%d, out_of_range=%d, bad_data=%d).",
            query,
            len(records),
            skipped_no_date + skipped_out_of_range + skipped_bad_data,
            skipped_no_date,
            skipped_out_of_range,
            skipped_bad_data,
        )

        return records
