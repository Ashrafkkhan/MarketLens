/**
 * MarketLens API Client Layer
 * Connects to FastAPI backend (default: http://localhost:8000)
 * Includes graceful mock fallbacks for development & offline resilience.
 */

import {
  AnalyzeResponse,
  ChatResponse,
  EventFeedItem,
  HealthStatus,
  PriceForecastRequest,
  PriceForecastResponse,
} from '@/types/marketlens';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// ---------------------------------------------------------------------------
// Realistic Mock Fallback Data (Dev / Offline Mode)
// ---------------------------------------------------------------------------

export const MOCK_EVENTS: EventFeedItem[] = [
  {
    id: 101,
    news_id: 1001,
    ticker: 'TCS',
    company_name: 'Tata Consultancy Services Ltd',
    headline: 'TCS bags mega $1.2B deal from European retail giant, revenue visibility strengthens',
    body: 'Tata Consultancy Services announced a multi-year digital transformation partnership to optimize cloud infrastructure and supply chain automation.',
    event_type: 'CONTRACT_ORDER',
    sentiment: 'POSITIVE',
    surprise: 'BEAT',
    magnitude: 1200.0,
    summary: 'TCS secures a $1.2B digital transformation deal over 5 years.',
    published_at: new Date(Date.now() - 1000 * 60 * 35).toISOString(),
    created_at: new Date(Date.now() - 1000 * 60 * 35).toISOString(),
  },
  {
    id: 102,
    news_id: 1002,
    ticker: 'INFY',
    company_name: 'Infosys Limited',
    headline: 'Infosys reports 10% jump in profit, beating market consensus on resilient cloud demand',
    body: 'Infosys announced strong Q2 financial results with operating margins expanding by 60 basis points.',
    event_type: 'EARNINGS',
    sentiment: 'POSITIVE',
    surprise: 'BEAT',
    magnitude: 10.0,
    summary: 'Infosys posts 10% YoY net profit growth beating street estimates.',
    published_at: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
    created_at: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
  },
  {
    id: 103,
    news_id: 1003,
    ticker: 'HDFCBANK',
    company_name: 'HDFC Bank Limited',
    headline: 'HDFC Bank quarterly profit disappoints, net interest margin compresses 15 bps',
    body: 'HDFC Bank reported net profit of ₹16,820 crore, impacted by cost of funds and deposit repricing pressures.',
    event_type: 'EARNINGS',
    sentiment: 'NEGATIVE',
    surprise: 'MISS',
    magnitude: 15.0,
    summary: 'HDFC Bank NIM contracts 15 bps, pulling net profits below consensus.',
    published_at: new Date(Date.now() - 1000 * 60 * 240).toISOString(),
    created_at: new Date(Date.now() - 1000 * 60 * 240).toISOString(),
  },
  {
    id: 104,
    news_id: 1004,
    ticker: 'RELIANCE',
    company_name: 'Reliance Industries Limited',
    headline: 'Reliance Retail expands partnership with global brands, adds 250 new store locations',
    body: 'Reliance Retail continued aggressive expansion across Tier-2 and Tier-3 urban markets.',
    event_type: 'CONTRACT_ORDER',
    sentiment: 'POSITIVE',
    surprise: 'IN_LINE',
    magnitude: 250.0,
    summary: 'Reliance Retail scales footprint with 250 new retail touchpoints.',
    published_at: new Date(Date.now() - 1000 * 60 * 380).toISOString(),
    created_at: new Date(Date.now() - 1000 * 60 * 380).toISOString(),
  },
  {
    id: 105,
    news_id: 1005,
    ticker: 'ICICIBANK',
    company_name: 'ICICI Bank Limited',
    headline: 'Reserve Bank of India holds repo rate steady at 6.5%, maintains cautious stance',
    body: 'The Monetary Policy Committee unanimously voted to keep rates unchanged while closely monitoring food inflation.',
    event_type: 'MONETARY_POLICY',
    sentiment: 'NEUTRAL',
    surprise: 'IN_LINE',
    magnitude: 6.5,
    summary: 'RBI MPC holds repo rate at 6.5%, retaining neutral liquidity policy.',
    published_at: new Date(Date.now() - 1000 * 60 * 600).toISOString(),
    created_at: new Date(Date.now() - 1000 * 60 * 600).toISOString(),
  },
];

export const MOCK_ANALYSIS: AnalyzeResponse = {
  headline: 'TCS bags mega $1.2B deal from European retail giant, revenue visibility strengthens',
  body: 'Tata Consultancy Services announced a multi-year digital transformation partnership to optimize cloud infrastructure and supply chain automation.',
  extracted_event: {
    news_id: 1001,
    ticker: 'TCS',
    event_type: 'CONTRACT_ORDER',
    gemini_sentiment: 'POSITIVE',
    magnitude: 1200.0,
    surprise: 'BEAT',
    summary: 'TCS signs a $1.2 billion contract with a European retail giant, strengthening multi-year revenue visibility.',
    finbert_positive: 0.946,
    finbert_negative: 0.013,
    finbert_neutral: 0.041,
  },
  technicals: {
    momentum_5d: -0.0163,
    momentum_20d: -0.1259,
    volatility_20d: 0.013,
    volume_anomaly: 0.87,
    market_return_5d: -0.0235,
  },
  prediction: {
    ticker: 'TCS',
    probability: 0.684,
    signal: 1,
    conviction: 'HIGH CONVICTION BULLISH',
    target_horizon: 'T+3 Trading Days (AR3 vs NIFTY 50)',
    features: {
      momentum_5d: -0.0163,
      momentum_20d: -0.1259,
      volatility_20d: 0.013,
      volume_anomaly: 0.87,
      market_return_5d: -0.0235,
      finbert_positive: 0.946,
      finbert_negative: 0.013,
      finbert_neutral: 0.041,
      evt_contract_order: 1.0,
      surp_beat: 1.0,
      magnitude: 1200.0,
    },
  },
  evidence: {
    candidates_fetched: 20,
    sample_size: 20,
    win_rate_ar3: 0.65,
    median_ar3: 0.0072,
    mean_ar3: 0.0064,
    std_ar3: 0.0086,
    max_gain: 0.0245,
    max_loss: -0.0102,
    top_analogs: [
      {
        rank: 1,
        event_id: 88,
        ticker: 'TCS',
        event_type: 'CONTRACT_ORDER',
        surprise: 'BEAT',
        event_date: '2026-09-23',
        ar_3: 0.0072,
        similarity_score: 0.725,
        summary: 'Tata Consultancy Services renews strategic transformation contract with European partner.',
      },
      {
        rank: 2,
        event_id: 74,
        ticker: 'INFY',
        event_type: 'CONTRACT_ORDER',
        surprise: 'BEAT',
        event_date: '2026-09-18',
        ar_3: 0.0118,
        similarity_score: 0.698,
        summary: 'Infosys inks $850M deal with global telecommunications provider for digital cloud services.',
      },
      {
        rank: 3,
        event_id: 62,
        ticker: 'TCS',
        event_type: 'CONTRACT_ORDER',
        surprise: 'IN_LINE',
        event_date: '2026-08-30',
        ar_3: 0.0045,
        similarity_score: 0.684,
        summary: 'TCS expands partnership with Nordic bank for IT core platform modernizations.',
      },
    ],
  },
  synthesis: {
    source_llm: 'google/gemini-2.5-flash',
    full_memo: `### 1. Executive Thesis\n- **Conviction Rating:** High Conviction Bullish\n- **Statistical Edge:** Model C estimates a **68.4% probability** of achieving a 3-day abnormal return exceeding +0.5% vs NIFTY 50 (Signal: 1).\n- **Core Thesis:** The $1.2B contract win provides strong revenue visibility for TCS over a multi-year horizon. While the broader IT sector has experienced 5-day beta pressure (-1.63%), this event catalyst presents a clear idiosyncratic alpha driver capable of generating positive abnormal decoupling.\n\n### 2. Technical & Event Catalysts\n- **Event Categorization & Surprise:** Classified as **CONTRACT_ORDER** with surprise status **BEAT** and confirmed magnitude of **$1.2B**.\n- **Sentiment Distribution:** FinBERT registers **94.6% positive probability mass**, confirming high institutional sentiment clarity.\n- **Market Backdrop:** Stock 5-day momentum is -1.63% while NIFTY 50 is -2.35%, indicating relative resilience before the announcement.\n\n### 3. Historical Analog Precedents\n- **Empirical Base Rate:** Across 20 retrieved FAISS historical contract wins, the historical win rate is **65.0%** with a median 3-day abnormal return of **+0.72%**.\n- **Key Case Studies:**\n  - [2026-09-23] TCS: Realized AR3 = +0.72% following a renewal agreement.\n  - [2026-09-18] INFY: Realized AR3 = +1.18% upon an $850M telecom award.\n\n### 4. Downside Risks & Invalidation Triggers\n- **Sector Macro:** Severe US/Europe tech budget cutbacks could delay margin realization.\n- **Invalidation Trigger:** A daily closing abnormal return below -0.8% relative to NIFTY 50 on T+1 invalidates the near-term outperformance thesis.`,
    sections: {
      executive_thesis:
        '- Conviction Rating: High Conviction Bullish\n- Statistical Edge: Model C estimates a 68.4% probability of achieving a 3-day abnormal return exceeding +0.5% vs NIFTY 50 (Signal: 1).\n- Core Thesis: The $1.2B contract win provides strong revenue visibility for TCS. While the broader IT sector has experienced 5-day beta pressure, this event catalyst presents a clear idiosyncratic alpha driver.',
      technical_event_drivers:
        '- Event Categorization: CONTRACT_ORDER | Surprise: BEAT | Magnitude: $1.2B.\n- Sentiment Distribution: FinBERT registers 94.6% positive probability mass.\n- Market Backdrop: Stock 5-day momentum is -1.63% while NIFTY 50 is -2.35%.',
      historical_precedent:
        '- Empirical Base Rate: Across 20 retrieved FAISS historical contract wins, the historical win rate is 65.0% with a median 3-day abnormal return of +0.72%.\n- Top Match: [2026-09-23] TCS realized +0.72% AR3.',
      downside_risks:
        '- Macro Risks: Prolonged decision cycles in European enterprise retail.\n- Invalidation Trigger: Daily close under -0.8% vs NIFTY 50 on T+1.',
    },
  },
};

// ---------------------------------------------------------------------------
// API Client Functions
// ---------------------------------------------------------------------------

export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });
    if (!res.ok) throw new Error(`Health HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return {
      status: 'DEGRADED',
      database: false,
      faiss_index: false,
      model_c: false,
      details: { note: 'Backend unreachable. Running in interactive demo mode.' },
    };
  }
}

export async function getRecentEvents(limit: number = 30): Promise<EventFeedItem[]> {
  try {
    const res = await fetch(`${BASE_URL}/events?limit=${limit}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });
    if (!res.ok) throw new Error(`Events HTTP ${res.status}`);
    const data = await res.json();
    if (data.events && data.events.length > 0) {
      return data.events;
    }
    return MOCK_EVENTS;
  } catch (err) {
    console.warn('[MarketLens API] /events call failed, using mock data:', err);
    return MOCK_EVENTS;
  }
}

export async function analyzeEvent(payload: {
  headline: string;
  body?: string;
  ticker?: string;
  top_k?: number;
}): Promise<AnalyzeResponse> {
  const res = await fetch(`${BASE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`Analyze HTTP ${res.status}: ${errBody}`);
  }
  return await res.json();
}

export async function sendChatMessage(
  messages: { role: string; content: string }[],
  context?: any
): Promise<ChatResponse> {
  try {
    const res = await fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, context }),
    });
    if (!res.ok) {
      throw new Error(`Chat HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.warn('[MarketLens API] /chat call failed, using client copilot response:', err);
    const lastMsg = messages[messages.length - 1]?.content.toLowerCase() || '';
    let response = '';

    if (lastMsg.includes('win rate') || lastMsg.includes('historical')) {
      response = `Across the retrieved FAISS historical analogs, the empirical win rate is 65.0% for AR3 > +0.5% with a median abnormal return of +0.72%. Similar large contract wins consistently protected downside.`;
    } else if (lastMsg.includes('volume') || lastMsg.includes('technicals')) {
      response = `Pre-event volume anomaly is currently 0.87x of its 20-day mean. A post-event volume surge above 1.5x on T+1 is typically required for institutional confirmation.`;
    } else if (lastMsg.includes('drawdown') || lastMsg.includes('risk') || lastMsg.includes('worst')) {
      response = `The worst historical 3-day abnormal return among matches was -1.02%. The invalidation trigger is placed at a -0.80% abnormal return closing below NIFTY 50 on T+1.`;
    } else {
      response = `Model C calculates a 68.4% probability of 3-day abnormal return outperformance (> +0.5% vs NIFTY 50). The combination of 94.6% FinBERT positive sentiment and contract value ($1.2B) supports an institutional BUY conviction.`;
    }

    return {
      response,
      source_llm: 'copilot/offline-grounded-analyst',
    };
  }
}

export async function priceForecast(
  payload: PriceForecastRequest
): Promise<PriceForecastResponse> {
  try {
    const res = await fetch(`${BASE_URL}/price-forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(`Price forecast HTTP ${res.status}: ${errBody}`);
    }
    return await res.json();
  } catch (err) {
    console.warn('[MarketLens API] /price-forecast call failed:', err);
    // Offline fallback — no real prices available
    return {
      ticker: payload.ticker,
      model_used: null,
      last_actual_date: null,
      last_actual_close: null,
      forecast_dates: [],
      forecast_prices: [],
      signal: 'Hold',
      confidence_pct: 0,
      price_change_pct: 0,
      model_metrics: {},
      available_models: ['Random Forest', 'XGBoost'],
      error: 'Backend unavailable — price forecast requires live connection.',
    };
  }
}
