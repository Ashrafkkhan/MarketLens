/**
 * MarketLens TypeScript Type Definitions
 * Maps 1-to-1 with backend FastAPI Pydantic models.
 */

export type SentimentType = 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
export type SurpriseType = 'BEAT' | 'MISS' | 'IN_LINE' | 'NOT_APPLICABLE';
export type EventCategory =
  | 'EARNINGS'
  | 'CONTRACT_ORDER'
  | 'MONETARY_POLICY'
  | 'REGULATION'
  | 'MERGER_ACQUISITION'
  | 'MANAGEMENT_CHANGE'
  | 'MACRO_DATA'
  | 'OTHER';

export interface ExtractedEvent {
  news_id?: number;
  ticker?: string | null;
  event_type: string;
  gemini_sentiment: string;
  magnitude?: number | null;
  surprise?: string | null;
  summary: string;
  finbert_positive: number;
  finbert_negative: number;
  finbert_neutral: number;
}

export interface Technicals {
  momentum_5d: number;
  momentum_20d: number;
  volatility_20d: number;
  volume_anomaly: number;
  market_return_5d: number;
}

export interface PredictResult {
  ticker?: string | null;
  probability: number;
  signal: number;
  conviction: string;
  target_horizon: string;
  features: Record<string, number>;
}

export interface AnalogItem {
  rank: number;
  event_id: number;
  ticker?: string | null;
  event_type: string;
  surprise?: string | null;
  event_date: string;
  ar_3?: number | null;
  similarity_score: number;
  summary: string;
}

export interface EvidenceResult {
  candidates_fetched: number;
  sample_size: number;
  win_rate_ar3?: number | null;
  median_ar3?: number | null;
  mean_ar3?: number | null;
  std_ar3?: number | null;
  max_gain?: number | null;
  max_loss?: number | null;
  top_analogs: AnalogItem[];
}

export interface SynthesisSections {
  executive_thesis?: string;
  technical_event_drivers?: string;
  historical_precedent?: string;
  downside_risks?: string;
}

export interface SynthesisResult {
  source_llm: string;
  full_memo: string;
  sections: SynthesisSections;
}

export interface AnalyzeResponse {
  headline: string;
  body?: string;
  extracted_event: ExtractedEvent;
  technicals: Technicals;
  prediction: PredictResult;
  evidence: EvidenceResult;
  synthesis: SynthesisResult;
}

export interface EventFeedItem {
  id: number;
  news_id: number;
  ticker?: string | null;
  company_name?: string | null;
  headline: string;
  body?: string | null;
  event_type: string;
  sentiment: string;
  surprise?: string | null;
  magnitude?: number | null;
  summary: string;
  published_at?: string | null;
  created_at?: string | null;
}

export interface EventFeedResponse {
  events: EventFeedItem[];
  total: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  source_llm?: string;
}

export interface ChatResponse {
  response: string;
  source_llm: string;
}

export interface HealthStatus {
  status: string;
  database: boolean;
  faiss_index: boolean;
  model_c: boolean;
  details: Record<string, any>;
}

// ---------------------------------------------------------------------------
// Price Forecast (Stock Price Prediction ML Pipeline)
// ---------------------------------------------------------------------------

export interface PriceForecastRequest {
  ticker: string;
  horizon_days?: number;
  model_name?: string | null;
}

export interface PriceForecastResponse {
  ticker: string;
  model_used: string | null;
  last_actual_date: string | null;
  last_actual_close: number | null;
  forecast_dates: string[];
  forecast_prices: number[];
  signal: 'Buy' | 'Sell' | 'Hold';
  confidence_pct: number;
  price_change_pct: number;
  model_metrics: Record<string, number>;
  available_models: string[];
  error: string | null;
}
