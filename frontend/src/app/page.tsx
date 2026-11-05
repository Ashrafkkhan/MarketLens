'use client';

import React, { useState, useEffect } from 'react';
import LiveEventFeed from '@/components/LiveEventFeed';
import AnalysisCockpit from '@/components/AnalysisCockpit';
import AnalystChat from '@/components/AnalystChat';
import Sidebar, { WorkspaceView } from '@/components/Sidebar';
import {
  EventFeedItem,
  AnalyzeResponse,
  HealthStatus,
  PriceForecastResponse,
} from '@/types/marketlens';
import {
  getRecentEvents,
  analyzeEvent,
  checkHealth,
  priceForecast,
  MOCK_EVENTS,
} from '@/lib/api';

const humanize = (value?: string | null) =>
  value ? value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase()) : '';

const DEFAULT_EVENT = {
  headline: 'Infosys Stock Update: Shares Slip Amid IT Sector Headwinds - LatestLY',
  ticker: 'INFY',
};
const DEFAULT_ANALYSIS_CACHE_KEY = 'marketlens:default-analysis:v1';

function StatusDot({ label, ok }: { label: string; ok: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
      {label}
    </span>
  );
}

export default function DashboardPage() {
  const [events, setEvents] = useState<EventFeedItem[]>(MOCK_EVENTS);
  const [selectedEventId, setSelectedEventId] = useState<number | null>(101);
  const [analysisData, setAnalysisData] = useState<AnalyzeResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [forecastData, setForecastData] = useState<PriceForecastResponse | null>(null);
  const [isFeedLoading, setIsFeedLoading] = useState(false);
  const [isAnalysisLoading, setIsAnalysisLoading] = useState(false);
  const [health, setHealth] = useState<HealthStatus>({
    status: 'CONNECTING',
    database: false,
    faiss_index: false,
    model_c: false,
    details: {},
  });
  const [view, setView] = useState<WorkspaceView>('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    if (window.innerWidth < 768) setSidebarCollapsed(true);
    loadHealth();
    loadEvents();
    loadDefaultAnalysis();
  }, []);

  // Analyze the default headline on startup. The result is cached in localStorage
  // so reloads show it instantly instead of re-running the slow pipeline.
  const loadDefaultAnalysis = async () => {
    try {
      const cached = localStorage.getItem(DEFAULT_ANALYSIS_CACHE_KEY);
      if (cached) {
        const parsed: AnalyzeResponse = JSON.parse(cached);
        setAnalysisData(parsed);
        loadForecastFor(parsed.extracted_event.ticker);
        return;
      }
    } catch {
      // storage unavailable or corrupt cache: fall through to a fresh analysis
    }

    setIsAnalysisLoading(true);
    setAnalysisError(null);
    try {
      const res = await analyzeEvent({
        headline: DEFAULT_EVENT.headline,
        ticker: DEFAULT_EVENT.ticker,
      });
      setAnalysisData(res);
      loadForecastFor(res.extracted_event.ticker);
      try {
        localStorage.setItem(DEFAULT_ANALYSIS_CACHE_KEY, JSON.stringify(res));
      } catch {
        // ignore storage errors
      }
    } catch (err) {
      failAnalysis(err);
    } finally {
      setIsAnalysisLoading(false);
    }
  };

  const loadHealth = async () => {
    const res = await checkHealth();
    setHealth(res);
  };

  const loadEvents = async () => {
    setIsFeedLoading(true);
    try {
      const data = await getRecentEvents(40);
      setEvents(data);
      if (data.length > 0 && !selectedEventId) {
        handleSelectEvent(data[0]);
      }
    } catch (err) {
      console.error('Failed loading events:', err);
    } finally {
      setIsFeedLoading(false);
    }
  };

  const loadForecastFor = (ticker: string | null | undefined) => {
    if (!ticker) {
      setForecastData(null);
      return;
    }
    priceForecast({ ticker, horizon_days: 7 })
      .then(setForecastData)
      .catch(() => setForecastData(null));
  };

  const failAnalysis = (err: unknown) => {
    console.error('Event analysis failed:', err);
    setAnalysisData(null);
    setForecastData(null);
    setAnalysisError(err instanceof Error ? err.message : 'Analysis failed');
  };

  const handleSelectEvent = async (eventItem: EventFeedItem) => {
    setSelectedEventId(eventItem.id);
    setIsAnalysisLoading(true);
    setAnalysisError(null);
    try {
      const res = await analyzeEvent({
        headline: eventItem.headline,
        body: eventItem.body || '',
        ticker: eventItem.ticker || undefined,
      });
      setAnalysisData(res);
      loadForecastFor(res.extracted_event.ticker);
    } catch (err) {
      failAnalysis(err);
    } finally {
      setIsAnalysisLoading(false);
    }
  };

  const handleCustomAnalyze = async (headline: string, ticker: string) => {
    setIsAnalysisLoading(true);
    setAnalysisError(null);
    try {
      const res = await analyzeEvent({
        headline,
        ticker: ticker || undefined,
      });
      setAnalysisData(res);
      loadForecastFor(res.extracted_event.ticker);

      const customItem: EventFeedItem = {
        id: Date.now(),
        news_id: Date.now(),
        ticker: res.extracted_event.ticker || ticker || 'MACRO',
        headline: res.headline,
        body: res.body || '',
        event_type: res.extracted_event.event_type,
        sentiment: res.extracted_event.gemini_sentiment,
        surprise: res.extracted_event.surprise,
        magnitude: res.extracted_event.magnitude,
        summary: res.extracted_event.summary,
        created_at: new Date().toISOString(),
      };
      setEvents((prev) => [customItem, ...prev]);
      setSelectedEventId(customItem.id);
    } catch (err) {
      failAnalysis(err);
    } finally {
      setIsAnalysisLoading(false);
    }
  };

  const selectEventAndOpenChat = (eventItem: EventFeedItem) => {
    setView('chat');
    handleSelectEvent(eventItem);
  };

  const customAndOpenChat = (headline: string, ticker: string) => {
    setView('chat');
    handleCustomAnalyze(headline, ticker);
  };

  const active = analysisData?.extracted_event;
  const meta = active
    ? [
        humanize(active.event_type),
        active.surprise && active.surprise !== 'NOT_APPLICABLE' ? `${humanize(active.surprise)} surprise` : null,
        active.magnitude != null ? `Magnitude ${active.magnitude}` : null,
      ]
        .filter(Boolean)
        .join(' · ')
    : '';

  const backendLive = health.status === 'HEALTHY' || health.database;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-zinc-950 text-zinc-100">
      <Sidebar
        view={view}
        onSelect={setView}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((v) => !v)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-11 flex-shrink-0 items-center justify-between border-b border-zinc-800/50 px-6 text-[11px] text-zinc-500">
          <div className="flex items-center gap-2">
            <span>NIFTY 50</span>
            <span className="font-mono text-zinc-200">24,850.30</span>
            <span className="font-mono text-emerald-400">+0.42%</span>
          </div>
          <div className="hidden items-center gap-5 md:flex">
            <StatusDot label="Model C" ok={health.model_c} />
            <StatusDot label="FAISS" ok={health.faiss_index} />
            <StatusDot label="Qwen (local)" ok />
            <StatusDot label="Backend" ok={backendLive} />
          </div>
        </header>

        <div className="flex-shrink-0 border-b border-zinc-800/50 px-8 py-5">
          {analysisData && active ? (
            <div className="flex items-start justify-between gap-6">
              <div className="min-w-0">
                <div className="font-mono text-2xl font-semibold tracking-tight text-zinc-50">
                  {active.ticker || 'MACRO'}
                </div>
                <p className="mt-1.5 text-lg leading-snug text-zinc-200">{analysisData.headline}</p>
                {meta && <p className="mt-2 text-xs text-zinc-500">{meta}</p>}
              </div>
              <button
                onClick={() => setView('events')}
                className="flex-shrink-0 rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:border-zinc-700 hover:text-zinc-200"
              >
                Change event
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-between">
              <p className={`text-sm ${analysisError ? 'text-rose-400' : 'text-zinc-500'}`}>
                {isAnalysisLoading
                  ? 'Analyzing event…'
                  : analysisError
                  ? `Analysis failed: ${analysisError}`
                  : 'No event selected.'}
              </p>
              <button
                onClick={() => setView('events')}
                className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:border-zinc-700 hover:text-zinc-200"
              >
                Choose an event
              </button>
            </div>
          )}
        </div>

        <main className="min-h-0 flex-1 overflow-hidden">
          <div className={`h-full ${view === 'chat' ? '' : 'hidden'}`}>
            <AnalystChat
              activeContext={analysisData}
              forecastData={forecastData}
              onViewForecast={() => setView('forecast')}
            />
          </div>

          {view === 'events' && (
            <div className="mx-auto h-full max-w-3xl">
              <LiveEventFeed
                events={events}
                selectedEventId={selectedEventId}
                onSelectEvent={selectEventAndOpenChat}
                onAnalyzeCustom={customAndOpenChat}
                isLoading={isFeedLoading}
                onRefresh={loadEvents}
              />
            </div>
          )}

          {view !== 'chat' && view !== 'events' && (
            <div className="h-full overflow-y-auto">
              <AnalysisCockpit
                data={analysisData}
                isLoading={isAnalysisLoading}
                forecastData={forecastData}
                view={view}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
