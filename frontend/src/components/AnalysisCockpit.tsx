'use client';

import React, { useState } from 'react';
import { Sparkles } from 'lucide-react';
import { AnalyzeResponse, PriceForecastResponse } from '@/types/marketlens';
import type { WorkspaceView } from '@/components/Sidebar';

interface AnalysisCockpitProps {
  data: AnalyzeResponse | null;
  isLoading: boolean;
  forecastData?: PriceForecastResponse | null;
  view: WorkspaceView;
}

const fmtPct = (val?: number | null, digits: number = 2) => {
  if (val === undefined || val === null || isNaN(val)) return 'N/A';
  return `${val > 0 ? '+' : ''}${(val * 100).toFixed(digits)}%`;
};

const fmtInr = (val?: number | null) =>
  val === undefined || val === null
    ? 'N/A'
    : `₹${val.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const toneFor = (val?: number | null) =>
  val === undefined || val === null ? 'text-zinc-200' : val >= 0 ? 'text-emerald-400' : 'text-rose-400';

const SECTION_TITLES: Partial<Record<WorkspaceView, string>> = {
  analysis: 'Stock Analysis',
  forecast: 'Price Forecast',
  evidence: 'Historical Evidence',
  technical: 'Technical Analysis',
  memo: 'Research Memo',
  features: 'ML Features',
};

function Metric({
  label,
  value,
  tone = 'text-zinc-100',
  hint,
}: {
  label: string;
  value: React.ReactNode;
  tone?: string;
  hint?: string;
}) {
  return (
    <div>
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`mt-1 font-mono text-xl ${tone}`}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-zinc-600">{hint}</div>}
    </div>
  );
}

function Disclosure({
  label,
  open,
  onToggle,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <button onClick={onToggle} className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors">
      {open ? `Hide ${label}` : `View ${label}`}
    </button>
  );
}

export default function AnalysisCockpit({ data, isLoading, forecastData, view }: AnalysisCockpitProps) {
  const [showAnalogs, setShowAnalogs] = useState(false);
  const [showFeatures, setShowFeatures] = useState(false);
  const [showForecastDetail, setShowForecastDetail] = useState(false);

  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-zinc-400">
        <Sparkles className="w-5 h-5 text-cyan-400 animate-pulse mb-4" />
        <div className="text-sm font-medium text-zinc-200">Analyzing event</div>
        <div className="mt-1 text-xs text-zinc-500">
          Extraction, technicals, Model C, historical evidence and memo
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-full flex items-center justify-center p-8 text-sm text-zinc-500">
        Select an event from Market Events, or analyze a custom headline.
      </div>
    );
  }

  const { extracted_event, technicals, prediction, evidence, synthesis } = data;
  const prob = prediction.probability * 100;
  const isBullish = prediction.signal === 1;
  const sectionTitle = SECTION_TITLES[view];

  const body = (() => {
    switch (view) {
      case 'analysis':
        return (
          <div className="space-y-10">
            <div className="grid gap-10 md:grid-cols-2">
              <div>
                <div className="text-xs text-zinc-500">P(3-day abnormal return &gt; +0.5%)</div>
                <div className="mt-1 font-mono text-5xl font-semibold text-zinc-50">{prob.toFixed(1)}%</div>
                <div className="mt-4 h-1.5 w-full rounded-full bg-zinc-800">
                  <div
                    className={`h-1.5 rounded-full ${isBullish ? 'bg-emerald-500' : 'bg-rose-500'}`}
                    style={{ width: `${Math.min(100, Math.max(0, prob))}%` }}
                  />
                </div>
              </div>
              <div className="space-y-3">
                <div className={`text-sm font-medium ${isBullish ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {isBullish ? 'Outperform (signal +1)' : 'Underperform (signal 0)'}
                </div>
                <div className="text-sm text-zinc-300">Rating: {prediction.conviction}</div>
                <div className="text-xs text-zinc-500">Horizon: {prediction.target_horizon}</div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-zinc-300">Sentiment</h3>
              <div className="mt-4 grid grid-cols-3 gap-6">
                <Metric label="Positive" value={`${(extracted_event.finbert_positive * 100).toFixed(1)}%`} tone="text-emerald-400" />
                <Metric label="Negative" value={`${(extracted_event.finbert_negative * 100).toFixed(1)}%`} tone="text-rose-400" />
                <Metric label="Neutral" value={`${(extracted_event.finbert_neutral * 100).toFixed(1)}%`} />
              </div>
            </div>

            {extracted_event.summary && (
              <p className="max-w-2xl text-[15px] leading-7 text-zinc-300">{extracted_event.summary}</p>
            )}
          </div>
        );

      case 'technical':
        return (
          <div className="grid grid-cols-2 gap-x-10 gap-y-8 md:grid-cols-3">
            <Metric label="5-day momentum" value={fmtPct(technicals.momentum_5d)} tone={toneFor(technicals.momentum_5d)} />
            <Metric label="20-day momentum" value={fmtPct(technicals.momentum_20d)} tone={toneFor(technicals.momentum_20d)} />
            <Metric
              label="Volume anomaly"
              value={`${technicals.volume_anomaly.toFixed(2)}x`}
              hint="vs 20-day average"
              tone={technicals.volume_anomaly > 1.2 ? 'text-cyan-300' : 'text-zinc-100'}
            />
            <Metric label="20-day volatility" value={`${(technicals.volatility_20d * 100).toFixed(2)}%`} />
            <Metric label="NIFTY 50 5-day" value={fmtPct(technicals.market_return_5d)} tone={toneFor(technicals.market_return_5d)} hint="benchmark drift" />
          </div>
        );

      case 'forecast': {
        if (!forecastData) {
          return (
            <p className="text-sm text-zinc-500">
              No forecast loaded for this event. Ask in Analyst Chat, for example &ldquo;Predict {extracted_event.ticker || 'this stock'} price&rdquo;.
            </p>
          );
        }
        if (forecastData.error || forecastData.forecast_prices.length === 0) {
          return <p className="text-sm text-zinc-500">{forecastData.error || 'No forecast available.'}</p>;
        }
        const last = forecastData.forecast_prices[forecastData.forecast_prices.length - 1];
        const change = forecastData.price_change_pct;
        return (
          <div className="space-y-10">
            <div className="grid grid-cols-2 gap-x-10 gap-y-8 md:grid-cols-4">
              <Metric label="Current price" value={fmtInr(forecastData.last_actual_close)} hint={forecastData.last_actual_date || undefined} />
              <Metric label={`Predicted (+${forecastData.forecast_prices.length} days)`} value={fmtInr(last)} tone="text-cyan-300" />
              <Metric label="Expected change" value={`${change >= 0 ? '+' : ''}${change.toFixed(2)}%`} tone={change >= 0 ? 'text-emerald-400' : 'text-rose-400'} />
              <Metric label="Model" value={forecastData.model_used || 'N/A'} hint={`${forecastData.confidence_pct}% confidence · ${forecastData.signal}`} />
            </div>
            <div>
              <Disclosure label="detailed forecast" open={showForecastDetail} onToggle={() => setShowForecastDetail((v) => !v)} />
              {showForecastDetail && (
                <div className="mt-4 max-w-md divide-y divide-zinc-800/60">
                  {forecastData.forecast_dates.map((d, i) => (
                    <div key={d} className="flex justify-between py-2 text-sm">
                      <span className="text-zinc-500">{d}</span>
                      <span className="font-mono text-zinc-200">{fmtInr(forecastData.forecast_prices[i])}</span>
                    </div>
                  ))}
                  {Object.keys(forecastData.model_metrics).length > 0 && (
                    <div className="pt-3 text-xs text-zinc-500">
                      {Object.entries(forecastData.model_metrics)
                        .map(([k, v]) => `${k} ${v}`)
                        .join(' · ')}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      }

      case 'evidence':
        return (
          <div className="space-y-10">
            <div className="grid grid-cols-2 gap-x-10 gap-y-8 md:grid-cols-4">
              <Metric label="Win rate (AR3 > +0.5%)" value={fmtPct(evidence.win_rate_ar3, 1)} tone={(evidence.win_rate_ar3 || 0) >= 0.5 ? 'text-emerald-400' : 'text-rose-400'} />
              <Metric label="Median realized AR3" value={fmtPct(evidence.median_ar3)} tone={toneFor(evidence.median_ar3)} />
              <Metric label="Maximum gain" value={fmtPct(evidence.max_gain)} tone="text-emerald-400" />
              <Metric label="Maximum loss" value={fmtPct(evidence.max_loss)} tone="text-rose-400" />
            </div>
            <div>
              <p className="text-sm text-zinc-400">
                {evidence.top_analogs.length} analogs found across {evidence.sample_size} sampled events.
              </p>
              <div className="mt-2">
                <Disclosure label="historical precedents" open={showAnalogs} onToggle={() => setShowAnalogs((v) => !v)} />
              </div>
              {showAnalogs && (
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full border-collapse text-left text-sm">
                    <thead>
                      <tr className="border-b border-zinc-800 text-xs text-zinc-500">
                        <th className="py-2 pr-4 font-normal">Rank</th>
                        <th className="py-2 pr-4 font-normal">Similarity</th>
                        <th className="py-2 pr-4 font-normal">Ticker</th>
                        <th className="py-2 pr-4 font-normal">Date</th>
                        <th className="py-2 pr-4 font-normal">AR3</th>
                        <th className="py-2 font-normal">Summary</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/60">
                      {evidence.top_analogs.map((a) => (
                        <tr key={a.event_id}>
                          <td className="py-2.5 pr-4 font-mono text-zinc-400">#{a.rank}</td>
                          <td className="py-2.5 pr-4 font-mono text-zinc-300">{a.similarity_score.toFixed(3)}</td>
                          <td className="py-2.5 pr-4 font-mono text-zinc-200">{a.ticker || 'MACRO'}</td>
                          <td className="py-2.5 pr-4 text-zinc-400">{a.event_date}</td>
                          <td className={`py-2.5 pr-4 font-mono ${toneFor(a.ar_3)}`}>{fmtPct(a.ar_3)}</td>
                          <td className="max-w-md py-2.5 text-zinc-300">{a.summary}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        );

      case 'memo':
        return (
          <div className="space-y-8 max-w-3xl">
            <div className="text-xs text-zinc-500">Generated by {synthesis.source_llm}</div>
            {synthesis.sections.executive_thesis ? (
              [
                ['Executive thesis', synthesis.sections.executive_thesis],
                ['Technical and event drivers', synthesis.sections.technical_event_drivers],
                ['Historical analog rationale', synthesis.sections.historical_precedent],
                ['Downside risks and invalidation', synthesis.sections.downside_risks],
              ].map(([title, text]) => (
                <section key={title}>
                  <h3 className="text-sm font-semibold text-zinc-200">{title}</h3>
                  <div className="mt-2 whitespace-pre-line text-[15px] leading-7 text-zinc-300">{text}</div>
                </section>
              ))
            ) : (
              <div className="whitespace-pre-line text-[15px] leading-7 text-zinc-300">{synthesis.full_memo}</div>
            )}
          </div>
        );

      case 'features': {
        const entries = Object.entries(prediction.features);
        return (
          <div className="space-y-4">
            <p className="text-sm text-zinc-400">{entries.length} input features fed into Model C.</p>
            <Disclosure label={`${entries.length} features`} open={showFeatures} onToggle={() => setShowFeatures((v) => !v)} />
            {showFeatures && (
              <div className="grid grid-cols-1 gap-x-10 gap-y-2 pt-2 sm:grid-cols-2">
                {entries.map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-zinc-800/60 py-2 text-sm">
                    <span className="text-zinc-500">{k}</span>
                    <span className="font-mono text-zinc-200">
                      {typeof v === 'number' ? (v % 1 !== 0 ? v.toFixed(4) : v) : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      }

      default:
        return null;
    }
  })();

  if (!sectionTitle) return null;

  return (
    <div className="px-8 py-8">
      <h2 className="text-lg font-semibold text-zinc-100">{sectionTitle}</h2>
      <div className="mt-8">{body}</div>
    </div>
  );
}
