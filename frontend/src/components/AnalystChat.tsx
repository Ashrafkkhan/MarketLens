'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, Trash2 } from 'lucide-react';
import { AnalyzeResponse, ChatMessage, PriceForecastResponse } from '@/types/marketlens';
import { sendChatMessage } from '@/lib/api';

type ChatItem = ChatMessage & {
  forecast?: PriceForecastResponse;
  training?: boolean;
};

interface AnalystChatProps {
  activeContext: AnalyzeResponse | null;
  forecastData?: PriceForecastResponse | null;
  onViewForecast: () => void;
}

const PRICE_INTENT = /price|predict|forecast|invest|target|buy|sell/i;

const INITIAL_MESSAGE: ChatItem = {
  id: 'init-1',
  role: 'assistant',
  content:
    'Ask about the active event, Model C probability, historical analogs, or a price forecast for any stock you name.',
};

const humanize = (value?: string | null) =>
  value ? value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase()) : '';

const fmtInr = (value?: number | null) =>
  value === undefined || value === null
    ? 'N/A'
    : `₹${value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

function ForecastCard({
  forecast,
  onView,
}: {
  forecast: PriceForecastResponse;
  onView: () => void;
}) {
  const prices = forecast.forecast_prices;
  const predicted = prices[prices.length - 1];
  const change = forecast.price_change_pct;
  return (
    <div className="max-w-md rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] tracking-[0.18em] text-zinc-500">PRICE FORECAST</span>
        <span className="font-mono text-xs text-zinc-300">{forecast.ticker}</span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-6">
        <div>
          <div className="text-xs text-zinc-500">Current price</div>
          <div className="mt-1 font-mono text-xl text-zinc-100">{fmtInr(forecast.last_actual_close)}</div>
        </div>
        <div>
          <div className="text-xs text-zinc-500">Predicted price</div>
          <div className="mt-1 font-mono text-xl text-cyan-300">{fmtInr(predicted)}</div>
        </div>
      </div>
      <div className="mt-4 flex items-center justify-between text-sm">
        <span className="text-zinc-500">
          Expected change{' '}
          <span className={`font-mono ${change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {change >= 0 ? '+' : ''}
            {change.toFixed(2)}%
          </span>
        </span>
        <span className="text-xs text-zinc-500">
          {forecast.model_used} · {prices.length}-day forecast
        </span>
      </div>
      <button onClick={onView} className="mt-4 text-xs text-cyan-400 hover:text-cyan-300 transition-colors">
        View forecast details
      </button>
    </div>
  );
}

export default function AnalystChat({ activeContext, forecastData, onViewForecast }: AnalystChatProps) {
  const [messages, setMessages] = useState<ChatItem[]>([INITIAL_MESSAGE]);
  const [inputValue, setInputValue] = useState('');
  const [isSending, setIsSending] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  const ticker = activeContext?.extracted_event.ticker || null;
  const meta = activeContext
    ? [humanize(activeContext.extracted_event.event_type), activeContext.extracted_event.surprise && activeContext.extracted_event.surprise !== 'NOT_APPLICABLE' ? `${humanize(activeContext.extracted_event.surprise)} surprise` : null]
        .filter(Boolean)
        .join(' · ')
    : '';

  const suggestions = [
    `Predict ${ticker || 'this stock'} price`,
    'Explain Model C signal',
    'Show historical analogs',
    'Why is this event bullish?',
  ];

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  const handleSendMessage = async (textToSend: string) => {
    if (!textToSend.trim() || isSending) return;

    const userMsg: ChatItem = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: textToSend.trim(),
    };

    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInputValue('');
    setIsSending(true);

    try {
      const apiMsgs = newHistory.map((m) => ({ role: m.role, content: m.content }));
      const context = activeContext
        ? { ...activeContext, price_forecast: forecastData || undefined }
        : forecastData
        ? { price_forecast: forecastData }
        : null;
      const res = await sendChatMessage(apiMsgs, context);

      const refersToForecastTicker =
        !!forecastData &&
        (new RegExp(`\\b${forecastData.ticker}\\b`, 'i').test(textToSend) ||
          /\b(this|it|here|current)\b|this stock/i.test(textToSend));
      const showForecast =
        PRICE_INTENT.test(textToSend) &&
        refersToForecastTicker &&
        !!forecastData &&
        !forecastData.error &&
        forecastData.forecast_prices.length > 0;

      const botMsg: ChatItem = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: res.response,
        source_llm: res.source_llm,
        training: /train/i.test(res.response) && !/failed/i.test(res.response),
        forecast: showForecast ? forecastData! : undefined,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: 'Could not reach the analyst service. Check the backend connection and try again.',
          source_llm: 'error',
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleClear = () => setMessages([{ ...INITIAL_MESSAGE, id: `init-${Date.now()}` }]);

  return (
    <div className="flex flex-col h-full min-h-0 bg-zinc-950">
      <header className="flex items-start justify-between gap-4 px-8 py-4 border-b border-zinc-800/50">
        <div className="min-w-0">
          <div className="text-[11px] tracking-[0.18em] text-zinc-500">ANALYST COPILOT</div>
          <div className="mt-1 flex items-baseline gap-2 text-sm text-zinc-300 min-w-0">
            <span className="font-mono font-semibold text-zinc-100">{ticker || 'No active stock'}</span>
            {meta && <span className="truncate text-zinc-500">{meta}</span>}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-[11px] text-zinc-500">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Grounded
          </span>
          <button
            onClick={handleClear}
            title="Clear chat"
            className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/60 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-6 py-8 space-y-8">
          {messages.map((msg) =>
            msg.role === 'user' ? (
              <div key={msg.id} className="flex justify-end">
                <div className="max-w-[80%] whitespace-pre-line rounded-2xl bg-zinc-800/70 px-4 py-2.5 text-[15px] leading-relaxed text-zinc-100">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div key={msg.id} className="space-y-3">
                <div className="text-[11px] tracking-wide text-zinc-500">Analyst</div>
                {msg.training ? (
                  <div className="max-w-xl rounded-xl bg-zinc-900/60 px-4 py-3">
                    <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
                      <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                      Preparing price forecast
                    </div>
                    <p className="mt-1.5 text-sm leading-6 text-zinc-400">{msg.content}</p>
                  </div>
                ) : (
                  <div className="whitespace-pre-line text-[15px] leading-7 text-zinc-200">{msg.content}</div>
                )}
                {msg.forecast && <ForecastCard forecast={msg.forecast} onView={onViewForecast} />}
                {msg.source_llm && (
                  <div className="text-[11px] text-zinc-600">Grounded via {msg.source_llm}</div>
                )}
              </div>
            )
          )}

          {isSending && <div className="text-sm text-zinc-500 animate-pulse">Analyst is working…</div>}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-zinc-800/50 bg-zinc-950">
        <div className="mx-auto max-w-3xl px-6 pt-3 pb-5">
          <div className="mb-3 flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <button
                key={s}
                onClick={() => handleSendMessage(s)}
                disabled={isSending}
                className="rounded-full border border-zinc-800 px-3 py-1 text-xs text-zinc-400 hover:border-zinc-700 hover:text-zinc-100 disabled:opacity-40 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage(inputValue);
            }}
            className="flex items-end gap-3 rounded-2xl border border-zinc-800 bg-zinc-900/70 px-4 py-3 focus-within:border-zinc-700 transition-colors"
          >
            <textarea
              rows={1}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(inputValue);
                }
              }}
              placeholder={`Ask MarketLens about ${ticker || 'a stock'}, forecasts, events…`}
              className="max-h-40 flex-1 resize-none bg-transparent text-[15px] leading-6 text-zinc-100 placeholder-zinc-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isSending}
              aria-label="Send"
              className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-cyan-500 text-zinc-950 hover:bg-cyan-400 disabled:bg-zinc-800 disabled:text-zinc-600 transition-colors"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
