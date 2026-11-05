'use strict';
'use client';

import React, { useState } from 'react';
import {
  Search,
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  Sparkles,
  Clock,
  Filter,
  RefreshCw,
  PlusCircle,
  Building2,
} from 'lucide-react';
import { EventFeedItem, SentimentType } from '@/types/marketlens';

interface LiveEventFeedProps {
  events: EventFeedItem[];
  selectedEventId: number | null;
  onSelectEvent: (event: EventFeedItem) => void;
  onAnalyzeCustom: (headline: string, ticker: string) => void;
  isLoading: boolean;
  onRefresh: () => void;
}

export default function LiveEventFeed({
  events,
  selectedEventId,
  onSelectEvent,
  onAnalyzeCustom,
  isLoading,
  onRefresh,
}: LiveEventFeedProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [customHeadline, setCustomHeadline] = useState('');
  const [customTicker, setCustomTicker] = useState('');
  const [showCustomModal, setShowCustomModal] = useState(false);

  // Filter events
  const filteredEvents = events.filter((evt) => {
    const matchesSearch =
      (evt.ticker && evt.ticker.toLowerCase().includes(searchQuery.toLowerCase())) ||
      evt.headline.toLowerCase().includes(searchQuery.toLowerCase()) ||
      evt.event_type.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesCategory =
      selectedCategory === 'ALL' || evt.event_type.toUpperCase() === selectedCategory;

    return matchesSearch && matchesCategory;
  });

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customHeadline.trim()) return;
    onAnalyzeCustom(customHeadline.trim(), customTicker.trim().toUpperCase());
    setCustomHeadline('');
    setCustomTicker('');
    setShowCustomModal(false);
  };

  const getSentimentPill = (sentiment: string) => {
    const s = sentiment.toUpperCase();
    if (s === 'POSITIVE') {
      return (
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <TrendingUp className="w-2.5 h-2.5" />
          POS
        </span>
      );
    }
    if (s === 'NEGATIVE') {
      return (
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
          <TrendingDown className="w-2.5 h-2.5" />
          NEG
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-zinc-700/30 text-zinc-400 border border-zinc-700/40">
        <Minus className="w-2.5 h-2.5" />
        NEU
      </span>
    );
  };

  const formatRelativeTime = (isoString?: string | null) => {
    if (!isoString) return 'Just now';
    const diffMs = Date.now() - new Date(isoString).getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.floor(diffHours / 24)}d ago`;
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950 border-r border-zinc-800/80 text-zinc-100 select-none">
      {/* Header */}
      <div className="p-3 border-b border-zinc-800/80 bg-zinc-900/60 backdrop-blur-md">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <h2 className="text-xs font-bold uppercase tracking-wider text-zinc-300 font-mono">
              Live Ingestion Feed
            </h2>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
              {filteredEvents.length}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setShowCustomModal(!showCustomModal)}
              title="Evaluate Custom Headline"
              className="p-1 rounded text-cyan-400 hover:bg-cyan-500/10 border border-cyan-500/20 transition-all text-xs flex items-center gap-1 font-mono"
            >
              <PlusCircle className="w-3.5 h-3.5" />
              <span>Analyze</span>
            </button>
            <button
              onClick={onRefresh}
              disabled={isLoading}
              title="Refresh Event Feed"
              className="p-1 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>
        </div>

        {/* Quick Search */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-zinc-500" />
          <input
            type="text"
            placeholder="Filter ticker, event type, or headline..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-xs bg-zinc-900/90 border border-zinc-800 rounded text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/30 transition-all"
          />
        </div>

        {/* Category Filter Chips */}
        <div className="flex items-center gap-1 overflow-x-auto no-scrollbar mt-2 pt-1 text-[11px] font-mono">
          {['ALL', 'EARNINGS', 'CONTRACT_ORDER', 'MONETARY_POLICY', 'REGULATION'].map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-2 py-0.5 rounded whitespace-nowrap text-[10px] transition-all ${
                selectedCategory === cat
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold'
                  : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
              }`}
            >
              {cat.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Custom Analysis Bar Drawer */}
      {showCustomModal && (
        <form
          onSubmit={handleCustomSubmit}
          className="p-3 bg-zinc-900/95 border-b border-cyan-500/30 animate-in fade-in slide-in-from-top-2 duration-150"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold text-cyan-400 font-mono flex items-center gap-1">
              <Sparkles className="w-3 h-3" /> Custom Arbitrary Event Analysis
            </span>
            <button
              type="button"
              onClick={() => setShowCustomModal(false)}
              className="text-zinc-500 hover:text-zinc-300 text-xs"
            >
              ✕
            </button>
          </div>
          <div className="space-y-2">
            <input
              type="text"
              placeholder="Headline (e.g. Infosys wins $500M cloud migration deal)"
              value={customHeadline}
              onChange={(e) => setCustomHeadline(e.target.value)}
              className="w-full px-2.5 py-1.5 text-xs bg-zinc-950 border border-zinc-700 rounded text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-cyan-400"
              required
            />
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Ticker (e.g. INFY, TCS)"
                value={customTicker}
                onChange={(e) => setCustomTicker(e.target.value)}
                className="w-1/2 px-2.5 py-1 text-xs bg-zinc-950 border border-zinc-700 rounded text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-cyan-400 font-mono uppercase"
              />
              <button
                type="submit"
                className="w-1/2 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-xs font-semibold font-mono transition-colors py-1 flex items-center justify-center gap-1"
              >
                <Zap className="w-3 h-3" /> Evaluate Alpha
              </button>
            </div>
          </div>
        </form>
      )}

      {/* Events List */}
      <div className="flex-1 overflow-y-auto divide-y divide-zinc-900/70 p-1.5 space-y-1">
        {filteredEvents.length === 0 ? (
          <div className="p-8 text-center text-zinc-500 text-xs font-mono">
            No events match the search filter.
          </div>
        ) : (
          filteredEvents.map((evt) => {
            const isSelected = selectedEventId === evt.id;
            return (
              <div
                key={evt.id}
                onClick={() => onSelectEvent(evt)}
                className={`p-2.5 rounded-md cursor-pointer transition-all duration-150 border ${
                  isSelected
                    ? 'bg-zinc-800/80 border-cyan-500/60 shadow-lg shadow-cyan-950/20'
                    : 'bg-zinc-900/30 hover:bg-zinc-900/70 border-zinc-800/50 hover:border-zinc-700'
                }`}
              >
                {/* Meta Row */}
                <div className="flex items-center justify-between gap-1 mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="px-1.5 py-0.5 rounded font-mono text-[11px] font-bold bg-cyan-950/60 text-cyan-300 border border-cyan-800/40 tracking-wider">
                      {evt.ticker || 'MACRO'}
                    </span>
                    <span className="text-[10px] font-mono text-zinc-400 px-1 py-0.5 rounded bg-zinc-800/60 border border-zinc-800">
                      {evt.event_type}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    {getSentimentPill(evt.sentiment)}
                    <span className="text-[10px] font-mono text-zinc-500 flex items-center gap-0.5">
                      <Clock className="w-2.5 h-2.5" />
                      {formatRelativeTime(evt.published_at || evt.created_at)}
                    </span>
                  </div>
                </div>

                {/* Headline snippet */}
                <p className="text-xs text-zinc-200 font-medium leading-snug line-clamp-2">
                  {evt.headline}
                </p>

                {/* Surprise / Magnitude badges */}
                <div className="flex items-center gap-2 mt-2 pt-1 border-t border-zinc-800/40 text-[10px] text-zinc-400 font-mono">
                  {evt.surprise && evt.surprise !== 'NOT_APPLICABLE' && (
                    <span
                      className={`font-semibold ${
                        evt.surprise === 'BEAT'
                          ? 'text-emerald-400'
                          : evt.surprise === 'MISS'
                          ? 'text-rose-400'
                          : 'text-amber-400'
                      }`}
                    >
                      SURPRISE: {evt.surprise}
                    </span>
                  )}
                  {evt.magnitude !== null && evt.magnitude !== undefined && (
                    <span className="text-zinc-400">MAG: {evt.magnitude}</span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
