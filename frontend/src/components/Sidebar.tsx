'use client';

import React from 'react';
import {
  Activity,
  Cpu,
  FileText,
  Gauge,
  Layers,
  LineChart,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Radio,
} from 'lucide-react';

export type WorkspaceView =
  | 'chat'
  | 'events'
  | 'analysis'
  | 'forecast'
  | 'evidence'
  | 'technical'
  | 'memo'
  | 'features';

const NAV_ITEMS: { key: WorkspaceView; label: string; icon: React.ElementType }[] = [
  { key: 'chat', label: 'Analyst Chat', icon: MessageSquare },
  { key: 'events', label: 'Market Events', icon: Radio },
  { key: 'analysis', label: 'Stock Analysis', icon: Gauge },
  { key: 'forecast', label: 'Price Forecast', icon: LineChart },
  { key: 'evidence', label: 'Historical Evidence', icon: Layers },
  { key: 'technical', label: 'Technical Analysis', icon: Activity },
  { key: 'memo', label: 'Research Memo', icon: FileText },
  { key: 'features', label: 'ML Features', icon: Cpu },
];

interface SidebarProps {
  view: WorkspaceView;
  onSelect: (view: WorkspaceView) => void;
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({ view, onSelect, collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={`${
        collapsed ? 'w-16' : 'w-60'
      } flex-shrink-0 h-full flex flex-col bg-zinc-900/50 border-r border-zinc-800/60 transition-[width] duration-200`}
    >
      <div className="flex items-start justify-between gap-2 px-4 pt-5 pb-6">
        {!collapsed && (
          <div className="min-w-0">
            <div className="text-sm font-semibold tracking-wide text-zinc-100">MARKETLENS</div>
            <div className="mt-0.5 text-[10px] tracking-[0.18em] text-zinc-500">AI INVESTMENT ANALYST</div>
          </div>
        )}
        <button
          onClick={onToggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/60 transition-colors"
        >
          {collapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
        </button>
      </div>

      <nav className="flex-1 px-2 space-y-0.5">
        {NAV_ITEMS.map(({ key, label, icon: Icon }) => {
          const active = view === key;
          return (
            <button
              key={key}
              onClick={() => onSelect(key)}
              title={collapsed ? label : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                collapsed ? 'justify-center' : ''
              } ${
                active
                  ? 'bg-zinc-800/80 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40'
              }`}
            >
              <Icon className={`w-4 h-4 flex-shrink-0 ${active ? 'text-cyan-400' : ''}`} />
              {!collapsed && <span className="truncate">{label}</span>}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
