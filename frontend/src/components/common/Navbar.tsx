'use client';

import React, { useEffect, useState } from 'react';
import { fetchHealth } from '@/lib/api';
import { Cpu, Zap, Wifi, WifiOff, RefreshCw } from 'lucide-react';

export function Navbar() {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const checkHealth = async () => {
    try {
      setLoading(true);
      const data = await fetchHealth();
      setHealth(data);
    } catch (e) {
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 border-b border-border bg-[#0b101d]/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-border text-xs">
          <span className="text-slate-400 font-medium">Domain:</span>
          <span className="text-emerald-400 font-mono">Precision Swine, Canine & Aquaculture</span>
        </div>
      </div>

      <div className="flex items-center gap-4 text-xs">
        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-border">
          {health ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-emerald-300 font-medium">Backend Online</span>
              <span className="text-slate-500 font-mono">v{health.version || '0.5.0'}</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-rose-400" />
              <span className="text-rose-300 font-medium">Connecting...</span>
            </>
          )}
          <button
            onClick={checkHealth}
            disabled={loading}
            className="text-slate-500 hover:text-slate-300 ml-1"
            title="Refresh status"
          >
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Hardware Acceleration Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-border text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">Device:</span>
          <span className="font-mono text-cyan-300">{health?.device || 'CPU'}</span>
        </div>
      </div>
    </header>
  );
}
