'use client';

import React, { useEffect, useState } from 'react';
import { Camera, Play, Square, Activity, Wifi, Plus, AlertCircle, Sparkles } from 'lucide-react';

export default function LiveStreamsPage() {
  const [streams, setStreams] = useState<any[]>([]);
  const [cameraId, setCameraId] = useState('barn_cam_01');
  const [rtspUrl, setRtspUrl] = useState('rtsp://localhost:8554/live');
  const [species, setSpecies] = useState('pig');
  const [targetFps, setTargetFps] = useState(15.0);
  const [loading, setLoading] = useState(false);

  const fetchActiveStreams = async () => {
    try {
      const res = await fetch('/api/backend/stream/active');
      if (res.ok) {
        const data = await res.json();
        setStreams(data.active_streams || []);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchActiveStreams();
    const interval = setInterval(fetchActiveStreams, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleStartStream = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await fetch('/api/backend/stream/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: cameraId,
          rtsp_url: rtspUrl,
          species,
          target_fps: targetFps,
        }),
      });
      await fetchActiveStreams();
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleStopStream = async (camId: string) => {
    try {
      await fetch(`/api/backend/stream/stop/${camId}`, { method: 'POST' });
      await fetchActiveStreams();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Camera className="w-8 h-8 text-emerald-400" />
            Live RTSP Camera Streams &amp; Edge Telemetry
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Connect live IP cameras, broadcast real-time behavioral alerts over WebSockets, and log events to MongoDB.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Stream Starter Card (Left 1 Col) */}
        <div className="bg-[#111827] border border-border rounded-3xl p-6 shadow-xl space-y-4">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Plus className="w-4 h-4 text-emerald-400" />
            Connect RTSP Camera Feed
          </h2>

          <form onSubmit={handleStartStream} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">Camera ID</label>
              <input
                type="text"
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                className="w-full bg-slate-900 border border-border rounded-xl px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-emerald-500"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">RTSP Stream URI</label>
              <input
                type="text"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
                className="w-full bg-slate-900 border border-border rounded-xl px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-emerald-500"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">Species Taxonomy</label>
              <select
                value={species}
                onChange={(e) => setSpecies(e.target.value)}
                className="w-full bg-slate-900 border border-border rounded-xl px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
              >
                <option value="pig">🐖 Swine (Sus scrofa)</option>
                <option value="dog">🐕 Canine (Canis lupus)</option>
                <option value="redclaw">🦀 Redclaw (Cherax)</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 px-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-bold text-xs transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
            >
              <Play className="w-3.5 h-3.5 fill-slate-950" />
              Start Live Ingestion
            </button>
          </form>
        </div>

        {/* Active Streams Monitor (Right 2 Cols) */}
        <div className="lg:col-span-2 bg-[#111827] border border-border rounded-3xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              Active Live Camera Workers ({streams.length})
            </h2>
          </div>

          <div className="space-y-4">
            {streams.map((st: any) => (
              <div
                key={st.camera_id}
                className="p-5 bg-slate-900 border border-border rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-4"
              >
                <div className="space-y-1 font-mono">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
                    <span className="font-bold text-white text-sm">{st.camera_id}</span>
                    <span className="text-[10px] text-slate-500">({st.species})</span>
                  </div>
                  <div className="text-xs text-slate-400 truncate max-w-md">{st.rtsp_url}</div>
                  <div className="text-[11px] text-emerald-400 flex items-center gap-3 pt-1">
                    <span>Processed: {st.processed_frames} frames</span>
                    <span>FPS: {st.current_fps || 15}</span>
                  </div>
                </div>

                <button
                  onClick={() => handleStopStream(st.camera_id)}
                  className="px-4 py-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
                >
                  <Square className="w-3.5 h-3.5" /> Stop Stream
                </button>
              </div>
            ))}

            {streams.length === 0 && (
              <div className="p-12 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-2xl">
                No active camera streams. Connect an RTSP feed on the left to start live monitoring.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
