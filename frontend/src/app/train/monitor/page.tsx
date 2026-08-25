'use client';

import React, { Suspense, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import {
  Activity,
  ArrowLeft,
  CheckCircle2,
  AlertCircle,
  Download,
  Boxes,
  TrendingUp,
  Cpu,
  Layers,
  Sparkles,
  RefreshCw,
} from 'lucide-react';
import { fetchTrainingJobStatus } from '@/lib/api';
import { createTrainingWebSocket } from '@/lib/ws';
import { EpochMetric, TrainingJob } from '@/types/training';

function TrainingMonitorContent() {
  const searchParams = useSearchParams();
  const jobId = searchParams.get('job_id') || '';

  const [job, setJob] = useState<TrainingJob | null>(null);
  const [history, setHistory] = useState<EpochMetric[]>([]);
  const [status, setStatus] = useState<string>('CONNECTING');
  const [wsConnected, setWsConnected] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // 1. Initial REST Fetch
  useEffect(() => {
    if (!jobId) return;
    fetchTrainingJobStatus(jobId)
      .then((data) => {
        setJob(data);
        setHistory(data.history || []);
        setStatus(data.status);
      })
      .catch((e) => console.error(e));
  }, [jobId]);

  // 2. Real-Time WebSocket Telemetry Subscription
  useEffect(() => {
    if (!jobId) return;

    const cleanup = createTrainingWebSocket(
      jobId,
      (msg) => {
        setWsConnected(true);
        if (msg.type === 'INITIAL_SNAPSHOT') {
          setJob(msg.data);
          setHistory(msg.data.history || []);
          setStatus(msg.data.status);
        } else if (msg.type === 'EPOCH_PROGRESS') {
          const newMetric: EpochMetric = msg.data;
          setHistory((prev) => [...prev, newMetric]);
          setJob((prev) =>
            prev
              ? {
                  ...prev,
                  current_epoch: newMetric.epoch,
                  progress_pct: (newMetric.epoch / newMetric.total_epochs) * 100,
                  latest_metrics: newMetric,
                }
              : null
          );
        } else if (msg.type === 'STATUS_UPDATE') {
          setStatus(msg.status);
          if (msg.status === 'COMPLETED') {
            setJob((prev) => (prev ? { ...prev, status: 'COMPLETED', report: msg.data } : null));
          }
        }
      },
      (err) => {
        setWsConnected(false);
      },
      () => {
        setWsConnected(false);
      }
    );

    return cleanup;
  }, [jobId]);

  // 3. Render Dynamic Loss Curves Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = (canvas.width = canvas.parentElement?.clientWidth || 700);
    const height = (canvas.height = 260);
    const padding = 40;

    ctx.clearRect(0, 0, width, height);

    // Draw Grid Lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let y = padding; y <= height - padding; y += 40) {
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    const maxLoss = Math.max(
      4.0,
      ...history.map((h) => Math.max(h.box_loss || 0, h.cls_loss || 0, h.dfl_loss || 0))
    );

    const getX = (idx: number) =>
      padding + (idx / Math.max(1, (job?.epochs || 10) - 1)) * (width - padding * 2);
    const getY = (val: number) =>
      height - padding - (val / maxLoss) * (height - padding * 2);

    // Helper to draw loss line
    const drawLine = (key: 'box_loss' | 'cls_loss' | 'dfl_loss', color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      history.forEach((h, idx) => {
        const x = getX(idx);
        const y = getY(h[key] || 0);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Draw points
      history.forEach((h, idx) => {
        const x = getX(idx);
        const y = getY(h[key] || 0);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    // Draw Box Loss (Emerald), Cls Loss (Amber), DFL Loss (Cyan)
    drawLine('box_loss', '#10b981');
    drawLine('cls_loss', '#f59e0b');
    drawLine('dfl_loss', '#38bdf8');
  }, [history, job]);

  const latest = history[history.length - 1];

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-5">
        <div className="flex items-center gap-3">
          <Link
            href="/train"
            className="p-2 rounded-xl bg-slate-900 border border-border text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold text-white tracking-tight">
                {job?.experiment_name || jobId || 'Training Monitor'}
              </h1>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold ${
                  status === 'COMPLETED'
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    : status === 'RUNNING'
                    ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30 animate-pulse'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {status}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5 font-mono">
              Job ID: {jobId} | Species: {job?.species} | Base: {job?.base_model}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-border text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                wsConnected ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'
              }`}
            />
            <span className="text-slate-400">
              {wsConnected ? 'Live WebSocket Stream' : 'Connecting WebSocket...'}
            </span>
          </div>

          {status === 'COMPLETED' && (
            <Link
              href="/models/test"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs transition-colors shadow-lg shadow-emerald-500/20"
            >
              <Boxes className="w-3.5 h-3.5" />
              Test Fine-Tuned Model
            </Link>
          )}
        </div>
      </div>

      {/* Progress Bar & Key Metric Gauges */}
      <div className="bg-[#111827] border border-border rounded-3xl p-6 shadow-xl space-y-5">
        <div className="flex items-center justify-between text-xs font-bold text-slate-300">
          <span className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-amber-400" />
            Training Progress: Epoch {job?.current_epoch || history.length} / {job?.epochs || 10}
          </span>
          <span className="font-mono text-amber-400 text-sm">
            {(job?.progress_pct || 0).toFixed(1)}%
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-amber-500 to-emerald-400 transition-all duration-500"
            style={{ width: `${Math.min(100, job?.progress_pct || 0)}%` }}
          />
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
          <div className="p-4 bg-slate-900/80 border border-border rounded-2xl">
            <div className="text-xs text-slate-400 mb-1">Validation mAP@50</div>
            <div className="text-2xl font-bold text-emerald-400 font-mono">
              {latest?.map50 ? `${(latest.map50 * 100).toFixed(1)}%` : '—'}
            </div>
            <div className="text-[10px] text-slate-500 mt-1">Detection Precision</div>
          </div>

          <div className="p-4 bg-slate-900/80 border border-border rounded-2xl">
            <div className="text-xs text-slate-400 mb-1">Validation Recall</div>
            <div className="text-2xl font-bold text-teal-400 font-mono">
              {latest?.recall ? `${(latest.recall * 100).toFixed(1)}%` : '—'}
            </div>
            <div className="text-[10px] text-slate-500 mt-1">Target Coverage</div>
          </div>

          <div className="p-4 bg-slate-900/80 border border-border rounded-2xl">
            <div className="text-xs text-slate-400 mb-1">Classification Loss</div>
            <div className="text-2xl font-bold text-amber-400 font-mono">
              {latest?.cls_loss ? latest.cls_loss.toFixed(3) : '—'}
            </div>
            <div className="text-[10px] text-slate-500 mt-1">Convergence Index</div>
          </div>

          <div className="p-4 bg-slate-900/80 border border-border rounded-2xl">
            <div className="text-xs text-slate-400 mb-1">Bounding Box Loss</div>
            <div className="text-2xl font-bold text-sky-400 font-mono">
              {latest?.box_loss ? latest.box_loss.toFixed(3) : '—'}
            </div>
            <div className="text-[10px] text-slate-500 mt-1">Spatial Regression</div>
          </div>
        </div>
      </div>

      {/* Loss Curves Canvas Chart & Checkpoints */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Loss Curves Canvas (Left 2 Cols) */}
        <div className="lg:col-span-2 bg-[#111827] border border-border rounded-3xl p-6 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-white">Real-Time Loss Convergence Curves</h2>
              <p className="text-xs text-slate-400">Streamed live per epoch over WebSocket telemetry</p>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="flex items-center gap-1 text-emerald-400">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Box
              </span>
              <span className="flex items-center gap-1 text-amber-400">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400" /> Class
              </span>
              <span className="flex items-center gap-1 text-sky-400">
                <span className="w-2.5 h-2.5 rounded-full bg-sky-400" /> DFL
              </span>
            </div>
          </div>

          <div className="relative w-full h-[260px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 p-2">
            <canvas ref={canvasRef} className="w-full h-full" />
            {history.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500 gap-2">
                <RefreshCw className="w-4 h-4 animate-spin text-amber-400" />
                Awaiting first epoch telemetry stream...
              </div>
            )}
          </div>
        </div>

        {/* Checkpoint Export Cards (Right 1 Col) */}
        <div className="bg-[#111827] border border-border rounded-3xl p-6 space-y-4 shadow-xl flex flex-col justify-between">
          <div className="space-y-3">
            <h2 className="text-lg font-bold text-white">Model Artifacts</h2>

            <div className="p-3.5 bg-slate-900 border border-border rounded-2xl space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-emerald-400 font-mono">best.pt</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-400 font-mono">
                  PyTorch
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Optimal validation checkpoint with highest mAP@50 score.
              </p>
              <div className="text-[10px] text-slate-500 font-mono truncate">
                models/trained/{job?.experiment_name}/weights/best.pt
              </div>
            </div>

            <div className="p-3.5 bg-slate-900 border border-border rounded-2xl space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-cyan-400 font-mono">best.onnx</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-cyan-500/10 text-cyan-400 font-mono">
                  ONNX Edge
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Quantized ONNX model for high-speed edge microservice inference.
              </p>
            </div>
          </div>

          <div className="pt-2">
            <Link
              href="/models"
              className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 font-semibold text-xs transition-colors flex items-center justify-center gap-2 border border-border"
            >
              <Boxes className="w-4 h-4" />
              View in Model Registry
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function TrainingMonitorPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-400 text-sm">Loading Training Monitor...</div>}>
      <TrainingMonitorContent />
    </Suspense>
  );
}

