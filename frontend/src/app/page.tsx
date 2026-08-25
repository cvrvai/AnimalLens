'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Database,
  PenTool,
  Flame,
  Boxes,
  Camera,
  Activity,
  ArrowRight,
  CheckCircle2,
  TrendingUp,
  Cpu,
  Layers,
  Sparkles,
} from 'lucide-react';
import { fetchDatasets, fetchHealth, fetchSpecies, fetchTrainingJobs } from '@/lib/api';

export default function DashboardPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [species, setSpecies] = useState<any[]>([]);
  const [jobs, setJobs] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [ds, sp, jb, hl] = await Promise.allSettled([
          fetchDatasets(),
          fetchSpecies(),
          fetchTrainingJobs(),
          fetchHealth(),
        ]);
        if (ds.status === 'fulfilled') setDatasets(ds.value);
        if (sp.status === 'fulfilled') setSpecies(sp.value);
        if (jb.status === 'fulfilled') setJobs(jb.value);
        if (hl.status === 'fulfilled') setHealth(hl.value);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const totalFrames = datasets.reduce((acc, d) => acc + (d.total_images || 0), 0);

  return (
    <div className="space-y-8">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-slate-900 via-emerald-950/40 to-slate-900 border border-emerald-500/20 p-8 shadow-2xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold uppercase tracking-wider mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            Animal Behavior Intelligence Studio v0.5
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight mb-3">
            Multi-Species Computer Vision &amp; Biological Reasoning Hub
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed mb-6">
            Manage video datasets, extract keyframes, visually verify AI pseudo-labels, fine-tune YOLOv8 models in 1 click, and stream real-time behavioral telemetry with Ollama clinical reasoning.
          </p>

          <div className="flex flex-wrap gap-3">
            <Link
              href="/datasets"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-sm transition-all shadow-lg shadow-emerald-500/20"
            >
              <Database className="w-4 h-4" />
              Upload &amp; Build Dataset
            </Link>
            <Link
              href="/train"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-sm border border-slate-700 transition-all"
            >
              <Flame className="w-4 h-4 text-amber-400" />
              Launch Model Training
            </Link>
            <Link
              href="/models/test"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-sm border border-slate-700 transition-all"
            >
              <Boxes className="w-4 h-4 text-cyan-400" />
              Live Visual Tester
            </Link>
          </div>
        </div>

        {/* Ambient Decorative Gradient */}
        <div className="absolute right-0 top-0 -mt-12 -mr-12 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#111827] border border-border p-5 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Available Datasets</span>
            <Database className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">{datasets.length}</div>
          <div className="text-xs text-slate-500">{totalFrames} annotated frames total</div>
        </div>

        <div className="bg-[#111827] border border-border p-5 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Supported Species</span>
            <Activity className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">{species.length || 3}</div>
          <div className="text-xs text-slate-500">Swine, Canine, Redclaw Crayfish</div>
        </div>

        <div className="bg-[#111827] border border-border p-5 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Training Jobs</span>
            <Flame className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">{jobs.length}</div>
          <div className="text-xs text-emerald-400">
            {jobs.filter((j) => j.status === 'COMPLETED').length} Successful Checkpoints
          </div>
        </div>

        <div className="bg-[#111827] border border-border p-5 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Hardware Backend</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">{health?.device || 'CPU'}</div>
          <div className="text-xs text-cyan-400">Sub-15ms YOLOv8 Inference</div>
        </div>
      </div>

      {/* 2-Column Section: Active Workspaces & Recent Runs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Dataset Studio Quick Access */}
        <div className="lg:col-span-2 bg-[#111827] border border-border rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-white">Dataset Ingestion &amp; Keyframe Studio</h2>
              <p className="text-xs text-slate-400">Slices raw farm video into YOLO train/validation partitions</p>
            </div>
            <Link href="/datasets" className="text-xs font-semibold text-emerald-400 hover:underline inline-flex items-center gap-1">
              View All <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {datasets.slice(0, 4).map((d) => (
              <Link
                key={d.id}
                href={`/datasets/${d.id}`}
                className="group p-4 bg-slate-900/80 hover:bg-slate-850 border border-border rounded-xl transition-all hover:border-emerald-500/40"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-sm text-slate-200 group-hover:text-emerald-300">
                    {d.name}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {d.total_images} imgs
                  </span>
                </div>
                <div className="text-xs text-slate-400 flex items-center gap-3">
                  <span>Train: {d.train_images}</span>
                  <span>Val: {d.val_images}</span>
                </div>
              </Link>
            ))}

            {datasets.length === 0 && !loading && (
              <div className="col-span-2 py-8 text-center text-slate-500 text-sm">
                No datasets found. Upload a video clip to generate keyframes.
              </div>
            )}
          </div>
        </div>

        {/* Right 1 Col: Species Taxonomies */}
        <div className="bg-[#111827] border border-border rounded-2xl p-6 space-y-4">
          <h2 className="text-lg font-bold text-white">Supported Taxonomies</h2>
          <div className="space-y-3">
            <div className="p-3 bg-slate-900 border border-border rounded-xl">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm text-slate-200">🐖 Domestic Swine</span>
                <span className="text-[10px] font-mono text-emerald-400">18 Behaviors</span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Rooting, nesting, lateral recumbency, thermal huddling index.
              </p>
            </div>

            <div className="p-3 bg-slate-900 border border-border rounded-xl">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm text-slate-200">🐕 Domestic Canine</span>
                <span className="text-[10px] font-mono text-teal-400">24-Point Pose</span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Skeletal kinematics, spine curvature κ, play bow vs aggression triage.
              </p>
            </div>

            <div className="p-3 bg-slate-900 border border-border rounded-xl">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm text-slate-200">🦀 Redclaw Crayfish</span>
                <span className="text-[10px] font-mono text-cyan-400">Markov States</span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Aquaculture shelter occupancy, agonistic bouts, Clark-Evans index.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
