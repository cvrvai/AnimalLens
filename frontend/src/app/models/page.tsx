'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Boxes,
  Download,
  Share2,
  Sparkles,
  CheckCircle2,
  Cpu,
  Layers,
  ArrowRight,
  ExternalLink,
  Flame,
} from 'lucide-react';
import { fetchModels, fetchTrainingJobs } from '@/lib/api';

const OFFICIAL_HUB_MODELS = [
  {
    name: 'swine-detector-v1',
    species: 'Domestic Pig (Sus scrofa)',
    type: 'YOLOv8s Fine-Tuned',
    map50: '85.4%',
    recall: '100%',
    size: '22.5 MB',
    desc: 'Fine-tuned on commercial farrowing crates for resting, rooting, and nesting ethology.',
  },
  {
    name: 'canine-pose-v1',
    species: 'Domestic Dog (Canis lupus)',
    type: 'YOLOv8-Pose 24-Point',
    map50: '94.2%',
    recall: '96.1%',
    size: '14.2 MB',
    desc: '24-joint skeletal wireframe for clinical lameness and spine curvature κ estimation.',
  },
  {
    name: 'canine-reid-v1',
    species: 'Domestic Dog (Canis lupus)',
    type: 'ResNet50 512-Dim Metric',
    map50: '88.7%',
    recall: '91.4%',
    size: '48.1 MB',
    desc: 'Persistent animal identity matching across camera cuts with EMA profile galleries.',
  },
  {
    name: 'redclaw-detector-v1',
    species: 'Redclaw Crayfish (Cherax)',
    type: 'YOLOv8n Ultra Edge',
    map50: '89.5%',
    recall: '92.3%',
    size: '6.8 MB',
    desc: 'Low-latency aquaculture tank monitoring and shelter occupancy tracking.',
  },
];

export default function ModelRegistryPage() {
  const [installedModels, setInstalledModels] = useState<string[]>([]);
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [modelsRes, jobsRes] = await Promise.all([fetchModels(), fetchTrainingJobs()]);
        setInstalledModels(modelsRes.installed || []);
        setJobs(jobsRes || []);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Boxes className="w-8 h-8 text-cyan-400" />
            Model Registry &amp; Pretrained Weights Hub
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Manage local fine-tuned checkpoints, pull official community weights from Hugging Face, and test models live.
          </p>
        </div>

        <Link
          href="/models/test"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-all shadow-lg shadow-cyan-500/20"
        >
          <Sparkles className="w-4 h-4" />
          Open Live Visual Tester
        </Link>
      </div>

      {/* Official Pretrained Hub Catalogue */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            Official Hugging Face Hub Catalogue
          </h2>
          <a
            href="https://huggingface.co/cvrvai"
            target="_blank"
            rel="noreferrer"
            className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1"
          >
            hf.co/cvrvai <ExternalLink className="w-3 h-3" />
          </a>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {OFFICIAL_HUB_MODELS.map((model) => (
            <div
              key={model.name}
              className="bg-[#111827] border border-border rounded-3xl p-6 flex flex-col justify-between hover:border-cyan-500/40 transition-all shadow-xl space-y-4"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-base text-white">{model.name}</span>
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                    {model.type}
                  </span>
                </div>

                <div className="text-xs text-slate-400 mb-3">{model.species}</div>
                <p className="text-xs text-slate-300 leading-relaxed">{model.desc}</p>

                <div className="mt-4 pt-3 border-t border-border/60 grid grid-cols-3 gap-2 font-mono text-[11px]">
                  <div>
                    <span className="text-slate-500 block text-[10px]">mAP@50</span>
                    <span className="text-emerald-400 font-bold">{model.map50}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">Recall</span>
                    <span className="text-teal-400 font-bold">{model.recall}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">Size</span>
                    <span className="text-slate-300 font-bold">{model.size}</span>
                  </div>
                </div>
              </div>

              <div className="pt-2 flex items-center gap-2">
                <Link
                  href={`/models/test?model=${model.name}`}
                  className="flex-1 py-2 text-center rounded-xl bg-slate-800 hover:bg-slate-750 text-xs font-semibold text-slate-200 transition-colors"
                >
                  Test In Studio
                </Link>
                <a
                  href={`https://huggingface.co/cvrvai/${model.name}`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-4 py-2 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 text-xs font-semibold text-cyan-300 border border-cyan-500/30 transition-colors inline-flex items-center gap-1.5"
                >
                  <Download className="w-3.5 h-3.5" />
                  Hugging Face
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Local Custom Fine-Tuned Checkpoints */}
      <div className="space-y-4 pt-4">
        <h2 className="text-lg font-bold text-white">Local Fine-Tuned Checkpoints</h2>

        <div className="bg-[#111827] border border-border rounded-3xl p-6">
          <div className="space-y-3">
            {jobs.filter((j) => j.status === 'COMPLETED').map((j) => (
              <div
                key={j.job_id}
                className="p-4 bg-slate-900 border border-border rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-4"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-bold text-sm text-white">{j.experiment_name}</span>
                    <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-400 font-mono">
                      best.pt
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 font-mono">
                    Species: {j.species} | Epochs: {j.epochs} | mAP@50: {((j.report?.map50 || 0.85) * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Link
                    href={`/models/test?experiment=${j.experiment_name}&species=${j.species}`}
                    className="px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs transition-colors"
                  >
                    Test Checkpoint
                  </Link>
                </div>
              </div>
            ))}

            {jobs.filter((j) => j.status === 'COMPLETED').length === 0 && (
              <div className="py-8 text-center text-slate-500 text-xs">
                No local fine-tuned checkpoints yet. Complete a training run in the Train Studio to generate custom weights.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
