'use client';

import React, { Suspense, useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';

export const dynamic = 'force-dynamic';
import {
  Flame,
  Cpu,
  Boxes,
  Sparkles,
  Layers,
  Database,
  Play,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Activity,
} from 'lucide-react';
import { fetchDatasets, fetchHealth, fetchSpecies, fetchTrainingJobs, startTrainingJob } from '@/lib/api';
import { Dataset } from '@/types/dataset';
import { SpeciesItem } from '@/types/species';
import { TrainingJob } from '@/types/training';

function TrainStudioContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialDataset = searchParams.get('dataset');

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [speciesList, setSpeciesList] = useState<SpeciesItem[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Form State
  const [selectedSpecies, setSelectedSpecies] = useState('pig');
  const [selectedDatasetYaml, setSelectedDatasetYaml] = useState('');
  const [baseModel, setBaseModel] = useState('yolov8s.pt');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(16);
  const [imgSize, setImgSize] = useState(640);
  const [device, setDevice] = useState('cpu');
  const [experimentName, setExperimentName] = useState('swine_farrowing_v1');

  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [ds, sp, jb, hl] = await Promise.all([
          fetchDatasets(),
          fetchSpecies(),
          fetchTrainingJobs(),
          fetchHealth(),
        ]);
        setDatasets(ds);
        setSpeciesList(sp);
        setJobs(jb);
        setHealth(hl);

        if (ds.length > 0) {
          if (initialDataset) {
            const match = ds.find((d) => d.id === initialDataset);
            if (match) setSelectedDatasetYaml(match.dataset_yaml);
            else setSelectedDatasetYaml(ds[0].dataset_yaml);
          } else {
            setSelectedDatasetYaml(ds[0].dataset_yaml);
          }
        }
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [initialDataset]);

  const handleLaunchTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDatasetYaml) {
      setError('Please select a valid dataset YAML configuration.');
      return;
    }

    setIsStarting(true);
    setError(null);

    try {
      const job = await startTrainingJob({
        species: selectedSpecies,
        dataset_yaml: selectedDatasetYaml,
        base_model: baseModel,
        epochs,
        batch: batchSize,
        imgsz: imgSize,
        device,
        experiment_name: experimentName,
      });

      router.push(`/train/monitor?job_id=${job.job_id}`);
    } catch (err: any) {
      setError(err.message || 'Failed to trigger training run.');
      setIsStarting(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Flame className="w-8 h-8 text-amber-400" />
            1-Click Model Training &amp; Fine-Tuning Studio
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Configure transfer learning hyperparameters, launch background GPU/CPU workers, and monitor real-time loss convergence.
          </p>
        </div>

        {jobs.length > 0 && (
          <Link
            href={`/train/monitor?job_id=${jobs[0].job_id}`}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-amber-400 text-xs font-semibold border border-amber-500/30 transition-colors"
          >
            <Activity className="w-3.5 h-3.5" />
            Active Monitor ({jobs[0].job_id})
          </Link>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left 2 Cols: Training Configuration Form */}
        <div className="lg:col-span-2 bg-[#111827] border border-border rounded-3xl p-6 sm:p-8 shadow-xl space-y-6">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            Training Job Configuration
          </h2>

          <form onSubmit={handleLaunchTraining} className="space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Dataset Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Training Dataset
                </label>
                <select
                  value={selectedDatasetYaml}
                  onChange={(e) => setSelectedDatasetYaml(e.target.value)}
                  className="w-full bg-slate-900 border border-border rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-amber-500"
                  required
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.dataset_yaml}>
                      📁 {d.name} ({d.total_images} images)
                    </option>
                  ))}
                  {datasets.length === 0 && (
                    <option value="">No datasets available</option>
                  )}
                </select>
              </div>

              {/* Target Species */}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Species Taxonomy
                </label>
                <select
                  value={selectedSpecies}
                  onChange={(e) => {
                    setSelectedSpecies(e.target.value);
                    setExperimentName(`${e.target.value}_model_v1`);
                  }}
                  className="w-full bg-slate-900 border border-border rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-amber-500"
                >
                  <option value="pig">🐖 Domestic Pig (Sus scrofa domesticus)</option>
                  <option value="dog">🐕 Domestic Dog (Canis lupus familiaris)</option>
                  <option value="redclaw">🦀 Redclaw Crayfish (Cherax)</option>
                </select>
              </div>

              {/* Base Model Architecture */}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Base Architecture Prior
                </label>
                <select
                  value={baseModel}
                  onChange={(e) => setBaseModel(e.target.value)}
                  className="w-full bg-slate-900 border border-border rounded-xl px-3.5 py-2.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-amber-500"
                >
                  <option value="yolov8s.pt">YOLOv8s (Small - 11.1M Params - Recommended)</option>
                  <option value="yolov8n.pt">YOLOv8n (Nano - 3.2M Params - Ultra Fast Edge)</option>
                  <option value="yolov8m.pt">YOLOv8m (Medium - 25.9M Params - High Accuracy)</option>
                  <option value="yolov8n-pose.pt">YOLOv8n-Pose (24-Point Skeletal Keypoints)</option>
                </select>
              </div>

              {/* Experiment Name */}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Experiment Name
                </label>
                <input
                  type="text"
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  className="w-full bg-slate-900 border border-border rounded-xl px-3.5 py-2.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-amber-500"
                  required
                />
              </div>
            </div>

            {/* Hyperparameter Sliders */}
            <div className="p-5 bg-slate-900/80 border border-border rounded-2xl space-y-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Hyperparameters &amp; Hardware
              </h3>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1.5">
                    <span>Epochs</span>
                    <span className="text-amber-400 font-mono">{epochs}</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="100"
                    step="5"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    className="w-full accent-amber-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1.5">
                    <span>Batch Size</span>
                    <span className="text-amber-400 font-mono">{batchSize}</span>
                  </div>
                  <input
                    type="range"
                    min="4"
                    max="64"
                    step="4"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    className="w-full accent-amber-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1.5">
                    <span>Image Resolution</span>
                    <span className="text-amber-400 font-mono">{imgSize}px</span>
                  </div>
                  <select
                    value={imgSize}
                    onChange={(e) => setImgSize(parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-border rounded-lg px-2.5 py-1 text-xs text-slate-200 font-mono"
                  >
                    <option value="480">480 x 480</option>
                    <option value="640">640 x 640 (Standard)</option>
                    <option value="1280">1280 x 1280 (High Res)</option>
                  </select>
                </div>
              </div>
            </div>

            {error && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={isStarting || datasets.length === 0}
              className="w-full py-3.5 px-6 rounded-2xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 disabled:opacity-50 text-slate-950 font-bold text-sm transition-all shadow-xl shadow-amber-500/20 flex items-center justify-center gap-2"
            >
              {isStarting ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Spawning Background Training Worker...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-slate-950" />
                  Start Transfer Learning Run
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Col: Recent Runs & Checkpoints */}
        <div className="bg-[#111827] border border-border rounded-3xl p-6 space-y-4">
          <h2 className="text-lg font-bold text-white">Recent Training Runs</h2>

          <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
            {jobs.map((j) => (
              <Link
                key={j.job_id}
                href={`/train/monitor?job_id=${j.job_id}`}
                className="block p-4 bg-slate-900/80 hover:bg-slate-850 border border-border rounded-2xl transition-all hover:border-amber-500/40"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-xs text-slate-200">{j.experiment_name}</span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                      j.status === 'COMPLETED'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        : j.status === 'RUNNING'
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30 animate-pulse'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {j.status}
                  </span>
                </div>

                <div className="text-[11px] text-slate-400 space-y-1 font-mono">
                  <div className="flex justify-between">
                    <span>Species: {j.species}</span>
                    <span>Epochs: {j.epochs}</span>
                  </div>
                  {j.report && (
                    <div className="text-emerald-400 font-semibold flex justify-between pt-1 border-t border-border/40">
                      <span>mAP@50: {(j.report.map50 * 100).toFixed(1)}%</span>
                      <span>mAP50-95: {(j.report.map50_95 * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              </Link>
            ))}

            {jobs.length === 0 && !loading && (
              <div className="py-8 text-center text-slate-500 text-xs">
                No past training jobs found.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function TrainStudioPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-400 text-sm">Loading Training Studio...</div>}>
      <TrainStudioContent />
    </Suspense>
  );
}

