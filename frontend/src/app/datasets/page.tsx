'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Database,
  UploadCloud,
  Layers,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  FileVideo,
  ArrowRight,
  RefreshCw,
} from 'lucide-react';
import { fetchDatasets, fetchSpecies, uploadDatasetVideo } from '@/lib/api';
import { Dataset } from '@/types/dataset';
import { SpeciesItem } from '@/types/species';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [speciesList, setSpeciesList] = useState<SpeciesItem[]>([]);
  const [loading, setLoading] = useState(true);

  // Upload Form State
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('pig_farrowing_dataset');
  const [selectedSpecies, setSelectedSpecies] = useState('pig');
  const [sampleFps, setSampleFps] = useState(2.0);
  const [valSplit, setValSplit] = useState(0.2);
  const [autoLabel, setAutoLabel] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      const [ds, sp] = await Promise.all([fetchDatasets(), fetchSpecies()]);
      setDatasets(ds);
      setSpeciesList(sp);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoFile) {
      setUploadError('Please select a video file (.mp4, .webm, .avi).');
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadResult(null);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('dataset_name', datasetName);
    formData.append('species', selectedSpecies);
    formData.append('sample_fps', sampleFps.toString());
    formData.append('val_split', valSplit.toString());
    formData.append('auto_pseudo_label', autoLabel ? 'true' : 'false');

    try {
      const res = await uploadDatasetVideo(formData);
      setUploadResult(res);
      await loadData();
    } catch (err: any) {
      setUploadError(err.message || 'Video upload and slicing failed.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Database className="w-8 h-8 text-emerald-400" />
            Dataset Ingestion &amp; Keyframe Studio
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Upload raw farm/pen video, extract uniform keyframes at configurable FPS, auto pseudo-label, and structure YOLO datasets.
          </p>
        </div>

        <button
          onClick={loadData}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-border transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Datasets
        </button>
      </div>

      {/* Video Slicer Upload Card */}
      <div className="bg-[#111827] border border-border rounded-3xl p-6 sm:p-8 shadow-xl">
        <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
          <UploadCloud className="w-5 h-5 text-emerald-400" />
          1-Click Video Slicer &amp; Auto Pseudo-Labeler
        </h2>
        <p className="text-xs text-slate-400 mb-6">
          Drop a video clip below. The backend will extract keyframes, generate YOLO bounding box labels with detection priors, and write `dataset.yaml`.
        </p>

        <form onSubmit={handleUpload} className="space-y-6">
          {/* File Input Drop Area */}
          <div className="border-2 border-dashed border-slate-700 hover:border-emerald-500/50 rounded-2xl p-6 text-center transition-colors bg-slate-900/50">
            <input
              type="file"
              id="video-upload"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setVideoFile(e.target.files[0]);
                  const cleanName = e.target.files[0].name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_]/g, '_');
                  setDatasetName(`${cleanName}_dataset`);
                }
              }}
            />
            <label htmlFor="video-upload" className="cursor-pointer block">
              <FileVideo className="w-12 h-12 text-slate-500 mx-auto mb-3" />
              {videoFile ? (
                <div>
                  <span className="font-semibold text-emerald-400 text-sm block mb-1">{videoFile.name}</span>
                  <span className="text-xs text-slate-500 font-mono">
                    ({(videoFile.size / 1024 / 1024).toFixed(1)} MB)
                  </span>
                </div>
              ) : (
                <div>
                  <span className="font-medium text-sm text-slate-300 block mb-1">
                    Click or drag &amp; drop a video file here (.mp4, .webm, .avi)
                  </span>
                  <span className="text-xs text-slate-500">e.g. pig_farm_pen.mp4, canine_agility.mp4</span>
                </div>
              )}
            </label>
          </div>

          {/* Form Settings Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Dataset Identifier</label>
              <input
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="w-full bg-slate-900 border border-border rounded-xl px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-emerald-500"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Species Taxonomy</label>
              <select
                value={selectedSpecies}
                onChange={(e) => setSelectedSpecies(e.target.value)}
                className="w-full bg-slate-900 border border-border rounded-xl px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
              >
                <option value="pig">🐖 Domestic Pig (Sus scrofa)</option>
                <option value="dog">🐕 Domestic Dog (Canis lupus)</option>
                <option value="redclaw">🦀 Redclaw Crayfish (Cherax)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Sampling Rate: <span className="text-emerald-400 font-mono">{sampleFps} FPS</span>
              </label>
              <input
                type="range"
                min="0.5"
                max="5.0"
                step="0.5"
                value={sampleFps}
                onChange={(e) => setSampleFps(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Validation Split: <span className="text-emerald-400 font-mono">{(valSplit * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={valSplit}
                onChange={(e) => setValSplit(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>
          </div>

          <div className="flex items-center justify-between pt-2">
            <label className="flex items-center gap-2.5 cursor-pointer text-xs text-slate-300">
              <input
                type="checkbox"
                checked={autoLabel}
                onChange={(e) => setAutoLabel(e.target.checked)}
                className="rounded bg-slate-900 border-border text-emerald-500 focus:ring-emerald-500 w-4 h-4"
              />
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                Run Automated AI Pseudo-Labeling with base detector priors
              </span>
            </label>

            <button
              type="submit"
              disabled={isUploading || !videoFile}
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-bold text-sm transition-all shadow-lg shadow-emerald-500/20"
            >
              {isUploading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Slicing Keyframes &amp; Labeling...
                </>
              ) : (
                <>
                  <Layers className="w-4 h-4" />
                  Extract &amp; Build Dataset
                </>
              )}
            </button>
          </div>
        </form>

        {/* Feedback Messages */}
        {uploadError && (
          <div className="mt-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{uploadError}</span>
          </div>
        )}

        {uploadResult && (
          <div className="mt-6 p-5 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 space-y-2">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
              <CheckCircle2 className="w-5 h-5" />
              Dataset Successfully Built: {uploadResult.dataset_name}
            </div>
            <div className="text-xs text-slate-300 grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono">
              <div>Total: {uploadResult.extracted_frames} frames</div>
              <div>Train: {uploadResult.train_frames} frames</div>
              <div>Val: {uploadResult.val_frames} frames</div>
              <div>Pseudo-Labels: {uploadResult.pseudo_labels_generated}</div>
            </div>
          </div>
        )}
      </div>

      {/* Dataset Explorer Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-white">Available Datasets ({datasets.length})</h2>
          <span className="text-xs text-slate-500 font-mono">Stored in datasets/ &amp; models/trained/</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {datasets.map((d) => (
            <div
              key={d.id}
              className="bg-[#111827] border border-border rounded-2xl p-5 flex flex-col justify-between hover:border-emerald-500/40 transition-all shadow-lg space-y-4"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-bold text-base text-white">{d.name}</h3>
                  <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {d.total_images} frames
                  </span>
                </div>

                <div className="text-xs text-slate-400 space-y-1.5">
                  <div className="flex justify-between">
                    <span>Train Split:</span>
                    <span className="font-mono text-slate-200">{d.train_images} images</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Validation Split:</span>
                    <span className="font-mono text-slate-200">{d.val_images} images</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Config:</span>
                    <span className="font-mono text-slate-400 text-[10px] truncate max-w-[180px]">
                      {d.dataset_yaml.split(/[\\/]/).pop()}
                    </span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-border flex items-center justify-between gap-2">
                <Link
                  href={`/datasets/${d.id}`}
                  className="flex-1 text-center py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors"
                >
                  Inspect Frames
                </Link>
                <Link
                  href={`/annotations?dataset=${d.id}`}
                  className="flex-1 text-center py-2 px-3 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 text-xs font-semibold text-emerald-400 border border-emerald-500/30 transition-colors inline-flex items-center justify-center gap-1"
                >
                  Annotate <ArrowRight className="w-3 h-3" />
                </Link>
              </div>
            </div>
          ))}

          {datasets.length === 0 && !loading && (
            <div className="col-span-3 p-12 text-center text-slate-500 text-sm bg-slate-900/50 rounded-2xl border border-dashed border-slate-800">
              No datasets found. Use the video slicer above to build your first training dataset.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
