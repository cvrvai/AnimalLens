'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import {
  Database,
  ArrowLeft,
  PenTool,
  Image as ImageIcon,
  Flame,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';
import { fetchDatasetFrames } from '@/lib/api';
import { Keyframe } from '@/types/dataset';

export default function DatasetDetailPage() {
  const params = useParams();
  const datasetId = params.id as string;

  const [frames, setFrames] = useState<Keyframe[]>([]);
  const [activeSplit, setActiveSplit] = useState<'all' | 'train' | 'val'>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadFrames() {
      try {
        setLoading(true);
        const data = await fetchDatasetFrames(datasetId);
        setFrames(data.frames || []);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    if (datasetId) loadFrames();
  }, [datasetId]);

  const filteredFrames = frames.filter((f) => activeSplit === 'all' || f.split === activeSplit);

  return (
    <div className="space-y-6">
      {/* Back Navigation & Header */}
      <div className="flex items-center justify-between border-b border-border pb-6">
        <div className="flex items-center gap-4">
          <Link
            href="/datasets"
            className="p-2 rounded-xl bg-slate-900 border border-border text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
              <Database className="w-6 h-6 text-emerald-400" />
              {datasetId.replace(/_/g, ' ').toUpperCase()}
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Extracted keyframes and bounding box ground truth annotations
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href={`/annotations?dataset=${datasetId}`}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs transition-colors shadow-lg shadow-emerald-500/20"
          >
            <PenTool className="w-3.5 h-3.5" />
            Open Annotation Canvas
          </Link>
          <Link
            href={`/train?dataset=${datasetId}`}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs transition-colors shadow-lg shadow-amber-500/20"
          >
            <Flame className="w-3.5 h-3.5" />
            Train This Dataset
          </Link>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 p-1 bg-slate-900 border border-border rounded-xl">
          <button
            onClick={() => setActiveSplit('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeSplit === 'all' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All Frames ({frames.length})
          </button>
          <button
            onClick={() => setActiveSplit('train')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeSplit === 'train' ? 'bg-slate-800 text-emerald-400' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Train Split ({frames.filter((f) => f.split === 'train').length})
          </button>
          <button
            onClick={() => setActiveSplit('val')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeSplit === 'val' ? 'bg-slate-800 text-cyan-400' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Validation Split ({frames.filter((f) => f.split === 'val').length})
          </button>
        </div>

        <span className="text-xs text-slate-500 font-mono">
          Showing {filteredFrames.length} keyframes
        </span>
      </div>

      {/* Frame Gallery Grid */}
      {loading ? (
        <div className="py-20 text-center text-slate-400 flex items-center justify-center gap-2 text-sm">
          <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
          Loading keyframes from disk...
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {filteredFrames.map((frame, idx) => (
            <Link
              key={frame.image_name}
              href={`/annotations?dataset=${datasetId}&frame=${frame.image_name}&split=${frame.split}`}
              className="group bg-[#111827] border border-border rounded-xl overflow-hidden hover:border-emerald-500/50 transition-all flex flex-col justify-between"
            >
              <div className="relative aspect-video bg-slate-950 overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={frame.image_url}
                  alt={frame.image_name}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  loading="lazy"
                />
                <span
                  className={`absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                    frame.split === 'train'
                      ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-500/40'
                      : 'bg-cyan-950/80 text-cyan-300 border border-cyan-500/40'
                  }`}
                >
                  {frame.split.toUpperCase()}
                </span>
                {frame.bboxes.length > 0 && (
                  <span className="absolute bottom-2 right-2 px-1.5 py-0.5 rounded text-[10px] font-mono bg-black/80 text-white">
                    {frame.bboxes.length} boxes
                  </span>
                )}
              </div>

              <div className="p-2.5 text-[11px] font-mono text-slate-400 flex items-center justify-between">
                <span className="truncate max-w-[100px]">{frame.image_name}</span>
                <span className="text-emerald-400 group-hover:underline">Edit</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
