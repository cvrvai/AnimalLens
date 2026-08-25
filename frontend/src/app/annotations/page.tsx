'use client';

import React, { Suspense, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import {
  PenTool,
  ArrowLeft,
  Save,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Plus,
  CheckCircle2,
  Database,
  Layers,
} from 'lucide-react';
import { fetchDatasetFrames, fetchDatasets, updateFrameAnnotation } from '@/lib/api';
import { Keyframe } from '@/types/dataset';

function AnnotationStudioContent() {
  const searchParams = useSearchParams();
  const initialDataset = searchParams.get('dataset') || 'pig_dataset';
  const initialFrameName = searchParams.get('frame');

  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState(initialDataset);
  const [frames, setFrames] = useState<Keyframe[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [savedSuccess, setSavedSuccess] = useState(false);

  // Canvas Drawing State
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null);
  const [currentBoxes, setCurrentBoxes] = useState<any[]>([]);
  const [selectedBoxIdx, setSelectedBoxIdx] = useState<number | null>(null);

  // 1. Load Datasets
  useEffect(() => {
    fetchDatasets().then((ds) => {
      setDatasets(ds);
      if (ds.length > 0 && !selectedDataset) {
        setSelectedDataset(ds[0].id);
      }
    });
  }, []);

  // 2. Load Frames for Selected Dataset
  useEffect(() => {
    if (!selectedDataset) return;
    setLoading(true);
    fetchDatasetFrames(selectedDataset)
      .then((res) => {
        setFrames(res.frames || []);
        if (initialFrameName) {
          const idx = res.frames.findIndex((f) => f.image_name === initialFrameName);
          if (idx !== -1) setCurrentIndex(idx);
        } else {
          setCurrentIndex(0);
        }
      })
      .finally(() => setLoading(false));
  }, [selectedDataset]);

  // 3. Sync Current Frame Boxes
  const currentFrame = frames[currentIndex];
  useEffect(() => {
    if (currentFrame) {
      setCurrentBoxes([...currentFrame.bboxes]);
      setSelectedBoxIdx(null);
      setSavedSuccess(false);
    }
  }, [currentIndex, currentFrame]);

  // 4. Render Canvas with Image & Bounding Boxes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentFrame) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = currentFrame.image_url;

    img.onload = () => {
      canvas.width = img.naturalWidth || 640;
      canvas.height = img.naturalHeight || 480;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw bounding boxes
      currentBoxes.forEach((b, idx) => {
        const isSelected = selectedBoxIdx === idx;
        const x1 = (b.x_center - b.width / 2) * canvas.width;
        const y1 = (b.y_center - b.height / 2) * canvas.height;
        const w = b.width * canvas.width;
        const h = b.height * canvas.height;

        // Box border
        ctx.strokeStyle = isSelected ? '#38bdf8' : '#10b981';
        ctx.lineWidth = isSelected ? 4 : 3;
        ctx.strokeRect(x1, y1, w, h);

        // Fill highlight
        ctx.fillStyle = isSelected ? 'rgba(56, 189, 248, 0.2)' : 'rgba(16, 185, 129, 0.15)';
        ctx.fillRect(x1, y1, w, h);

        // Tag label
        ctx.fillStyle = isSelected ? '#0284c7' : '#059669';
        ctx.fillRect(x1, Math.max(0, y1 - 22), 80, 22);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(`Target #${idx + 1}`, x1 + 6, Math.max(14, y1 - 6));
      });
    };
  }, [currentFrame, currentBoxes, selectedBoxIdx]);

  // Canvas Mouse Interactions
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    setIsDrawing(true);
    setStartPos({ x, y });
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPos || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const endX = (e.clientX - rect.left) / rect.width;
    const endY = (e.clientY - rect.top) / rect.height;

    const x_min = Math.max(0, Math.min(startPos.x, endX));
    const x_max = Math.min(1, Math.max(startPos.x, endX));
    const y_min = Math.max(0, Math.min(startPos.y, endY));
    const y_max = Math.min(1, Math.max(startPos.y, endY));

    const width = x_max - x_min;
    const height = y_max - y_min;

    if (width > 0.02 && height > 0.02) {
      const newBox = {
        class_id: 0,
        x_center: x_min + width / 2,
        y_center: y_min + height / 2,
        width,
        height,
      };
      setCurrentBoxes([...currentBoxes, newBox]);
    }

    setIsDrawing(false);
    setStartPos(null);
  };

  const handleSave = async () => {
    if (!currentFrame) return;
    try {
      await updateFrameAnnotation(selectedDataset, currentFrame.image_name, currentFrame.split, currentBoxes);
      setSavedSuccess(true);
      setTimeout(() => setSavedSuccess(false), 2500);
    } catch (e) {
      console.error('Failed to save annotation', e);
    }
  };

  const handleDeleteSelected = () => {
    if (selectedBoxIdx !== null) {
      setCurrentBoxes(currentBoxes.filter((_, idx) => idx !== selectedBoxIdx));
      setSelectedBoxIdx(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-5">
        <div className="flex items-center gap-3">
          <Link
            href="/datasets"
            className="p-2 rounded-xl bg-slate-900 border border-border text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
              <PenTool className="w-6 h-6 text-emerald-400" />
              Interactive Keyframe Annotation Studio
            </h1>
            <p className="text-xs text-slate-400">
              Draw and verify bounding box annotations for transfer learning fine-tuning.
            </p>
          </div>
        </div>

        {/* Dataset Selector */}
        <div className="flex items-center gap-3">
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="bg-slate-900 border border-border rounded-xl px-3.5 py-2 text-xs text-slate-200 font-semibold focus:outline-none focus:border-emerald-500"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                📁 {d.name} ({d.total_images} frames)
              </option>
            ))}
          </select>

          <button
            onClick={handleSave}
            disabled={!currentFrame}
            className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-bold text-xs transition-all shadow-lg shadow-emerald-500/20"
          >
            {savedSuccess ? (
              <>
                <CheckCircle2 className="w-4 h-4" />
                Saved to Disk!
              </>
            ) : (
              <>
                <Save className="w-4 h-4" />
                Save Annotation
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Annotation Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Canvas Viewer (Left 3 Cols) */}
        <div className="lg:col-span-3 bg-[#111827] border border-border rounded-3xl p-5 flex flex-col items-center justify-between shadow-2xl">
          {/* Canvas Top Bar */}
          <div className="w-full flex items-center justify-between pb-3 mb-3 border-b border-border/60 text-xs">
            <div className="flex items-center gap-2 font-mono text-slate-300">
              <span className="text-emerald-400 font-bold">
                Frame {currentIndex + 1} / {frames.length}
              </span>
              <span className="text-slate-500">|</span>
              <span className="text-slate-400">{currentFrame?.image_name}</span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${
                  currentFrame?.split === 'train'
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-cyan-500/10 text-cyan-400'
                }`}
              >
                {currentFrame?.split}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
                disabled={currentIndex === 0}
                className="p-1.5 rounded-lg bg-slate-900 border border-border text-slate-300 hover:text-white disabled:opacity-30"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setCurrentIndex((prev) => Math.min(frames.length - 1, prev + 1))}
                disabled={currentIndex >= frames.length - 1}
                className="p-1.5 rounded-lg bg-slate-900 border border-border text-slate-300 hover:text-white disabled:opacity-30"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Interactive Canvas Container */}
          <div className="relative w-full max-h-[600px] flex items-center justify-center bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 cursor-crosshair">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseUp={handleMouseUp}
              className="max-w-full max-h-[560px] object-contain shadow-2xl"
            />
          </div>

          {/* Instructions Footer */}
          <div className="w-full pt-3 mt-3 border-t border-border/60 flex items-center justify-between text-xs text-slate-500">
            <span>Click and drag on the image to draw a new bounding box.</span>
            <span>Hotkeys: Arrow Left / Right to navigate frames.</span>
          </div>
        </div>

        {/* Right Sidebar: Bounding Boxes & Tooling */}
        <div className="bg-[#111827] border border-border rounded-3xl p-5 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-white">Bounding Boxes ({currentBoxes.length})</h3>
              {selectedBoxIdx !== null && (
                <button
                  onClick={handleDeleteSelected}
                  className="text-xs text-rose-400 hover:text-rose-300 flex items-center gap-1"
                >
                  <Trash2 className="w-3.5 h-3.5" /> Delete
                </button>
              )}
            </div>

            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
              {currentBoxes.map((box, idx) => (
                <div
                  key={idx}
                  onClick={() => setSelectedBoxIdx(idx)}
                  className={`p-3 rounded-xl border text-xs font-mono cursor-pointer transition-all ${
                    selectedBoxIdx === idx
                      ? 'bg-sky-500/10 border-sky-500/50 text-sky-300'
                      : 'bg-slate-900 border-border text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold">Box #{idx + 1}</span>
                    <span className="text-[10px] text-slate-500">Class 0</span>
                  </div>
                  <div className="text-[10px] text-slate-500">
                    center: ({box.x_center.toFixed(2)}, {box.y_center.toFixed(2)}) | size: {box.width.toFixed(2)} x {box.height.toFixed(2)}
                  </div>
                </div>
              ))}

              {currentBoxes.length === 0 && (
                <div className="py-8 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-xl">
                  No bounding boxes on this frame. Drag on canvas to add.
                </div>
              )}
            </div>
          </div>

          <div className="p-3 bg-slate-900 border border-border rounded-xl text-[11px] text-slate-400 space-y-1">
            <div className="font-semibold text-slate-300">YOLO Ground Truth Format:</div>
            <div className="font-mono text-slate-500">&lt;class&gt; &lt;x_center&gt; &lt;y_center&gt; &lt;w&gt; &lt;h&gt;</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function AnnotationStudioPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-400 text-sm">Loading Annotation Studio...</div>}>
      <AnnotationStudioContent />
    </Suspense>
  );
}

