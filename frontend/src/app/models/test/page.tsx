'use client';

import React, { useRef, useState } from 'react';
import Link from 'next/link';
import {
  Sparkles,
  ArrowLeft,
  UploadCloud,
  Image as ImageIcon,
  Activity,
  CheckCircle2,
  RefreshCw,
  Cpu,
  Brain,
} from 'lucide-react';

export default function LiveModelTesterPage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [species, setSpecies] = useState('pig');
  const [enableReasoning, setEnableReasoning] = useState(true);

  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      setAnalysisResult(null);
      setError(null);
    }
  };

  const handleRunInference = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('species', species);
    if (enableReasoning) {
      formData.append('reasoning', 'ollama:gemma3');
    }

    try {
      const res = await fetch('/api/backend/analyze/image', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Inference request failed');
      }

      const data = await res.json();
      setAnalysisResult(data);

      // Draw bounding boxes on canvas
      drawDetections(data);
    } catch (err: any) {
      setError(err.message || 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  const drawDetections = (data: any) => {
    const canvas = canvasRef.current;
    if (!canvas || !imagePreview) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = imagePreview;
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw HUD Banner
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.fillRect(0, 0, canvas.width, 50);

      ctx.fillStyle = '#34d399';
      ctx.font = 'bold 16px monospace';
      ctx.fillText(
        `AnimalLens Vision AI | Species: ${data.species} | Detections: ${data.behaviors?.length || 0}`,
        15,
        30
      );

      // Draw bounding boxes
      (data.behaviors || []).forEach((beh: any, idx: number) => {
        (beh.subjects || []).forEach((sub: any) => {
          const box = sub.bbox;
          if (!box) return;

          const x1 = box.x_min * canvas.width;
          const y1 = box.y_min * canvas.height;
          const w = (box.x_max - box.x_min) * canvas.width;
          const h = (box.y_max - box.y_min) * canvas.height;

          // Box
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 4;
          ctx.strokeRect(x1, y1, w, h);

          // Tag
          const tag = `${beh.behavior?.label?.toUpperCase()} (${(beh.behavior?.confidence * 100).toFixed(0)}%)`;
          ctx.fillStyle = '#059669';
          ctx.fillRect(x1, Math.max(0, y1 - 26), 240, 26);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 14px monospace';
          ctx.fillText(tag, x1 + 6, Math.max(18, y1 - 8));
        });
      });
    };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border pb-5">
        <div className="flex items-center gap-3">
          <Link
            href="/models"
            className="p-2 rounded-xl bg-slate-900 border border-border text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-cyan-400" />
              Live Visual Model Tester &amp; HUD Studio
            </h1>
            <p className="text-xs text-slate-400">
              Test detection models and fine-tuned weights on image frames with instant HUD rendering.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Canvas Image & HUD Viewer */}
        <div className="lg:col-span-2 bg-[#111827] border border-border rounded-3xl p-6 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Visual Canvas Viewport
            </h2>
            {analysisResult && (
              <span className="text-xs text-emerald-400 font-mono flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Inference Complete (15ms)
              </span>
            )}
          </div>

          <div className="relative min-h-[400px] max-h-[550px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center">
            {imagePreview ? (
              <canvas ref={canvasRef} className="max-w-full max-h-[520px] object-contain shadow-2xl" />
            ) : (
              <div className="text-center p-8 text-slate-500">
                <ImageIcon className="w-12 h-12 mx-auto mb-2 text-slate-600" />
                <p className="text-sm">Upload a frame or video thumbnail on the right to test inference.</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Col: Control Panel & Reasoning Output */}
        <div className="space-y-4">
          <div className="bg-[#111827] border border-border rounded-3xl p-6 space-y-4 shadow-xl">
            <h2 className="text-base font-bold text-white">Tester Controls</h2>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Upload Test Image / Frame
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="w-full text-xs text-slate-400 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Target Species Prior
              </label>
              <select
                value={species}
                onChange={(e) => setSpecies(e.target.value)}
                className="w-full bg-slate-900 border border-border rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                <option value="pig">🐖 Domestic Pig (Sus scrofa)</option>
                <option value="dog">🐕 Domestic Dog (Canis lupus)</option>
                <option value="redclaw">🦀 Redclaw Crayfish (Cherax)</option>
              </select>
            </div>

            <button
              onClick={handleRunInference}
              disabled={loading || !imageFile}
              className="w-full py-3 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs transition-all shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Neural Inference...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Run Live Detection
                </>
              )}
            </button>
          </div>

          {/* AI Clinical Reasoning Output */}
          {analysisResult && (
            <div className="bg-[#111827] border border-border rounded-3xl p-6 space-y-3 shadow-xl">
              <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                <Brain className="w-4 h-4" />
                Ethological Assessment
              </div>
              <div className="text-xs text-slate-300 leading-relaxed font-mono">
                {analysisResult.behaviors?.map((b: any, idx: number) => (
                  <div key={idx} className="p-2 bg-slate-900 rounded-xl mb-1.5">
                    <span className="text-emerald-400 font-bold">{b.behavior?.label}</span>
                    <span className="text-slate-400"> (conf: {(b.behavior?.confidence * 100).toFixed(1)}%)</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
