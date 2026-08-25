'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Database,
  PenTool,
  Flame,
  Boxes,
  Camera,
  Activity,
  Github,
  BookOpen,
} from 'lucide-react';

const NAV_ITEMS = [
  { name: 'Overview', href: '/', icon: LayoutDashboard },
  { name: 'Datasets Hub', href: '/datasets', icon: Database },
  { name: 'Annotation Studio', href: '/annotations', icon: PenTool },
  { name: 'Train & Fine-Tune', href: '/train', icon: Flame },
  { name: 'Model Registry', href: '/models', icon: Boxes },
  { name: 'Live Edge Streams', href: '/streams', icon: Camera },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border bg-[#0b101d] flex flex-col justify-between shrink-0 h-screen sticky top-0">
      <div>
        {/* Brand Logo & Name */}
        <div className="h-16 flex items-center px-6 border-b border-border gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-emerald-300 to-teal-200 bg-clip-text text-transparent">
              AnimalLens
            </span>
            <span className="block text-[10px] text-slate-500 font-mono tracking-wider uppercase">
              Vision & Ethology Studio
            </span>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="p-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href));
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer Info & Links */}
      <div className="p-4 border-t border-border space-y-2">
        <div className="bg-slate-900/80 rounded-xl p-3 border border-border/60">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Inference Engine</span>
            <span className="text-emerald-400 font-mono font-medium">YOLOv8 + BoT-SORT</span>
          </div>
          <div className="text-[11px] text-slate-500">
            Decoupled 2-Layer Vision & Ollama AI
          </div>
        </div>

        <div className="flex items-center justify-between px-2 pt-1 text-xs text-slate-500">
          <a
            href="https://github.com/cvrvai/AnimalLens"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 hover:text-emerald-400 transition-colors"
          >
            <Github className="w-3.5 h-3.5" />
            GitHub
          </a>
          <a
            href="http://localhost:8088/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 hover:text-emerald-400 transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5" />
            API Docs
          </a>
        </div>
      </div>
    </aside>
  );
}
