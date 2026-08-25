import type { Metadata } from 'next';
import './globals.css';
import { Sidebar } from '@/components/common/Sidebar';
import { Navbar } from '@/components/common/Navbar';

export const metadata: Metadata = {
  title: 'AnimalLens Studio — Multi-Species Vision & Ethology Platform',
  description: 'Web studio for dataset management, video keyframe annotation, 1-click model training, and real-time animal behavior intelligence.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#090d16] text-slate-100 min-h-screen flex antialiased">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <Navbar />
          <main className="flex-1 p-6 max-w-7xl w-full mx-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
