import { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Hero3D, ScanState } from './components/Hero3D';
import { LiveKiosk } from './components/LiveKiosk';
import { EnrollmentWizard } from './components/EnrollmentWizard';
import { AdminDashboard } from './components/AdminDashboard';
import { Camera, UserPlus, ShieldCheck, ArrowRight, Sparkles, Cpu, Lock } from 'lucide-react';

export function App() {
  const [activeView, setActiveView] = useState<'home' | 'kiosk' | 'enroll' | 'admin'>('home');
  const [heroScanState, setHeroScanState] = useState<ScanState>('idle');

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      <Navbar activeView={activeView} setActiveView={setActiveView} />

      <main className="flex-1">
        {activeView === 'home' && (
          <div>
            {/* Hero Section */}
            <div className="relative overflow-hidden pt-12 pb-20 border-b border-slate-200 dark:border-slate-800">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                {/* Left Pitch */}
                <div className="space-y-6 z-10">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-sky-50 dark:bg-sky-950/60 text-sky-600 dark:text-sky-400 border border-sky-500/20">
                    <Sparkles className="w-3.5 h-3.5" />
                    Open-Set Metric Learning &bull; Zero Retraining
                  </div>

                  <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight leading-tight text-slate-900 dark:text-white">
                    Next-Gen Facial <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-500 to-indigo-500">
                      Attendance &amp; Verification
                    </span>
                  </h1>

                  <p className="text-base sm:text-lg text-slate-600 dark:text-slate-400 leading-relaxed max-w-xl">
                    Enroll new people in under 1 minute without retraining neural networks. Powered by deep 512-dimensional ArcFace metric embeddings, passive anti-spoof liveness detection, and instant browser webcam capture.
                  </p>

                  <div className="flex flex-wrap items-center gap-4 pt-2">
                    <button
                      onClick={() => setActiveView('kiosk')}
                      className="px-6 py-3 rounded-2xl bg-sky-500 hover:bg-sky-600 text-white font-bold text-sm flex items-center gap-2 shadow-lg shadow-sky-500/25 transition"
                    >
                      <Camera className="w-4 h-4" /> Launch Live Kiosk
                      <ArrowRight className="w-4 h-4" />
                    </button>

                    <button
                      onClick={() => setActiveView('enroll')}
                      className="px-6 py-3 rounded-2xl bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 font-bold text-sm text-slate-900 dark:text-white flex items-center gap-2 transition"
                    >
                      <UserPlus className="w-4 h-4 text-sky-500" /> Enroll Person
                    </button>
                  </div>

                  {/* Trust Badges */}
                  <div className="grid grid-cols-3 gap-4 pt-4 border-t border-slate-200 dark:border-slate-800/80">
                    <div>
                      <p className="text-xs text-slate-500">Inference Latency</p>
                      <p className="text-base font-bold text-slate-900 dark:text-white">&lt; 15 ms / face</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500">Retraining Required</p>
                      <p className="text-base font-bold text-emerald-500">0 Seconds</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500">Anti-Spoofing</p>
                      <p className="text-base font-bold text-sky-500">Passive PAD</p>
                    </div>
                  </div>
                </div>

                {/* Right Interactive 3D Scene */}
                <div className="relative rounded-3xl bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-900/60 dark:to-slate-950 border border-slate-200 dark:border-slate-800/80 p-4 shadow-2xl flex flex-col items-center">
                  <div className="w-full h-[400px]">
                    <Hero3D scanState={heroScanState} />
                  </div>

                  {/* State Simulation Interactive Controls */}
                  <div className="flex items-center gap-2 mt-2 p-1.5 rounded-2xl bg-white/70 dark:bg-slate-900/80 backdrop-blur-md border border-slate-200 dark:border-slate-800">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 px-2">
                      Scan Simulator:
                    </span>
                    <button
                      onClick={() => setHeroScanState('idle')}
                      className={`px-2.5 py-1 rounded-xl text-xs font-semibold transition ${
                        heroScanState === 'idle' ? 'bg-sky-500 text-white' : 'text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      Idle
                    </button>
                    <button
                      onClick={() => setHeroScanState('scanning')}
                      className={`px-2.5 py-1 rounded-xl text-xs font-semibold transition ${
                        heroScanState === 'scanning' ? 'bg-sky-500 text-white' : 'text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      Scanning
                    </button>
                    <button
                      onClick={() => setHeroScanState('success')}
                      className={`px-2.5 py-1 rounded-xl text-xs font-semibold transition ${
                        heroScanState === 'success' ? 'bg-emerald-500 text-white' : 'text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      Verified
                    </button>
                    <button
                      onClick={() => setHeroScanState('unknown')}
                      className={`px-2.5 py-1 rounded-xl text-xs font-semibold transition ${
                        heroScanState === 'unknown' ? 'bg-amber-500 text-white' : 'text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      Unknown
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Feature Highlights Grid */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
                  <div className="w-12 h-12 rounded-2xl bg-sky-500/10 text-sky-500 flex items-center justify-center">
                    <Cpu className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                    Open-Set Metric Learning
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                    Replaced brittle closed-set classification with 512-dimensional L2-normalized ArcFace metric vectors. Supports infinite scalable identities with zero retraining overhead.
                  </p>
                </div>

                <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
                  <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center">
                    <ShieldCheck className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                    Passive Anti-Spoof Protection
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                    Fourier high-frequency moiré analysis, chrominance gamut inspection, and specular reflection checks reject printed photo attacks and digital screen replays.
                  </p>
                </div>

                <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
                  <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 text-indigo-500 flex items-center justify-center">
                    <Lock className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                    Privacy &amp; DPDP Compliance
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                    Mandatory explicit consent audit timestamps at enrollment. Raw images are discarded; irreversible mathematical vectors are stored, with full Right to be Forgotten data erasure.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeView === 'kiosk' && <LiveKiosk />}
        {activeView === 'enroll' && <EnrollmentWizard onEnrollComplete={() => setActiveView('admin')} />}
        {activeView === 'admin' && <AdminDashboard />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 py-6 text-center text-xs text-slate-500 dark:text-slate-400">
        <p>
          BioScan 2.0 &bull; Production-Grade Biometric Attendance System &bull; Released under MIT License
        </p>
      </footer>
    </div>
  );
}
