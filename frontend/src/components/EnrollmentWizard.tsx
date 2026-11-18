import React, { useRef, useState, useEffect } from 'react';
import { Camera, Check, ShieldCheck, UserPlus, AlertCircle, ArrowRight, RotateCcw } from 'lucide-react';
import { api } from '../services/api';

const CAPTURE_PROMPTS = [
  { step: 1, title: 'Front Pose', desc: 'Look directly into the camera with a neutral expression' },
  { step: 2, title: 'Slight Left', desc: 'Gently turn your head ~15 degrees to the left' },
  { step: 3, title: 'Slight Right', desc: 'Gently turn your head ~15 degrees to the right' },
];

export const EnrollmentWizard: React.FC<{ onEnrollComplete?: () => void }> = ({ onEnrollComplete }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  // Form State
  const [fullName, setFullName] = useState('');
  const [externalId, setExternalId] = useState('');
  const [department, setDepartment] = useState('Computer Science');
  const [consentGiven, setConsentGiven] = useState(false);

  // Capture State
  const [currentPromptIdx, setCurrentPromptIdx] = useState(0);
  const [capturedBlobs, setCapturedBlobs] = useState<Blob[]>([]);
  const [capturedThumbnails, setCapturedThumbnails] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Initialize camera for enrollment wizard
  useEffect(() => {
    let activeStream: MediaStream | null = null;
    async function startCamera() {
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = activeStream;
          videoRef.current.play();
        }
        setStream(activeStream);
      } catch (err: any) {
        setStatusMessage({ type: 'error', text: 'Camera access denied or unavailable.' });
      }
    }
    startCamera();

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const takeSnapshot = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

    canvas.toBlob((blob) => {
      if (blob) {
        setCapturedBlobs((prev) => [...prev, blob]);
        setCapturedThumbnails((prev) => [...prev, dataUrl]);
        if (currentPromptIdx < CAPTURE_PROMPTS.length - 1) {
          setCurrentPromptIdx((prev) => prev + 1);
        }
      }
    }, 'image/jpeg', 0.9);
  };

  const handleReset = () => {
    setCapturedBlobs([]);
    setCapturedThumbnails([]);
    setCurrentPromptIdx(0);
    setStatusMessage(null);
  };

  const handleSubmitEnrollment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!consentGiven) {
      setStatusMessage({ type: 'error', text: 'You must check the biometric privacy consent agreement.' });
      return;
    }
    if (capturedBlobs.length < 3) {
      setStatusMessage({ type: 'error', text: 'Please complete all 3 guided angle captures.' });
      return;
    }

    setIsSubmitting(true);
    setStatusMessage(null);

    try {
      // 1. Create person record with consent timestamp
      const person = await api.createPerson({
        full_name: fullName,
        external_id: externalId || undefined,
        department,
        consent_given: true,
      });

      // 2. Upload facial photos for quality gating and 512-d ArcFace vector generation
      await api.enrollFaces(person.id, capturedBlobs);

      setStatusMessage({
        type: 'success',
        text: `Successfully enrolled ${fullName}! Zero retraining required—ready for instant recognition.`,
      });

      // Clear fields
      setFullName('');
      setExternalId('');
      setCapturedBlobs([]);
      setCapturedThumbnails([]);
      setCurrentPromptIdx(0);
      setConsentGiven(false);

      if (onEnrollComplete) onEnrollComplete();
    } catch (err: any) {
      setStatusMessage({ type: 'error', text: err.message || 'Enrollment failed.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <UserPlus className="w-6 h-6 text-sky-500" />
          Zero-Retraining Biometric Enrollment Wizard
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Enrolls new individuals directly into the deep metric vector index in &lt;1 minute without model retraining.
        </p>
      </div>

      {statusMessage && (
        <div
          className={`mb-6 p-4 rounded-2xl flex items-center gap-3 border ${
            statusMessage.type === 'success'
              ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-500/30 text-emerald-800 dark:text-emerald-300'
              : 'bg-rose-50 dark:bg-rose-950/40 border-rose-500/30 text-rose-800 dark:text-rose-300'
          }`}
        >
          {statusMessage.type === 'success' ? (
            <Check className="w-5 h-5 flex-shrink-0" />
          ) : (
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
          )}
          <span className="text-sm font-medium">{statusMessage.text}</span>
        </div>
      )}

      <form onSubmit={handleSubmitEnrollment} className="space-y-6">
        {/* Step 1: Personal Info */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-4 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">1. Identity Information</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Full Name *
              </label>
              <input
                type="text"
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="e.g. Dr. Jane Doe"
                className="w-full px-4 py-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white text-sm focus:ring-2 focus:ring-sky-500 outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                ID / Roll Number
              </label>
              <input
                type="text"
                value={externalId}
                onChange={(e) => setExternalId(e.target.value)}
                placeholder="e.g. STU-2026-042"
                className="w-full px-4 py-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white text-sm focus:ring-2 focus:ring-sky-500 outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
              Department / Faculty
            </label>
            <input
              type="text"
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
              placeholder="e.g. Computer Science"
              className="w-full px-4 py-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white text-sm focus:ring-2 focus:ring-sky-500 outline-none"
            />
          </div>

          {/* Privacy Consent Checkbox */}
          <div className="pt-2">
            <label className="flex items-start gap-3 cursor-pointer p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700">
              <input
                type="checkbox"
                required
                checked={consentGiven}
                onChange={(e) => setConsentGiven(e.target.checked)}
                className="mt-1 h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500"
              />
              <span className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                <strong className="text-slate-900 dark:text-white">Biometric Consent:</strong> I explicitly consent to the capture of my facial features converted into irreversible mathematical embeddings for automated attendance marking, in compliance with India's DPDP Act 2023. I understand I have the right to request erasure at any time.
              </span>
            </label>
          </div>
        </div>

        {/* Step 2: Guided Camera Capture */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-4 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
              2. Guided Multi-Angle Capture ({capturedBlobs.length}/3)
            </h3>
            {capturedBlobs.length > 0 && (
              <button
                type="button"
                onClick={handleReset}
                className="text-xs text-rose-500 hover:text-rose-600 flex items-center gap-1 font-medium"
              >
                <RotateCcw className="w-3.5 h-3.5" /> Retake Photos
              </button>
            )}
          </div>

          {/* Current Prompt Indicator */}
          {capturedBlobs.length < 3 && (
            <div className="p-4 rounded-2xl bg-sky-50 dark:bg-sky-950/40 border border-sky-500/20 flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-sky-600 dark:text-sky-400 uppercase tracking-wide">
                  Step {CAPTURE_PROMPTS[currentPromptIdx].step} of 3: {CAPTURE_PROMPTS[currentPromptIdx].title}
                </p>
                <p className="text-sm text-slate-700 dark:text-slate-200 mt-0.5">
                  {CAPTURE_PROMPTS[currentPromptIdx].desc}
                </p>
              </div>
              <button
                type="button"
                onClick={takeSnapshot}
                className="px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-600 text-white font-semibold text-xs flex items-center gap-2 shadow-md transition"
              >
                <Camera className="w-4 h-4" /> Capture Angle
              </button>
            </div>
          )}

          {/* Video Preview */}
          <div className="relative rounded-2xl overflow-hidden bg-slate-950 aspect-[4/3] max-w-md mx-auto border border-slate-800">
            <video
              ref={videoRef}
              playsInline
              muted
              className="w-full h-full object-cover transform -scale-x-100"
            />
            {/* Guide overlay */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <div className="w-48 h-60 rounded-[50%] border-2 border-dashed border-sky-400/60" />
            </div>
          </div>

          {/* Captured Thumbnails */}
          {capturedThumbnails.length > 0 && (
            <div className="grid grid-cols-3 gap-3 pt-2">
              {capturedThumbnails.map((src, i) => (
                <div key={i} className="relative rounded-xl overflow-hidden border border-slate-700 aspect-square">
                  <img src={src} alt={`Angle ${i + 1}`} className="w-full h-full object-cover" />
                  <span className="absolute bottom-1 right-1 bg-black/70 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
                    Angle {i + 1}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isSubmitting || capturedBlobs.length < 3 || !consentGiven || !fullName}
          className="w-full py-3.5 rounded-2xl bg-sky-500 hover:bg-sky-600 disabled:opacity-50 text-white font-bold text-sm flex items-center justify-center gap-2 shadow-lg transition"
        >
          {isSubmitting ? (
            'Processing Vector Prototypes...'
          ) : (
            <>
              <ShieldCheck className="w-5 h-5" /> Complete Enrollment &amp; Register Prototype
            </>
          )}
        </button>
      </form>
    </div>
  );
};
