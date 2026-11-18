import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, CheckCircle2, AlertTriangle, ShieldAlert, Sparkles, Volume2, VolumeX } from 'lucide-react';
import { api } from '../services/api';

export const LiveKiosk: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [fps, setFps] = useState(0);
  const [lastAttendance, setLastAttendance] = useState<{ name: string; timestamp: string } | null>(null);
  const [detectedCount, setDetectedCount] = useState(0);

  // Play audio chime using Web Audio API
  const playSuccessChime = useCallback(() => {
    if (!soundEnabled) return;
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
      osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.15); // A5

      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + 0.4);
    } catch {
      // AudioContext may be restricted by browser policy before first user gesture
    }
  }, [soundEnabled]);

  // Start webcam
  useEffect(() => {
    let stream: MediaStream | null = null;

    async function initCamera() {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error('Camera access requires HTTPS or localhost in your browser.');
        }

        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setIsStreaming(true);
        }
      } catch (err: any) {
        setCameraError(err.message || 'Failed to access webcam.');
      }
    }

    initCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Frame Capture & WebSocket Loop (~5 FPS)
  useEffect(() => {
    if (!isStreaming) return;

    let frameCount = 0;
    let lastFpsTime = performance.now();
    let intervalId: any;

    // Connect WebSocket
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host;
    const wsUrl = `${wsProto}//${wsHost}/ws/stream`;

    try {
      const socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          renderBoundingBoxes(data.faces || []);
          setDetectedCount(data.faces_detected || 0);

          if (data.attendance_event?.event === 'ATTENDANCE_SUCCESS') {
            setLastAttendance({
              name: data.attendance_event.person_name,
              timestamp: new Date().toLocaleTimeString(),
            });
            playSuccessChime();
          }
        } catch {
          // Ignore parse errors
        }
      };

      socket.onerror = () => {
        // Fall back to REST polling if WebSocket unavailable
      };
    } catch {
      // WebSocket setup error
    }

    // Capture & Send Frame
    intervalId = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

      const offscreen = document.createElement('canvas');
      offscreen.width = video.videoWidth || 640;
      offscreen.height = video.videoHeight || 480;
      const ctx = offscreen.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
      const b64 = offscreen.toDataURL('image/jpeg', 0.7);

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ frame: b64, session: 'Main' }));
      } else {
        // Fallback REST call
        try {
          const res = await api.recognizeFrame(b64, 'Main', true);
          renderBoundingBoxes(res.faces);
          setDetectedCount(res.faces_detected);
          if (res.attendance_marked && res.marked_person_name) {
            setLastAttendance({
              name: res.marked_person_name,
              timestamp: new Date().toLocaleTimeString(),
            });
            playSuccessChime();
          }
        } catch {
          // Skip frame error
        }
      }

      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastFpsTime = now;
      }
    }, 200); // 5 FPS

    return () => {
      clearInterval(intervalId);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isStreaming, playSuccessChime]);

  // Canvas Bounding Box Rendering
  const renderBoundingBoxes = (faces: any[]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / (video.videoWidth || 640);
    const scaleY = canvas.height / (video.videoHeight || 480);

    faces.forEach((f) => {
      const [x, y, w, h] = f.bbox;
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledW = w * scaleX;
      const scaledH = h * scaleY;

      // Color coding: Green for Known Live, Red for Spoof, Yellow for Unknown
      let strokeColor = '#f59e0b'; // Yellow (Unknown)
      let label = 'Unknown Person';

      if (!f.is_live) {
        strokeColor = '#ef4444'; // Red (Spoof)
        label = `SPOOF ALERT (${f.spoof_type || 'FAKE'})`;
      } else if (f.is_known) {
        strokeColor = '#10b981'; // Green (Verified)
        label = `${f.person_name} (${(f.confidence * 100).toFixed(0)}%)`;
      }

      // Draw bounding box
      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

      // Draw corner accents
      const cornerLen = 12;
      ctx.lineWidth = 4;
      ctx.beginPath();
      // Top left
      ctx.moveTo(scaledX, scaledY + cornerLen);
      ctx.lineTo(scaledX, scaledY);
      ctx.lineTo(scaledX + cornerLen, scaledY);
      // Top right
      ctx.moveTo(scaledX + scaledW - cornerLen, scaledY);
      ctx.lineTo(scaledX + scaledW, scaledY);
      ctx.lineTo(scaledX + scaledW, scaledY + cornerLen);
      ctx.stroke();

      // Label background & text
      ctx.font = 'bold 13px -apple-system, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = strokeColor;
      ctx.fillRect(scaledX, Math.max(0, scaledY - 24), textWidth + 14, 24);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, scaledX + 7, Math.max(16, scaledY - 7));
    });
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Camera className="w-6 h-6 text-sky-500" />
            Live Attendance Kiosk
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Real-time biometric attendance with anti-spoof presentation attack detection.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            {fps} FPS
          </div>

          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="p-2 rounded-xl bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-700 transition"
            title={soundEnabled ? 'Mute sound' : 'Enable audio chime'}
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Video Viewport */}
      <div className="relative rounded-3xl overflow-hidden bg-slate-900 shadow-2xl border border-slate-800 aspect-[4/3] flex items-center justify-center">
        {cameraError ? (
          <div className="text-center p-8 max-w-md">
            <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto mb-3" />
            <p className="text-white font-semibold mb-1">Camera Inaccessible</p>
            <p className="text-sm text-slate-400 mb-4">{cameraError}</p>
            <p className="text-xs text-slate-500">
              Note: Webcam access requires a secure context (HTTPS) or localhost.
            </p>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              playsInline
              muted
              className="w-full h-full object-cover transform -scale-x-100"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100"
            />

            {/* Overlays */}
            <div className="absolute top-4 left-4 flex gap-2">
              <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-black/60 text-emerald-400 border border-emerald-500/30 backdrop-blur-md">
                Liveness Protected
              </span>
              {detectedCount > 0 && (
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-black/60 text-white border border-white/20 backdrop-blur-md">
                  {detectedCount} Face{detectedCount > 1 ? 's' : ''}
                </span>
              )}
            </div>

            {/* Attendance Toast Notification */}
            {lastAttendance && (
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-emerald-500/90 text-white px-6 py-3 rounded-2xl shadow-xl backdrop-blur-md flex items-center gap-3 border border-emerald-400/50 animate-bounce">
                <CheckCircle2 className="w-5 h-5 text-white" />
                <div>
                  <p className="font-bold text-sm">Attendance Recorded!</p>
                  <p className="text-xs text-emerald-100">
                    {lastAttendance.name} at {lastAttendance.timestamp}
                  </p>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Legend & Instructions */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
        <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-emerald-500" />
          <div>
            <p className="text-xs font-semibold text-slate-900 dark:text-white">Live Recognized</p>
            <p className="text-xs text-slate-500">Confidence above threshold</p>
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-amber-500" />
          <div>
            <p className="text-xs font-semibold text-slate-900 dark:text-white">Unknown Face</p>
            <p className="text-xs text-slate-500">Requires registration in system</p>
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-rose-500" />
          <div>
            <p className="text-xs font-semibold text-slate-900 dark:text-white">Anti-Spoof Block</p>
            <p className="text-xs text-slate-500">Screen/print attacks rejected</p>
          </div>
        </div>
      </div>
    </div>
  );
};
