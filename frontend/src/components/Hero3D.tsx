import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export type ScanState = 'idle' | 'scanning' | 'success' | 'unknown';

interface Hero3DProps {
  scanState?: ScanState;
}

export const Hero3D: React.FC<Hero3DProps> = ({ scanState = 'idle' }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [webglSupported, setWebglSupported] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    // Check reduced motion
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(motionQuery.matches);
    const motionHandler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    motionQuery.addEventListener('change', motionHandler);
    return () => motionQuery.removeEventListener('change', motionHandler);
  }, []);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // Check WebGL availability
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) {
        setWebglSupported(false);
        return;
      }
    } catch {
      setWebglSupported(false);
      return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 4.5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 1. Procedural 3D Particle Face-Mesh
    const particleCount = 1800;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    // Generate an anthropomorphic face-head shape using parametric ellipsoid + facial depressions
    for (let i = 0; i < particleCount; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);

      // Anatomical proportions (height 1.4, width 1.0, depth 1.1)
      let r = 1.0;
      let x = r * Math.sin(phi) * Math.cos(theta) * 0.95;
      let y = r * Math.cos(phi) * 1.35;
      let z = r * Math.sin(phi) * Math.sin(theta) * 1.05;

      // Eye socket hollows and nose ridge indentation
      if (z > 0 && y > 0 && y < 0.4) {
        if (Math.abs(x) > 0.2 && Math.abs(x) < 0.6) {
          z *= 0.82; // Eye depressions
        } else if (Math.abs(x) < 0.15) {
          z *= 1.25; // Nose bridge projection
        }
      }

      // Chin taper
      if (y < -0.3) {
        const taper = 1.0 + (y + 0.3) * 0.6;
        x *= Math.max(0.4, taper);
        z *= Math.max(0.5, taper);
      }

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      // Default Cyan-Blue particle color
      colors[i * 3] = 0.05;
      colors[i * 3 + 1] = 0.65;
      colors[i * 3 + 2] = 0.95;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.045,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
    });

    const particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);

    // 2. Outer Biometric Orbital Ring
    const ringGeo = new THREE.TorusGeometry(1.9, 0.008, 16, 100);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.4,
      wireframe: true,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 4;
    scene.add(ring);

    // 3. Scanning Laser Line
    const scanPlaneGeo = new THREE.RingGeometry(0.01, 1.8, 64);
    const scanPlaneMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
    });
    const scanPlane = new THREE.Mesh(scanPlaneGeo, scanPlaneMat);
    scanPlane.rotation.x = Math.PI / 2;
    scene.add(scanPlane);

    // Animation Loop
    let animationFrameId: number;
    let clock = new THREE.Clock();

    const render = () => {
      const elapsed = clock.getElapsedTime();

      if (!reducedMotion) {
        // State-reactive dynamics
        if (scanState === 'scanning') {
          particleSystem.rotation.y = Math.sin(elapsed * 1.5) * 0.35;
          ring.rotation.z += 0.02;
          // Oscillate scanning laser plane
          scanPlane.position.y = Math.sin(elapsed * 4.0) * 1.2;
          scanPlaneMat.opacity = 0.35;
          material.color.setHex(0x38bdf8); // Cyan
        } else if (scanState === 'success') {
          particleSystem.rotation.y = Math.sin(elapsed * 0.5) * 0.15;
          ring.rotation.z += 0.005;
          scanPlaneMat.opacity = 0.0;
          material.color.setHex(0x10b981); // Emerald Green
        } else if (scanState === 'unknown') {
          particleSystem.rotation.y = Math.sin(elapsed * 4.0) * 0.25; // Perturbation shake
          ring.rotation.z += 0.01;
          scanPlaneMat.opacity = 0.0;
          material.color.setHex(0xf59e0b); // Amber Alert
        } else {
          // Idle state
          particleSystem.rotation.y = Math.sin(elapsed * 0.5) * 0.25;
          particleSystem.rotation.x = Math.cos(elapsed * 0.3) * 0.08;
          ring.rotation.z += 0.004;
          scanPlaneMat.opacity = 0.0;
          material.color.setHex(0x0284c7);
        }
      }

      renderer.render(scene, camera);
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    // Resize Observer
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      ringGeo.dispose();
      ringMat.dispose();
      renderer.dispose();
    };
  }, [scanState, reducedMotion]);

  if (!webglSupported) {
    return (
      <div className="w-full h-full flex items-center justify-center p-8 text-center text-slate-400">
        <div className="border border-slate-700 rounded-2xl p-6 bg-slate-900/40">
          <p className="text-sm font-medium">3D WebGL Visualization Fallback</p>
          <p className="text-xs text-slate-500 mt-1">Biometric Attendance active in standard 2D mode</p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={mountRef}
      className="w-full h-full min-h-[380px] cursor-grab active:cursor-grabbing relative"
      aria-label="3D Biometric Face Mesh Visualizer"
    />
  );
};
