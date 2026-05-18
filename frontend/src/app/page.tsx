"use client";

/**
 * DEEPVAULT SECURE TERMINAL
 * Hyper-futuristic dashboard interface for deepfake image analysis.
 * Supports standard local file ingestion and live camera feed analysis.
 */

import { useState, useRef, useEffect } from "react";
import { FileUpload } from "@/components/ui/file-upload";

// Interface for type-safe prediction results
interface PredictionResult {
  confidence: number;
  is_fake: boolean;
  grad_cam_url: string | null;
  processing_time_ms: number;
}

export default function Home() {
  // Application Data States
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Interface/Webcam States
  const [isWebcamMode, setIsWebcamMode] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [webcamError, setWebcamError] = useState<string | null>(null);

  // References to video & hidden canvas for snapshot extraction
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Automatically shut down webcam stream when switching modes or unmounting
  useEffect(() => {
    return () => {
      if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [webcamStream]);

  // Launches webcam hardware capture
  const startWebcam = async () => {
    setWebcamError(null);
    setIsWebcamActive(true);
    setResult(null);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 640, facingMode: "user" },
        audio: false,
      });
      setWebcamStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err: any) {
      console.error("Error accessing hardware camera:", err);
      setWebcamError("Camera access denied or hardware not found.");
      setIsWebcamActive(false);
    }
  };

  // Stops webcam streaming and releases hardware lock
  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      setWebcamStream(null);
    }
    setIsWebcamActive(false);
  };

  // Grabs a frame from the live video feed and triggers Deep Learning analysis
  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      if (context) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 640;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(
          (blob) => {
            if (blob) {
              const file = new File([blob], "live_capture.jpg", { type: "image/jpeg" });
              setSelectedFile(file);
              setPreviewUrl(URL.createObjectURL(blob));
              stopWebcam();
              processImage(file);
            }
          },
          "image/jpeg",
          0.95
        );
      }
    }
  };

  // Ingests local dropped or uploaded files
  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    processImage(file);
  };

  // Sends the processed image (from webcam or file upload) to the FastAPI server
  const processImage = async (file: File) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/api/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Analysis failed. Server returned an invalid response code.");
      }

      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to establish gateway connection with server.");
    } finally {
      setIsLoading(false);
    }
  };

  // Toggles scanner modes and ensures webcam shuts down cleanly
  const toggleMode = (webcamMode: boolean) => {
    setIsWebcamMode(webcamMode);
    if (!webcamMode) {
      stopWebcam();
    }
    setResult(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-[#030307] text-[#e0e0ea] p-4 md:p-12 font-mono relative overflow-hidden selection:bg-[#00f3ff]/30 selection:text-white">
      {/* High-tech grid background overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f293708_1px,transparent_1px),linear-gradient(to_bottom,#1f293708_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />
      


      <div className="max-w-6xl mx-auto space-y-8 relative z-10">
        
        {/* Terminal Header */}
        <header className="border-b border-[#1b1b3a] pb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <span className="h-2 w-2 rounded-full bg-[#00f3ff] animate-ping" />
              <p className="text-xs uppercase tracking-[0.25em] text-[#00f3ff]">Deep Neural Diagnostics</p>
            </div>
            <h1 className="text-2xl md:text-3xl font-black tracking-wider text-white">
              DEEPVAULT // ANALYSIS TERMINAL
            </h1>
          </div>
          <div className="flex items-center gap-4 text-xs text-[#6e6e8a] border border-[#1b1b3a] bg-[#070714] px-4 py-2 rounded-md">
            <div>
              <span className="text-[#a855f7]">GATEWAY:</span>{" "}
              <span className="text-gray-300 font-bold">ONLINE</span>
            </div>
            <div className="h-3 w-[1px] bg-[#1b1b3a]" />
            <div>
              <span className="text-[#00f3ff]">CORE:</span>{" "}
              <span className="text-gray-300 font-bold">SGAN-RES50</span>
            </div>
          </div>
        </header>

        {/* Diagnostic Grid Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Block: Scanner Input Console (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Input Mode Selector */}
            <div className="bg-[#070714] border border-[#1b1b3a] rounded-xl p-2 flex gap-2">
              <button
                onClick={() => toggleMode(false)}
                className={`flex-1 py-3 text-xs uppercase tracking-wider rounded-lg transition-all flex items-center justify-center gap-2 font-bold ${
                  !isWebcamMode
                    ? "bg-[#00f3ff]/10 text-[#00f3ff] border border-[#00f3ff]/20 shadow-[0_0_15px_rgba(0,243,255,0.05)]"
                    : "text-[#6e6e8a] hover:text-[#a0a0c0] hover:bg-white/5"
                }`}
              >
                <svg suppressHydrationWarning={true} className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Upload File
              </button>
              <button
                onClick={() => toggleMode(true)}
                className={`flex-1 py-3 text-xs uppercase tracking-wider rounded-lg transition-all flex items-center justify-center gap-2 font-bold ${
                  isWebcamMode
                    ? "bg-[#a855f7]/10 text-[#a855f7] border border-[#a855f7]/20 shadow-[0_0_15px_rgba(168,85,247,0.05)]"
                    : "text-[#6e6e8a] hover:text-[#a0a0c0] hover:bg-white/5"
                }`}
              >
                <svg suppressHydrationWarning={true} className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Live Camera
              </button>
            </div>

            {/* Ingestion Panel */}
            <div className="bg-[#070714] border border-[#1b1b3a] rounded-xl p-6 shadow-2xl space-y-6 relative overflow-hidden">
              
              {!isWebcamMode ? (
                // File Upload View
                <div className="space-y-4">
                  <div>
                    <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-1">Select Input Source</h2>
                    <p className="text-xs text-[#6e6e8a]">Ingest compressed raster files for cryptographic GAN analysis.</p>
                  </div>
                  <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />
                </div>
              ) : (
                // Live Webcam Scanner View
                <div className="space-y-4">
                  <div>
                    <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-1">Live Lens Scanner</h2>
                    <p className="text-xs text-[#6e6e8a]">Capture live real-time high-resolution frames directly from source hardware.</p>
                  </div>

                  {isWebcamActive ? (
                    <div className="relative aspect-square w-full bg-black rounded-lg overflow-hidden border border-[#a855f7]/30">
                      
                      {/* Video Stream Element */}
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="object-cover w-full h-full transform scale-x-[-1]"
                      />

                      {/* Sci-Fi Target HUD Lines */}
                      <div className="absolute inset-4 pointer-events-none border border-white/5 flex flex-col justify-between">
                        <div className="flex justify-between">
                          <div className="h-4 w-4 border-l-2 border-t-2 border-[#a855f7]" />
                          <div className="h-4 w-4 border-r-2 border-t-2 border-[#a855f7]" />
                        </div>
                        
                        {/* Interactive crosshair */}
                        <div className="self-center flex items-center justify-center">
                          <div className="h-8 w-8 border border-dashed border-[#a855f7]/35 rounded-full absolute" />
                          <div className="h-[1px] w-4 bg-[#a855f7]/40" />
                          <div className="h-4 w-[1px] bg-[#a855f7]/40 absolute" />
                        </div>

                        <div className="flex justify-between">
                          <div className="h-4 w-4 border-l-2 border-b-2 border-[#a855f7]" />
                          <div className="h-4 w-4 border-r-2 border-b-2 border-[#a855f7]" />
                        </div>
                      </div>

                      {/* Moving laser scan line */}
                      <div className="scanline absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#a855f7] to-transparent shadow-[0_0_10px_#a855f7] pointer-events-none" />

                      {/* Capture Trigger Button */}
                      <div className="absolute bottom-4 left-4 right-4 flex gap-2">
                        <button
                          onClick={captureFrame}
                          disabled={isLoading}
                          className="flex-1 py-3 bg-[#a855f7] text-black hover:bg-[#b966ff] disabled:bg-gray-800 disabled:text-gray-500 font-bold rounded-lg text-xs uppercase tracking-wider transition-all shadow-[0_0_20px_rgba(168,85,247,0.3)] hover:shadow-[0_0_30px_rgba(168,85,247,0.5)]"
                        >
                          Analyze Frame
                        </button>
                        <button
                          onClick={stopWebcam}
                          className="px-4 bg-[#1b1b3a] hover:bg-[#252550] font-bold text-white rounded-lg text-xs uppercase tracking-wider transition-all"
                        >
                          Abort
                        </button>
                      </div>
                    </div>
                  ) : (
                    // Camera Offline / Start Trigger
                    <div className="border border-dashed border-[#1b1b3a] rounded-lg p-10 flex flex-col items-center justify-center text-center space-y-4 hover:border-[#a855f7]/40 transition-colors">
                      <div className="p-4 bg-[#a855f7]/5 rounded-full text-[#a855f7] border border-[#a855f7]/10">
                        <svg suppressHydrationWarning={true} className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                        </svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-bold text-gray-300">Live Camera Inactive</p>
                        <p className="text-xs text-[#6e6e8a] max-w-xs">Initialize local camera hardware to capture instant snapshot data.</p>
                      </div>
                      {webcamError && (
                        <p className="text-xs text-[#ff003c] border border-[#ff003c]/20 bg-[#ff003c]/5 px-3 py-2 rounded">
                          {webcamError}
                        </p>
                      )}
                      <button
                        onClick={startWebcam}
                        className="py-3 px-6 bg-[#a855f7]/10 hover:bg-[#a855f7]/20 border border-[#a855f7]/30 hover:border-[#a855f7] text-[#a855f7] font-bold rounded-lg text-xs uppercase tracking-wider transition-all"
                      >
                        Initialize Video Lens
                      </button>
                    </div>
                  )}
                </div>
              )}

              {error && (
                <div className="p-4 bg-[#ff003c]/10 border border-[#ff003c]/30 rounded-xl text-[#ff003c] text-xs leading-relaxed">
                  <div className="font-bold uppercase tracking-wider mb-1">SYSTEM EXCEPTION</div>
                  {error}
                </div>
              )}

              {/* Hidden Canvas used to process camera frame */}
              <canvas ref={canvasRef} className="hidden" />
            </div>
          </div>

          {/* Right Block: Diagnostics & Grad-CAM Analysis Report (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            {previewUrl ? (
              <div className="bg-[#070714] border border-[#1b1b3a] rounded-xl p-6 shadow-2xl space-y-6 flex flex-col min-h-[460px]">
                <div>
                  <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-1">Diagnostic Report</h2>
                  <p className="text-xs text-[#6e6e8a]">Complete computational artifact details and network verification statistics.</p>
                </div>

                {/* Main Visuals Display Side-by-Side */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1">
                  
                  {/* Left: Input Image Preview */}
                  <div className="space-y-2">
                    <p className="text-xs uppercase tracking-wider text-[#6e6e8a]">Base Input Capture</p>
                    <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-black/60 border border-[#1b1b3a] flex items-center justify-center">
                      <img src={previewUrl} alt="Base input" className="object-contain w-full h-full" />
                    </div>
                  </div>

                  {/* Right: Grad-CAM Explainability Map */}
                  <div className="space-y-2">
                    <p className="text-xs uppercase tracking-wider text-[#00f3ff]">Deepfake Artifact Map</p>
                    <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-black/60 border border-[#00f3ff]/20 flex items-center justify-center shadow-[0_0_20px_rgba(0,243,255,0.02)]">
                      {isLoading ? (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/85 z-10 space-y-3">
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#00f3ff]"></div>
                          <p className="text-xs uppercase tracking-[0.15em] text-[#00f3ff] animate-pulse">Running Neural Scan...</p>
                        </div>
                      ) : null}

                      {result && result.grad_cam_url ? (
                        <img src={result.grad_cam_url} alt="Artifact map" className="object-contain w-full h-full" />
                      ) : (
                        <div className="text-center p-6 text-xs text-[#6e6e8a] flex flex-col items-center justify-center h-full">
                          <svg suppressHydrationWarning={true} className="h-10 w-10 mb-3 opacity-40 animate-pulse text-[#00f3ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                          </svg>
                          <p>Waiting for deep neural processing to complete.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Numerical Statistics Summary */}
                {result && !isLoading && (
                  <div className="space-y-4">
                    {/* Status Alert Banner */}
                    <div
                      className={`p-4 rounded-lg border flex items-center justify-between transition-all ${
                        result.is_fake
                          ? "bg-[#ff003c]/5 border-[#ff003c]/25 shadow-[0_0_15px_rgba(255,0,60,0.03)]"
                          : "bg-[#00ff66]/5 border-[#00ff66]/25 shadow-[0_0_15px_rgba(0,255,102,0.03)]"
                      }`}
                    >
                      <div className="space-y-0.5">
                        <p className="text-[10px] uppercase tracking-wider text-gray-500">Security Classification</p>
                        <p className={`text-lg font-black tracking-wider ${result.is_fake ? 'text-[#ff003c]' : 'text-[#00ff66]'}`}>
                          {result.is_fake ? "FORGERY // DETECTED" : "VERIFIED // AUTHENTIC"}
                        </p>
                      </div>
                      <div className={`px-4 py-2 rounded text-[10px] font-bold tracking-widest border uppercase ${
                        result.is_fake ? 'border-[#ff003c]/40 text-[#ff003c] bg-[#ff003c]/5' : 'border-[#00ff66]/40 text-[#00ff66] bg-[#00ff66]/5'
                      }`}>
                        {result.is_fake ? "MALICIOUS" : "SECURE"}
                      </div>
                    </div>

                    {/* Technical Diagnostic Grid */}
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      
                      <div className="p-4 bg-[#0a0a1a] border border-[#1b1b3a] rounded-lg">
                        <p className="text-[10px] uppercase tracking-wider text-[#6e6e8a] mb-1">Verdict Score</p>
                        <p className={`text-xl font-bold ${result.is_fake ? 'text-[#ff003c]' : 'text-[#00ff66]'}`}>
                          {(result.confidence * 100).toFixed(2)}%
                        </p>
                        <p className="text-[9px] text-[#4b4b66] mt-0.5">Discriminator probability</p>
                      </div>

                      <div className="p-4 bg-[#0a0a1a] border border-[#1b1b3a] rounded-lg">
                        <p className="text-[10px] uppercase tracking-wider text-[#6e6e8a] mb-1">Scan Latency</p>
                        <p className="text-xl font-bold text-white">
                          {result.processing_time_ms.toFixed(0)} ms
                        </p>
                        <p className="text-[9px] text-[#4b4b66] mt-0.5">PyTorch execution time</p>
                      </div>

                      <div className="p-4 bg-[#0a0a1a] border border-[#1b1b3a] rounded-lg">
                        <p className="text-[10px] uppercase tracking-wider text-[#6e6e8a] mb-1">Explainability</p>
                        <p className="text-xl font-bold text-[#00f3ff]">
                          {result.grad_cam_url ? "ACTIVE" : "INACTIVE"}
                        </p>
                        <p className="text-[9px] text-[#4b4b66] mt-0.5">Grad-CAM resolution maps</p>
                      </div>

                    </div>
                  </div>
                )}

              </div>
            ) : (
              // Empty State HUD display when no image is loaded
              <div className="bg-[#070714] border border-[#1b1b3a] rounded-xl p-6 shadow-2xl min-h-[460px] flex flex-col items-center justify-center text-center text-[#6e6e8a] relative">
                {/* HUD Cybernetic decorative targets */}
                <div className="absolute top-4 left-4 h-3 w-3 border-l border-t border-[#1b1b3a]" />
                <div className="absolute top-4 right-4 h-3 w-3 border-r border-t border-[#1b1b3a]" />
                <div className="absolute bottom-4 left-4 h-3 w-3 border-l border-b border-[#1b1b3a]" />
                <div className="absolute bottom-4 right-4 h-3 w-3 border-r border-b border-[#1b1b3a]" />
                
                <svg suppressHydrationWarning={true} className="h-16 w-16 mb-4 text-[#1b1b3a]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <p className="text-sm uppercase tracking-[0.2em] font-bold text-gray-500 mb-1">Ingest Frame Target</p>
                <p className="text-xs max-w-sm">Provide an input asset from your system files or start the live camera lens scanning framework to initiate analysis.</p>
              </div>
            )}
          </div>

        </div>

      </div>
    </main>
  );
}
