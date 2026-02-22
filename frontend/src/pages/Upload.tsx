// =============================================================================
// AgriLite-Hybrid Frontend
// src/pages/Upload.tsx - Image Upload Page
// 
// Allows user to upload an image from file or capture using camera.
// Supports drag-and-drop and camera capture via getUserMedia.
// =============================================================================

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Upload as UploadIcon,
  Camera,
  X,
  Image as ImageIcon,
  ArrowLeft,
  ArrowRight,
  Loader2,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import { useDetection } from '../contexts/DetectionContext';
import toast from 'react-hot-toast';

const Upload: React.FC = () => {
  const navigate = useNavigate();
  const {
    selectedCrop,
    imageFile,
    imagePreview,
    setImage,
    clearImage,
    analyzeImage,
    isAnalyzing,
  } = useDetection();

  // Camera state
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Drag and drop state
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ---------------------------------------------------------------------------
  // Redirect if no crop selected
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!selectedCrop) {
      navigate('/select-crop');
    }
  }, [selectedCrop, navigate]);

  // ---------------------------------------------------------------------------
  // Clean up camera stream on unmount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Attach stream to video element when camera is active
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (isCameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [isCameraActive]);

  // ---------------------------------------------------------------------------
  // File Selection Handler
  // ---------------------------------------------------------------------------
  const handleFileSelect = useCallback(
    (file: File) => {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
      if (!validTypes.includes(file.type)) {
        toast.error('Please upload a valid image (JPEG, PNG, or WebP)');
        return;
      }

      // Validate file size (max 10MB)
      const maxSize = 10 * 1024 * 1024;
      if (file.size > maxSize) {
        toast.error('Image size must be less than 10MB');
        return;
      }

      setImage(file);
    },
    [setImage]
  );

  // ---------------------------------------------------------------------------
  // File Input Handler
  // ---------------------------------------------------------------------------
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // ---------------------------------------------------------------------------
  // Drag and Drop Handlers
  // ---------------------------------------------------------------------------
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // ---------------------------------------------------------------------------
  // Camera Handlers
  // ---------------------------------------------------------------------------
  const startCamera = async (mode: 'environment' | 'user' = facingMode) => {
    setCameraError(null);
    
    // Stop existing stream first
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    
    try {
      // Try with exact facingMode first (back camera)
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: { exact: mode }, 
            width: { ideal: 1280 }, 
            height: { ideal: 720 } 
          },
        });
      } catch {
        // Fallback without exact constraint if device doesn't support it
        stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: mode, 
            width: { ideal: 1280 }, 
            height: { ideal: 720 } 
          },
        });
      }

      streamRef.current = stream;
      setFacingMode(mode);
      setIsCameraActive(true);
      
      // Wait for state update and DOM render, then attach stream
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(console.error);
        }
      }, 100);
    } catch (err) {
      console.error('Camera error:', err);
      setCameraError('Could not access camera. Please check permissions or use file upload.');
      toast.error('Could not access camera');
    }
  };

  const flipCamera = () => {
    const newMode = facingMode === 'environment' ? 'user' : 'environment';
    startCamera(newMode);
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
    setCameraError(null);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      // Set canvas dimensions to video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw video frame to canvas
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);

        // Convert canvas to blob
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const file = new File([blob], `capture_${Date.now()}.jpg`, {
                type: 'image/jpeg',
              });
              handleFileSelect(file);
              stopCamera();
            }
          },
          'image/jpeg',
          0.9
        );
      }
    }
  };

  // ---------------------------------------------------------------------------
  // Analysis Handler
  // ---------------------------------------------------------------------------
  const handleAnalyze = async () => {
    const success = await analyzeImage();
    if (success) {
      navigate('/result');
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  if (!selectedCrop) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 sm:py-12">
      {/* Back Link */}
      <Link
        to="/select-crop"
        className="inline-flex items-center text-gray-600 hover:text-primary-600 mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Change crop
      </Link>

      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 mb-3">
          Upload {selectedCrop.name} Image
        </h1>
        <p className="text-lg text-gray-600">
          Take a photo or upload an image of the affected leaf for analysis.
        </p>
      </div>

      {/* Hidden canvas for camera capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Main Content */}
      <div className="space-y-6">
        {/* Camera View */}
        {isCameraActive && (
          <div className="card p-4 sm:p-6">
            <div className="relative rounded-xl overflow-hidden bg-gray-900 shadow-inner">
              {/* Camera indicator */}
              <div className="absolute top-4 left-4 z-10 flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                <span className="text-white text-sm font-medium bg-black/50 px-2 py-1 rounded">
                  {facingMode === 'environment' ? 'Back Camera' : 'Front Camera'}
                </span>
              </div>
              
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                onLoadedMetadata={(e) => (e.target as HTMLVideoElement).play()}
                className="w-full object-cover rounded-lg"
                style={{ 
                  minHeight: '400px', 
                  maxHeight: '70vh',
                  transform: facingMode === 'user' ? 'scaleX(-1)' : 'none'
                }}
              />
              
              {/* Camera Controls */}
              <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent">
                <div className="flex justify-center items-center space-x-4">
                  <button
                    onClick={flipCamera}
                    className="p-3 bg-white/20 backdrop-blur-sm text-white rounded-full hover:bg-white/30 transition-colors"
                    title="Flip camera"
                  >
                    <RefreshCw className="w-6 h-6" />
                  </button>
                  <button
                    onClick={capturePhoto}
                    className="p-4 bg-white text-gray-900 rounded-full hover:bg-gray-100 shadow-lg transition-transform hover:scale-105"
                  >
                    <Camera className="w-8 h-8" />
                  </button>
                  <button
                    onClick={stopCamera}
                    className="p-3 bg-red-500/80 backdrop-blur-sm text-white rounded-full hover:bg-red-600 transition-colors"
                    title="Cancel"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Image Preview */}
        {imagePreview && !isCameraActive && (
          <div className="card p-6">
            <div className="relative">
              <img
                src={imagePreview}
                alt="Selected leaf"
                className="w-full max-h-96 object-contain rounded-lg mx-auto"
              />
              <button
                onClick={clearImage}
                className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 shadow-lg transition-colors"
                title="Remove image"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="mt-4 text-center text-sm text-gray-600">
              {imageFile?.name}
              <span className="mx-2">•</span>
              {imageFile && (imageFile.size / 1024 / 1024).toFixed(2)} MB
            </div>
          </div>
        )}

        {/* Upload Area */}
        {!imagePreview && !isCameraActive && (
          <div className="grid gap-6 md:grid-cols-2">
            {/* File Upload */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className={`
                drop-zone flex flex-col items-center justify-center p-8 min-h-64 cursor-pointer
                ${isDragging ? 'drag-over' : ''}
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/jpg,image/webp"
                onChange={handleFileInputChange}
                className="hidden"
              />
              <div className="p-4 bg-primary-100 rounded-full mb-4">
                <UploadIcon className="w-8 h-8 text-primary-600" />
              </div>
              <p className="text-lg font-medium text-gray-900 mb-1">
                Upload Image
              </p>
              <p className="text-sm text-gray-500 text-center">
                Drag and drop or click to browse
              </p>
              <p className="text-xs text-gray-400 mt-2">
                JPEG, PNG, WebP • Max 10MB
              </p>
            </div>

            {/* Camera Capture */}
            <button
              onClick={() => startCamera('environment')}
              className="drop-zone flex flex-col items-center justify-center p-8 min-h-64"
            >
              <div className="p-4 bg-green-100 rounded-full mb-4">
                <Camera className="w-8 h-8 text-green-600" />
              </div>
              <p className="text-lg font-medium text-gray-900 mb-1">
                Use Camera
              </p>
              <p className="text-sm text-gray-500 text-center">
                Take a photo with your device camera
              </p>
              <p className="text-xs text-gray-400 mt-2">
                Best for mobile devices
              </p>
            </button>
          </div>
        )}

        {/* Camera Error */}
        {cameraError && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center text-red-700">
            <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
            <span>{cameraError}</span>
          </div>
        )}

        {/* Tips Card */}
        <div className="card p-6 bg-blue-50 border-blue-100">
          <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
            <ImageIcon className="w-5 h-5 mr-2" />
            Tips for Better Results
          </h3>
          <ul className="text-sm text-blue-800 space-y-2">
            <li className="flex items-start">
              <span className="mr-2">•</span>
              Ensure good lighting - natural daylight works best
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              Focus on the affected area of the leaf
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              Avoid blurry or out-of-focus images
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              Include both healthy and diseased parts if visible
            </li>
          </ul>
        </div>

        {/* Action Buttons */}
        {imagePreview && !isCameraActive && (
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => {
                clearImage();
                fileInputRef.current?.click();
              }}
              className="btn btn-secondary flex items-center justify-center"
            >
              <RefreshCw className="w-5 h-5 mr-2" />
              Choose Different Image
            </button>
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="btn btn-primary flex items-center justify-center text-lg px-8"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze Image
                  <ArrowRight className="w-5 h-5 ml-2" />
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;
