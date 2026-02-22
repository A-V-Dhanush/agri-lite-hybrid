// =============================================================================
// AgriLite-Hybrid Frontend
// src/pages/Result.tsx - Detection Results Page
// 
// Displays disease detection results including confidence, severity,
// Grad-CAM heatmap, and treatment recommendations.
// =============================================================================

import React, { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  RefreshCw,
  Download,
  Save,
  Share2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Leaf,
  Thermometer,
  Droplets,
  Eye,
  EyeOff,
  Loader2,
} from 'lucide-react';
import { useDetection } from '../contexts/DetectionContext';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

// ---------------------------------------------------------------------------
// Severity Badge Component
// ---------------------------------------------------------------------------
interface SeverityBadgeProps {
  severity: string;
  size?: 'sm' | 'md' | 'lg';
}

const SeverityBadge: React.FC<SeverityBadgeProps> = ({ severity, size = 'md' }) => {
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-2 text-base',
  };

  const getIcon = () => {
    switch (severity.toLowerCase()) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4" />;
      case 'severe':
        return <XCircle className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  return (
    <span
      className={`inline-flex items-center space-x-1 font-medium rounded-full ${sizeClasses[size]} severity-${severity.toLowerCase()}`}
    >
      {getIcon()}
      <span className="capitalize">{severity}</span>
    </span>
  );
};

// ---------------------------------------------------------------------------
// Treatment Card Component
// ---------------------------------------------------------------------------
interface TreatmentCardProps {
  treatment: {
    type: string;
    name: string;
    dosage: string;
    application: string;
    schedule: string;
  };
}

const TreatmentCard: React.FC<TreatmentCardProps> = ({ treatment }) => {
  const treatmentType = treatment?.type || 'chemical';
  const isOrganic = treatmentType.toLowerCase() === 'organic';

  return (
    <div
      className={`card p-4 border-l-4 ${
        isOrganic ? 'border-l-green-500 bg-green-50' : 'border-l-blue-500 bg-blue-50'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded-full ${
            isOrganic
              ? 'bg-green-200 text-green-800'
              : 'bg-blue-200 text-blue-800'
          }`}
        >
          {treatmentType}
        </span>
      </div>
      <h4 className="font-semibold text-gray-900 mb-2">{treatment?.name || 'Treatment'}</h4>
      <dl className="text-sm space-y-1">
        {treatment?.dosage && (
        <div>
          <dt className="text-gray-500 inline">Dosage: </dt>
          <dd className="text-gray-700 inline">{treatment.dosage}</dd>
        </div>
        )}
        {treatment?.application && (
        <div>
          <dt className="text-gray-500 inline">Application: </dt>
          <dd className="text-gray-700 inline">{treatment.application}</dd>
        </div>
        )}
        {treatment?.schedule && (
        <div>
          <dt className="text-gray-500 inline">Schedule: </dt>
          <dd className="text-gray-700 inline">{treatment.schedule}</dd>
        </div>
        )}
      </dl>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
const Result: React.FC = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const {
    selectedCrop,
    predictionResult,
    imagePreview,
    resetWorkflow,
    saveToHistory,
  } = useDetection();

  const [showHeatmap, setShowHeatmap] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // ---------------------------------------------------------------------------
  // Redirect if no result
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!predictionResult || !selectedCrop) {
      navigate('/select-crop');
    }
  }, [predictionResult, selectedCrop, navigate]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleStartOver = () => {
    resetWorkflow();
    navigate('/select-crop');
  };

  const handleSaveToHistory = async () => {
    if (!isAuthenticated) {
      toast.error('Please login to save detection history');
      navigate('/login', { state: { from: '/result' } });
      return;
    }

    setIsSaving(true);
    const success = await saveToHistory();
    setIsSaving(false);

    if (success) {
      setSaved(true);
    }
  };

  const handleDownload = () => {
    // Download the result image
    const imageData = showHeatmap && predictionResult?.heatmap
      ? predictionResult.heatmap
      : predictionResult?.original_image || imagePreview;

    if (imageData) {
      const link = document.createElement('a');
      link.href = imageData.startsWith('data:')
        ? imageData
        : `data:image/jpeg;base64,${imageData}`;
      link.download = `${selectedCrop?.name}_${predictionResult?.disease}_${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      toast.success('Image downloaded');
    }
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `Plant Disease Detection - ${selectedCrop?.name}`,
          text: `Detected: ${predictionResult?.disease} (${predictionResult?.severity}) with ${Math.round((predictionResult?.confidence || 0) * 100)}% confidence`,
        });
      } catch (err) {
        // User cancelled or error
        console.log('Share cancelled or failed:', err);
      }
    } else {
      // Fallback: copy to clipboard
      const text = `Plant Disease Detection Result
Crop: ${selectedCrop?.name}
Disease: ${predictionResult?.disease}
Severity: ${predictionResult?.severity}
Confidence: ${Math.round((predictionResult?.confidence || 0) * 100)}%`;

      await navigator.clipboard.writeText(text);
      toast.success('Result copied to clipboard');
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  if (!predictionResult || !selectedCrop) {
    return null;
  }

  const confidencePercent = Math.round(predictionResult.confidence * 100);
  const isHealthy = predictionResult.severity.toLowerCase() === 'healthy';

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 sm:py-12">
      {/* Back Link */}
      <button
        onClick={() => navigate('/upload')}
        className="inline-flex items-center text-gray-600 hover:text-primary-600 mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back to upload
      </button>

      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-3">
          Detection Results
        </h1>
        <p className="text-lg text-gray-600">
          Analysis for {selectedCrop.display_name || selectedCrop.name}
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left Column - Image */}
        <div className="space-y-4">
          {/* Image Display */}
          <div className="card overflow-hidden">
            <div className="relative">
              <img
                src={
                  showHeatmap && predictionResult.heatmap
                    ? predictionResult.heatmap.startsWith('data:')
                      ? predictionResult.heatmap
                      : `data:image/jpeg;base64,${predictionResult.heatmap}`
                    : predictionResult.original_image
                      ? predictionResult.original_image.startsWith('data:')
                        ? predictionResult.original_image
                        : `data:image/jpeg;base64,${predictionResult.original_image}`
                      : imagePreview || ''
                }
                alt="Detection result"
                className="w-full h-auto max-h-96 object-contain bg-gray-100"
              />
              {/* Toggle Heatmap */}
              {predictionResult.heatmap && (
                <button
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className="absolute bottom-4 right-4 btn bg-white/90 hover:bg-white shadow-lg"
                >
                  {showHeatmap ? (
                    <>
                      <EyeOff className="w-4 h-4 mr-2" />
                      Hide Heatmap
                    </>
                  ) : (
                    <>
                      <Eye className="w-4 h-4 mr-2" />
                      Show Heatmap
                    </>
                  )}
                </button>
              )}
            </div>
            <div className="p-4 bg-gray-50 border-t flex justify-center space-x-3">
              <button
                onClick={handleDownload}
                className="btn btn-secondary flex items-center"
              >
                <Download className="w-4 h-4 mr-2" />
                Download
              </button>
              <button
                onClick={handleShare}
                className="btn btn-secondary flex items-center"
              >
                <Share2 className="w-4 h-4 mr-2" />
                Share
              </button>
              {!saved && (
                <button
                  onClick={handleSaveToHistory}
                  disabled={isSaving}
                  className="btn btn-primary flex items-center"
                >
                  {isSaving ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="w-4 h-4 mr-2" />
                  )}
                  Save
                </button>
              )}
              {saved && (
                <span className="btn bg-green-100 text-green-700 cursor-default flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Saved
                </span>
              )}
            </div>
          </div>

          {/* Environmental Risk (if available) */}
          {predictionResult.environmental_risk && (
            <div className="card p-4">
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-yellow-500" />
                Environmental Risk Assessment
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                {predictionResult.environmental_risk.temperature !== undefined && (
                  <div className="flex items-center space-x-2">
                    <Thermometer className="w-4 h-4 text-red-500" />
                    <span className="text-gray-600">Temperature:</span>
                    <span className="font-medium">
                      {predictionResult.environmental_risk.temperature}°C
                    </span>
                  </div>
                )}
                {predictionResult.environmental_risk.humidity !== undefined && (
                  <div className="flex items-center space-x-2">
                    <Droplets className="w-4 h-4 text-blue-500" />
                    <span className="text-gray-600">Humidity:</span>
                    <span className="font-medium">
                      {predictionResult.environmental_risk.humidity}%
                    </span>
                  </div>
                )}
              </div>
              {predictionResult.environmental_risk.risk_level && (
                <div className="mt-3 p-2 bg-yellow-50 rounded text-sm text-yellow-800">
                  Risk Level: <span className="font-medium capitalize">
                    {predictionResult.environmental_risk.risk_level}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column - Results */}
        <div className="space-y-6">
          {/* Main Result Card */}
          <div
            className={`card p-6 border-2 ${
              isHealthy ? 'border-green-400 bg-green-50' : 'border-yellow-400 bg-yellow-50'
            }`}
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <span className="text-sm text-gray-500 uppercase tracking-wide">
                  Detected Disease
                </span>
                <h2 className="text-2xl font-bold text-gray-900 mt-1">
                  {predictionResult.disease}
                </h2>
              </div>
              <SeverityBadge severity={predictionResult.severity} size="lg" />
            </div>

            {/* Confidence Meter */}
            <div className="mb-4">
              <div className="flex justify-between items-center text-sm mb-1">
                <span className="text-gray-600">Confidence</span>
                <span className="font-semibold text-gray-900">{confidencePercent}%</span>
              </div>
              <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    confidencePercent >= 80
                      ? 'bg-green-500'
                      : confidencePercent >= 60
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                  }`}
                  style={{ width: `${confidencePercent}%` }}
                />
              </div>
            </div>

            {/* Crop Info */}
            <div className="flex items-center text-sm text-gray-600 pt-4 border-t">
              <Leaf className="w-4 h-4 mr-2 text-primary-500" />
              <span>
                Crop: <strong>{selectedCrop.name}</strong>
              </span>
            </div>
          </div>

          {/* Treatment Recommendations */}
          {!isHealthy && predictionResult.treatment && predictionResult.treatment.length > 0 && (
            <div className="card p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Treatment Recommendations
              </h3>
              <div className="space-y-4">
                {predictionResult.treatment.map((treatment, index) => (
                  <TreatmentCard key={index} treatment={treatment} />
                ))}
              </div>
              <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm text-gray-600">
                <strong>Note:</strong> Always follow product labels and local agricultural
                guidelines. Consider consulting with local agricultural extension services
                for region-specific recommendations.
              </div>
            </div>
          )}

          {/* Healthy Plant Message */}
          {isHealthy && (
            <div className="card p-6 bg-green-50 border border-green-200">
              <div className="flex items-center space-x-3 mb-4">
                <CheckCircle className="w-8 h-8 text-green-500" />
                <h3 className="text-xl font-semibold text-green-800">
                  Your Plant is Healthy!
                </h3>
              </div>
              <p className="text-green-700 mb-4">
                No disease detected. Continue with your current care regimen to maintain
                plant health.
              </p>
              <h4 className="font-medium text-green-800 mb-2">
                Prevention Tips:
              </h4>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• Ensure proper spacing between plants for air circulation</li>
                <li>• Water at the base to keep leaves dry</li>
                <li>• Remove any dead or yellowing leaves promptly</li>
                <li>• Monitor regularly for early signs of disease</li>
                <li>• Apply preventive fungicide during monsoon season</li>
              </ul>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4">
            <button
              onClick={handleStartOver}
              className="btn btn-primary flex items-center justify-center flex-1"
            >
              <RefreshCw className="w-5 h-5 mr-2" />
              Analyze Another Image
            </button>
            <Link
              to="/history"
              className="btn btn-secondary flex items-center justify-center flex-1"
            >
              View History
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Result;
