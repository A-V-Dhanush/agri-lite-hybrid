// =============================================================================
// AgriLite-Hybrid Frontend
// src/contexts/DetectionContext.tsx - Detection Workflow Context Provider
// 
// Manages the multi-step detection workflow: crop selection -> image capture
// -> analysis -> results display.
// =============================================================================

import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import type { Crop, PredictionResult } from '../types';
import { predictAPI, historyAPI } from '../services/api';
import toast from 'react-hot-toast';

// =============================================================================
// Context Types
// =============================================================================

interface DetectionState {
  selectedCrop: Crop | null;
  imageFile: File | null;
  imagePreview: string | null;
  predictionResult: PredictionResult | null;
  isAnalyzing: boolean;
  error: string | null;
}

interface DetectionContextType extends DetectionState {
  // Crop selection
  selectCrop: (crop: Crop) => void;
  clearCrop: () => void;

  // Image management
  setImage: (file: File) => void;
  clearImage: () => void;

  // Analysis
  analyzeImage: (options?: AnalysisOptions) => Promise<boolean>;
  clearResult: () => void;

  // Reset workflow
  resetWorkflow: () => void;

  // Save to history
  saveToHistory: (notes?: string) => Promise<boolean>;
}

interface AnalysisOptions {
  temperature?: number;
  humidity?: number;
  saveHistory?: boolean;
}

// =============================================================================
// Initial State
// =============================================================================

const initialState: DetectionState = {
  selectedCrop: null,
  imageFile: null,
  imagePreview: null,
  predictionResult: null,
  isAnalyzing: false,
  error: null,
};

// =============================================================================
// Context Creation
// =============================================================================

const DetectionContext = createContext<DetectionContextType | undefined>(undefined);

// =============================================================================
// Detection Provider Component
// =============================================================================

interface DetectionProviderProps {
  children: React.ReactNode;
}

export const DetectionProvider: React.FC<DetectionProviderProps> = ({ children }) => {
  const [state, setState] = useState<DetectionState>(initialState);

  // ---------------------------------------------------------------------------
  // Crop Selection
  // ---------------------------------------------------------------------------
  const selectCrop = useCallback((crop: Crop) => {
    setState((prev) => ({
      ...prev,
      selectedCrop: crop,
      // Clear previous results when changing crop
      imageFile: null,
      imagePreview: null,
      predictionResult: null,
      error: null,
    }));
  }, []);

  const clearCrop = useCallback(() => {
    setState((prev) => ({
      ...prev,
      selectedCrop: null,
    }));
  }, []);

  // ---------------------------------------------------------------------------
  // Image Management
  // ---------------------------------------------------------------------------
  const setImage = useCallback((file: File) => {
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);

    // Clean up previous preview URL
    if (state.imagePreview) {
      URL.revokeObjectURL(state.imagePreview);
    }

    setState((prev) => ({
      ...prev,
      imageFile: file,
      imagePreview: previewUrl,
      predictionResult: null,
      error: null,
    }));
  }, [state.imagePreview]);

  const clearImage = useCallback(() => {
    // Clean up preview URL
    if (state.imagePreview) {
      URL.revokeObjectURL(state.imagePreview);
    }

    setState((prev) => ({
      ...prev,
      imageFile: null,
      imagePreview: null,
      predictionResult: null,
      error: null,
    }));
  }, [state.imagePreview]);

  // ---------------------------------------------------------------------------
  // Image Analysis
  // ---------------------------------------------------------------------------
  const analyzeImage = useCallback(
    async (options?: AnalysisOptions): Promise<boolean> => {
      if (!state.selectedCrop || !state.imageFile) {
        toast.error('Please select a crop and upload an image');
        return false;
      }

      setState((prev) => ({
        ...prev,
        isAnalyzing: true,
        error: null,
      }));

      try {
        const response = await predictAPI.analyze(
          state.selectedCrop.name.toLowerCase(),
          state.imageFile,
          {
            temperature: options?.temperature,
            humidity: options?.humidity,
            saveHistory: options?.saveHistory,
          }
        );

        if (response.success && response.data) {
          setState((prev) => ({
            ...prev,
            predictionResult: response.data,
            isAnalyzing: false,
          }));

          // Show severity-based notification
          const severity = response.data.severity;
          if (severity === 'healthy') {
            toast.success('Great news! Your plant appears healthy.');
          } else if (severity === 'severe') {
            toast.error(`Disease detected: ${response.data.disease} (Severe)`);
          } else {
            toast(`Disease detected: ${response.data.disease}`, {
              icon: '⚠️',
            });
          }

          return true;
        } else {
          throw new Error(response.error || 'Analysis failed');
        }
      } catch (error: unknown) {
        const err = error as { response?: { data?: { error?: string } }; message?: string };
        const message = err.response?.data?.error || err.message || 'Analysis failed';

        setState((prev) => ({
          ...prev,
          isAnalyzing: false,
          error: message,
        }));

        toast.error(message);
        return false;
      }
    },
    [state.selectedCrop, state.imageFile]
  );

  const clearResult = useCallback(() => {
    setState((prev) => ({
      ...prev,
      predictionResult: null,
      error: null,
    }));
  }, []);

  // ---------------------------------------------------------------------------
  // Save to History
  // ---------------------------------------------------------------------------
  const saveToHistory = useCallback(
    async (notes?: string): Promise<boolean> => {
      if (!state.predictionResult || !state.selectedCrop) {
        toast.error('No detection result to save');
        return false;
      }

      try {
        await historyAPI.save({
          crop: state.selectedCrop.name,
          disease: state.predictionResult.disease,
          severity: state.predictionResult.severity,
          confidence: state.predictionResult.confidence,
          notes: notes || '',
          original_image: state.predictionResult.original_image,
          heatmap_image: state.predictionResult.heatmap,
        });

        toast.success('Saved to history');
        return true;
      } catch (error: unknown) {
        const err = error as { response?: { data?: { error?: string } } };
        const message = err.response?.data?.error || 'Failed to save to history';
        toast.error(message);
        return false;
      }
    },
    [state.predictionResult, state.selectedCrop]
  );

  // ---------------------------------------------------------------------------
  // Reset Workflow
  // ---------------------------------------------------------------------------
  const resetWorkflow = useCallback(() => {
    // Clean up preview URL
    if (state.imagePreview) {
      URL.revokeObjectURL(state.imagePreview);
    }

    setState(initialState);
  }, [state.imagePreview]);

  // ---------------------------------------------------------------------------
  // Memoized Context Value
  // ---------------------------------------------------------------------------
  const value = useMemo(
    () => ({
      ...state,
      selectCrop,
      clearCrop,
      setImage,
      clearImage,
      analyzeImage,
      clearResult,
      resetWorkflow,
      saveToHistory,
    }),
    [
      state,
      selectCrop,
      clearCrop,
      setImage,
      clearImage,
      analyzeImage,
      clearResult,
      resetWorkflow,
      saveToHistory,
    ]
  );

  return (
    <DetectionContext.Provider value={value}>{children}</DetectionContext.Provider>
  );
};

// =============================================================================
// Custom Hook
// =============================================================================

export const useDetection = (): DetectionContextType => {
  const context = useContext(DetectionContext);
  if (context === undefined) {
    throw new Error('useDetection must be used within a DetectionProvider');
  }
  return context;
};

export default DetectionContext;
