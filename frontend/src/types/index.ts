// =============================================================================
// AgriLite-Hybrid Frontend
// src/types/index.ts - TypeScript Type Definitions
// 
// Central location for all application type definitions.
// =============================================================================

// =============================================================================
// User & Authentication Types
// =============================================================================

/**
 * User profile information
 */
export interface User {
  id: number;
  email: string;
  first_name: string | null;
  last_name: string | null;
  full_name: string;
  role: 'user' | 'admin';
  is_active: boolean;
  created_at: string;
  last_login: string | null;
}

/**
 * Login request payload
 */
export interface LoginCredentials {
  email: string;
  password: string;
}

/**
 * Registration request payload
 */
export interface RegisterData {
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
  phone?: string;
}

/**
 * Authentication response from login/register
 */
export interface AuthResponse {
  success: boolean;
  message: string;
  access_token: string;
  refresh_token: string;
  user: User;
}

// =============================================================================
// Crop Types
// =============================================================================

/**
 * Crop information
 */
export interface Crop {
  name: string;
  display_name: string;
  description: string;
  image: string;
  diseases: string[];
  color: string;
}

/**
 * Crops API response
 */
export interface CropsResponse {
  success: boolean;
  data: {
    crops: Crop[];
    total: number;
  };
}

// =============================================================================
// Prediction & Detection Types
// =============================================================================

/**
 * Severity level type
 */
export type Severity = 'mild' | 'medium' | 'severe';

/**
 * Prediction request payload
 */
export interface PredictionRequest {
  crop: string;
  image: File;
  temperature?: number;
  humidity?: number;
  save_history?: boolean;
}

/**
 * Prediction result from API
 */
export interface PredictionResult {
  crop: string;
  disease: string;
  severity: Severity;
  confidence: number;
  treatment: string[];
  heatmap_base64: string;
  original_image_base64: string;
  environmental_risk: 'normal' | 'elevated' | 'high' | 'unknown';
  model_used: 'keras' | 'tflite' | 'placeholder';
  history_id?: number;
}

/**
 * Prediction API response
 */
export interface PredictionResponse {
  success: boolean;
  data: PredictionResult;
  error?: string;
}

// =============================================================================
// Detection History Types
// =============================================================================

/**
 * Detection history record
 */
export interface DetectionHistory {
  id: number;
  user_id: number;
  crop: string;
  disease: string;
  severity: Severity;
  confidence: number;
  treatment: string[];
  temperature: number | null;
  humidity: number | null;
  environmental_risk: string | null;
  device_type: string;
  location: string | null;
  notes: string | null;
  created_at: string;
  original_image?: string;
  heatmap_image?: string;
}

/**
 * Paginated history response
 */
export interface HistoryResponse {
  success: boolean;
  data: {
    items: DetectionHistory[];
    page: number;
    per_page: number;
    total: number;
    pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

/**
 * History statistics
 */
export interface HistoryStats {
  total_detections: number;
  by_crop: Record<string, number>;
  by_severity: Record<string, number>;
  top_diseases: Array<{ disease: string; count: number }>;
  healthy_count: number;
}

// =============================================================================
// UI State Types
// =============================================================================

/**
 * Loading state for API operations
 */
export interface LoadingState {
  isLoading: boolean;
  error: string | null;
}

/**
 * Toast notification type
 */
export type ToastType = 'success' | 'error' | 'loading' | 'info';

/**
 * Detection flow state
 */
export interface DetectionState {
  selectedCrop: Crop | null;
  imageFile: File | null;
  imagePreview: string | null;
  predictionResult: PredictionResult | null;
  isProcessing: boolean;
  error: string | null;
}

// =============================================================================
// API Response Types
// =============================================================================

/**
 * Generic API success response
 */
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  details?: unknown;
}

/**
 * API error response
 */
export interface ApiError {
  success: false;
  error: string;
  message?: string;
  details?: unknown;
  status?: number;
}

// =============================================================================
// Component Prop Types
// =============================================================================

/**
 * Common button props
 */
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
}

/**
 * Form input props
 */
export interface InputProps {
  label?: string;
  error?: string;
  helperText?: string;
  required?: boolean;
  fullWidth?: boolean;
}

/**
 * Modal component props
 */
export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

// =============================================================================
// Environmental Data Types
// =============================================================================

/**
 * Environmental sensor data (from DHT22 on Raspberry Pi)
 */
export interface EnvironmentalData {
  temperature: number;  // Celsius
  humidity: number;     // Percentage
  timestamp: string;
}

/**
 * Environmental risk assessment
 */
export interface EnvironmentalRisk {
  level: 'normal' | 'elevated' | 'high';
  factors: string[];
  recommendations: string[];
}
