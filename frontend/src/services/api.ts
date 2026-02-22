// =============================================================================
// AgriLite-Hybrid Frontend
// src/services/api.ts - API Service Layer
// 
// Axios instance configuration with interceptors for authentication,
// error handling, and API endpoints.
// =============================================================================

import axios, { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import toast from 'react-hot-toast';
import type {
  AuthResponse,
  LoginCredentials,
  RegisterData,
  CropsResponse,
  PredictionResponse,
  HistoryResponse,
  HistoryStats,
  DetectionHistory,
  User,
  ApiResponse,
} from '../types';

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

/**
 * Create configured Axios instance with interceptors
 */
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds for image processing
});

// =============================================================================
// Request Interceptor
// =============================================================================

api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Add auth token to requests
    const token = localStorage.getItem('access_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// =============================================================================
// Response Interceptor
// =============================================================================

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError<ApiResponse>) => {
    // Handle specific error codes
    if (error.response) {
      const { status, data } = error.response;

      switch (status) {
        case 401:
          // Unauthorized - clear tokens and redirect to login
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user');
          
          // Only redirect if not already on login page
          if (!window.location.pathname.includes('/login')) {
            window.location.href = '/login';
          }
          break;

        case 403:
          toast.error('You do not have permission to perform this action');
          break;

        case 429:
          toast.error('Too many requests. Please try again later.');
          break;

        case 500:
          toast.error('Server error. Please try again later.');
          break;

        default:
          // Show error message from API if available
          if (data?.error) {
            toast.error(data.error);
          }
      }
    } else if (error.request) {
      // Network error
      toast.error('Network error. Please check your connection.');
    }

    return Promise.reject(error);
  }
);

// =============================================================================
// Authentication API
// =============================================================================

export const authAPI = {
  /**
   * Login with email and password
   */
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/api/auth/login', credentials);
    return response.data;
  },

  /**
   * Register new user account
   */
  register: async (data: RegisterData): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/api/auth/register', data);
    return response.data;
  },

  /**
   * Refresh access token
   */
  refresh: async (): Promise<{ access_token: string }> => {
    const refreshToken = localStorage.getItem('refresh_token');
    const response = await api.post(
      '/api/auth/refresh',
      {},
      {
        headers: {
          Authorization: `Bearer ${refreshToken}`,
        },
      }
    );
    return response.data;
  },

  /**
   * Get current user profile
   */
  getProfile: async (): Promise<ApiResponse<User>> => {
    const response = await api.get('/api/auth/profile');
    return response.data;
  },

  /**
   * Update user profile
   */
  updateProfile: async (data: Partial<User>): Promise<ApiResponse<User>> => {
    const response = await api.put('/api/auth/profile', data);
    return response.data;
  },

  /**
   * Change password
   */
  changePassword: async (currentPassword: string, newPassword: string): Promise<ApiResponse> => {
    const response = await api.post('/api/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
    return response.data;
  },

  /**
   * Logout (client-side cleanup)
   */
  logout: (): void => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
  },
};

// =============================================================================
// Crops API
// =============================================================================

export const cropsAPI = {
  /**
   * Get all supported crops
   */
  getAll: async (): Promise<CropsResponse> => {
    const response = await api.get<CropsResponse>('/api/crops/');
    return response.data;
  },

  /**
   * Get single crop details
   */
  getById: async (cropName: string): Promise<ApiResponse> => {
    const response = await api.get(`/api/crops/${cropName}`);
    return response.data;
  },

  /**
   * Get crop names only
   */
  getNames: async (): Promise<ApiResponse<string[]>> => {
    const response = await api.get('/api/crops/names');
    return response.data;
  },
};

// =============================================================================
// Prediction API
// =============================================================================

export const predictAPI = {
  /**
   * Analyze image for disease detection
   */
  analyze: async (
    crop: string,
    image: File,
    options?: {
      temperature?: number;
      humidity?: number;
      saveHistory?: boolean;
    }
  ): Promise<PredictionResponse> => {
    const formData = new FormData();
    formData.append('crop', crop);
    formData.append('image', image);

    if (options?.temperature !== undefined) {
      formData.append('temperature', options.temperature.toString());
    }
    if (options?.humidity !== undefined) {
      formData.append('humidity', options.humidity.toString());
    }
    if (options?.saveHistory) {
      formData.append('save_history', 'true');
    }

    const response = await api.post<PredictionResponse>('/api/predict/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  /**
   * Analyze base64 encoded image
   */
  analyzeBase64: async (
    crop: string,
    imageBase64: string,
    options?: {
      temperature?: number;
      humidity?: number;
    }
  ): Promise<PredictionResponse> => {
    const response = await api.post<PredictionResponse>('/api/predict/base64', {
      crop,
      image_base64: imageBase64,
      ...options,
    });

    return response.data;
  },

  /**
   * Get treatment recommendations
   */
  getTreatment: async (disease: string, severity: string): Promise<ApiResponse> => {
    const response = await api.get('/api/predict/treatment', {
      params: { disease, severity },
    });
    return response.data;
  },
};

// =============================================================================
// History API
// =============================================================================

export const historyAPI = {
  /**
   * Get user's detection history
   */
  getAll: async (params?: {
    page?: number;
    per_page?: number;
    crop?: string;
    severity?: string;
  }): Promise<HistoryResponse> => {
    const response = await api.get<HistoryResponse>('/api/history/', { params });
    return response.data;
  },

  /**
   * Get single detection details
   */
  getById: async (id: number, includeImages = true): Promise<ApiResponse<DetectionHistory>> => {
    const response = await api.get(`/api/history/${id}`, {
      params: { include_images: includeImages },
    });
    return response.data;
  },

  /**
   * Save detection to history
   */
  save: async (detection: Partial<DetectionHistory>): Promise<ApiResponse<DetectionHistory>> => {
    const response = await api.post('/api/history/', detection);
    return response.data;
  },

  /**
   * Update detection notes
   */
  update: async (id: number, notes: string): Promise<ApiResponse<DetectionHistory>> => {
    const response = await api.put(`/api/history/${id}`, { notes });
    return response.data;
  },

  /**
   * Delete detection from history
   */
  delete: async (id: number): Promise<ApiResponse> => {
    const response = await api.delete(`/api/history/${id}`);
    return response.data;
  },

  /**
   * Get history statistics
   */
  getStats: async (): Promise<ApiResponse<HistoryStats>> => {
    const response = await api.get('/api/history/stats');
    return response.data;
  },
};

// =============================================================================
// Health Check API
// =============================================================================

export const healthAPI = {
  /**
   * Check API health status
   */
  check: async (): Promise<ApiResponse> => {
    const response = await api.get('/health');
    return response.data;
  },
};

// Export default api instance for custom requests
export default api;
