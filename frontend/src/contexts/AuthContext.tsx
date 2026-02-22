// =============================================================================
// AgriLite-Hybrid Frontend
// src/contexts/AuthContext.tsx - Authentication Context Provider
// 
// Manages user authentication state, login/logout, and token persistence.
// =============================================================================

import React, { createContext, useContext, useEffect, useState, useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import type { User, LoginCredentials, RegisterData, AuthResponse } from '../types';
import { authAPI } from '../services/api';

// =============================================================================
// Context Types
// =============================================================================

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginCredentials) => Promise<boolean>;
  register: (data: RegisterData) => Promise<boolean>;
  logout: () => void;
  updateUser: (user: User) => void;
}

// =============================================================================
// Context Creation
// =============================================================================

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// =============================================================================
// Auth Provider Component
// =============================================================================

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // ---------------------------------------------------------------------------
  // Initialize auth state from localStorage
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const initAuth = async () => {
      try {
        const storedUser = localStorage.getItem('user');
        const accessToken = localStorage.getItem('access_token');

        if (storedUser && accessToken) {
          // Validate token by fetching profile
          try {
            const response = await authAPI.getProfile();
            if (response.data) {
              setUser(response.data);
              localStorage.setItem('user', JSON.stringify(response.data));
            }
          } catch {
            // Token invalid, clear storage
            authAPI.logout();
            setUser(null);
          }
        }
      } catch (error) {
        console.error('Auth initialization error:', error);
        authAPI.logout();
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, []);

  // ---------------------------------------------------------------------------
  // Login handler
  // ---------------------------------------------------------------------------
  const login = useCallback(async (credentials: LoginCredentials): Promise<boolean> => {
    setIsLoading(true);
    try {
      const response: AuthResponse = await authAPI.login(credentials);

      // Store tokens
      localStorage.setItem('access_token', response.access_token);
      if (response.refresh_token) {
        localStorage.setItem('refresh_token', response.refresh_token);
      }

      // Store user data
      localStorage.setItem('user', JSON.stringify(response.user));
      setUser(response.user);

      toast.success(`Welcome back, ${response.user.name || response.user.email}!`);
      return true;
    } catch (error: unknown) {
      const err = error as { response?: { data?: { error?: string } } };
      const message = err.response?.data?.error || 'Login failed. Please try again.';
      toast.error(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Register handler
  // ---------------------------------------------------------------------------
  const register = useCallback(async (data: RegisterData): Promise<boolean> => {
    setIsLoading(true);
    try {
      const response: AuthResponse = await authAPI.register(data);

      // Store tokens
      localStorage.setItem('access_token', response.access_token);
      if (response.refresh_token) {
        localStorage.setItem('refresh_token', response.refresh_token);
      }

      // Store user data
      localStorage.setItem('user', JSON.stringify(response.user));
      setUser(response.user);

      toast.success('Account created successfully!');
      return true;
    } catch (error: unknown) {
      const err = error as { response?: { data?: { error?: string } } };
      const message = err.response?.data?.error || 'Registration failed. Please try again.';
      toast.error(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Logout handler
  // ---------------------------------------------------------------------------
  const logout = useCallback(() => {
    authAPI.logout();
    setUser(null);
    toast.success('Logged out successfully');
  }, []);

  // ---------------------------------------------------------------------------
  // Update user data
  // ---------------------------------------------------------------------------
  const updateUser = useCallback((updatedUser: User) => {
    setUser(updatedUser);
    localStorage.setItem('user', JSON.stringify(updatedUser));
  }, []);

  // ---------------------------------------------------------------------------
  // Memoized context value
  // ---------------------------------------------------------------------------
  const value = useMemo(
    () => ({
      user,
      isAuthenticated: !!user,
      isLoading,
      login,
      register,
      logout,
      updateUser,
    }),
    [user, isLoading, login, register, logout, updateUser]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// =============================================================================
// Custom Hook
// =============================================================================

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
