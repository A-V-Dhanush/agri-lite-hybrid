// =============================================================================
// AgriLite-Hybrid Frontend
// src/App.tsx - Root Application Component
// 
// Sets up routing, context providers, and global configuration.
// =============================================================================

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

// Context Providers
import { AuthProvider } from './contexts/AuthContext';
import { DetectionProvider } from './contexts/DetectionContext';

// Layout Components
import Header from './components/Header';
import Footer from './components/Footer';

// Page Components
import Home from './pages/Home';
import CropSelector from './pages/CropSelector';
import Upload from './pages/Upload';
import Result from './pages/Result';
import History from './pages/History';
import Login from './pages/Login';
import Register from './pages/Register';
import NotFound from './pages/NotFound';

// Route Protection
import ProtectedRoute from './components/ProtectedRoute';

/**
 * Root Application Component
 * 
 * Configures:
 * - React Router for client-side navigation
 * - Context providers for global state
 * - Toast notifications
 * - Layout structure
 */
function App() {
  return (
    <BrowserRouter>
      {/* Authentication Provider */}
      <AuthProvider>
        {/* Detection Context Provider */}
        <DetectionProvider>
          {/* Main Application Layout */}
          <div className="flex flex-col min-h-screen bg-gray-50">
            {/* Global Header */}
            <Header />
            
            {/* Main Content Area */}
            <main className="flex-1">
              <Routes>
                {/* Public Routes */}
                <Route path="/" element={<Home />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                
                {/* Detection Flow Routes */}
                <Route path="/select-crop" element={<CropSelector />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="/result" element={<Result />} />
                
                {/* Protected Routes (require authentication) */}
                <Route
                  path="/history"
                  element={
                    <ProtectedRoute>
                      <History />
                    </ProtectedRoute>
                  }
                />
                
                {/* 404 Not Found */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>
            
            {/* Global Footer */}
            <Footer />
          </div>
          
          {/* Toast Notifications Container */}
          <Toaster
            position="top-right"
            toastOptions={{
              // Default toast options
              duration: 4000,
              style: {
                background: '#fff',
                color: '#1f2937',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                borderRadius: '0.75rem',
                padding: '1rem',
              },
              // Success toast styling
              success: {
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#fff',
                },
              },
              // Error toast styling
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </DetectionProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
