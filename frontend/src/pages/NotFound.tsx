// =============================================================================
// AgriLite-Hybrid Frontend
// src/pages/NotFound.tsx - 404 Not Found Page
// 
// Displayed when user navigates to a non-existent route.
// =============================================================================

import React from 'react';
import { Link } from 'react-router-dom';
import { Home, Leaf, ArrowLeft } from 'lucide-react';

const NotFound: React.FC = () => {
  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center px-4">
      <div className="text-center max-w-md">
        {/* Icon */}
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-primary-100 text-primary-600 mb-6">
          <Leaf className="w-10 h-10" />
        </div>

        {/* 404 Number */}
        <h1 className="text-6xl font-bold text-gray-900 mb-4">404</h1>

        {/* Message */}
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">
          Page Not Found
        </h2>
        <p className="text-gray-600 mb-8">
          Oops! The page you're looking for doesn't exist or has been moved.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/"
            className="btn btn-primary inline-flex items-center justify-center"
          >
            <Home className="w-5 h-5 mr-2" />
            Go to Home
          </Link>
          <button
            onClick={() => window.history.back()}
            className="btn btn-secondary inline-flex items-center justify-center"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Go Back
          </button>
        </div>

        {/* Quick Links */}
        <div className="mt-10 pt-8 border-t border-gray-200">
          <p className="text-sm text-gray-500 mb-4">Or try these pages:</p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              to="/select-crop"
              className="text-primary-600 hover:text-primary-700 text-sm"
            >
              Detect Disease
            </Link>
            <Link
              to="/history"
              className="text-primary-600 hover:text-primary-700 text-sm"
            >
              Detection History
            </Link>
            <Link
              to="/login"
              className="text-primary-600 hover:text-primary-700 text-sm"
            >
              Login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
