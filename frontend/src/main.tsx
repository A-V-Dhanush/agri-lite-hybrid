// =============================================================================
// AgriLite-Hybrid Frontend
// src/main.tsx - Application Entry Point
// 
// This is the main entry point for the React application.
// It sets up the root element and renders the App component.
// =============================================================================

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Get root element
const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Failed to find the root element');
}

// Create root and render application
createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
