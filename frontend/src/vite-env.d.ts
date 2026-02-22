/// <reference types="vite/client" />

// =============================================================================
// AgriLite-Hybrid Frontend
// src/vite-env.d.ts - Vite Environment Type Definitions
// =============================================================================

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_APP_NAME: string;
  readonly VITE_ENABLE_MOCK: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
