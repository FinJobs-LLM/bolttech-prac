import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standalone prediction app. /api/* is proxied to the FastAPI backend so the
// browser can call the model without CORS friction. Point this at whichever
// serving app is running:
//   - serve.py        (dashboard backend)  -> port 8000
//   - serve_api.py    (production API)      -> port 8001
// Override the target with the API_TARGET env var at dev time.
const API_TARGET = process.env.API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174, // different port from the main dashboard (5173) so both can run
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
