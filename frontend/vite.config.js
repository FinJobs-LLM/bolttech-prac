import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During dev, /api/* is proxied to the FastAPI server so the prediction demo
// works without CORS friction. Dashboard data is loaded from /dashboard_data.json
// in the public/ folder (written by src/run_pipeline.py).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
