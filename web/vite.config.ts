import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" makes the built bundle path-agnostic — it works served at the
// domain root (sisyphus-pbpk.io) OR at a subpath (sisyphus-pbpk.io/app/).
// Data is fetched via import.meta.env.BASE_URL so it resolves the same way.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2020",
  },
  server: {
    port: 5173,
    // Phase-2 hook: when a live FastAPI engine exists, proxy /api to it in dev.
    // proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
