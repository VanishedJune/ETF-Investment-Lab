import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: { baseURL: "http://127.0.0.1:8765", headless: true },
  webServer: {
    command: '".venv\\Scripts\\python.exe" -m uvicorn backend.web:app --host 127.0.0.1 --port 8765',
    cwd: "..",
    url: "http://127.0.0.1:8765/api/health",
    timeout: 90_000,
    reuseExistingServer: true,
  },
});
