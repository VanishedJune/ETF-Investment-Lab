import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', testMatch: 'catalog-ui.spec.ts', workers: 1,
  use: { baseURL: 'http://127.0.0.1:4178', headless: true, viewport: { width: 1500, height: 1050 } },
  webServer: { command: 'npm run preview -- --port 4178', url: 'http://127.0.0.1:4178', reuseExistingServer: true },
});
