const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  timeout: 30000,
  retries: 0,
  workers: 2,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:8765',
    headless: true,
    launchOptions: process.env.BROWSER_EXECUTABLE
      ? { executablePath: process.env.BROWSER_EXECUTABLE }
      : {},
  },
  webServer: {
    command: 'python3 -m http.server 8765 --bind 127.0.0.1 --directory site/.browser-build',
    url: 'http://127.0.0.1:8765/docs/',
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
    stderr: 'ignore',
  },
});
