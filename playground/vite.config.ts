import { defineConfig } from "vitest/config";

// Project Pages serves under /<repo>/. The deploy workflow sets GITHUB_ACTIONS=true,
// so production builds get the "/nanoACE/" base while local dev/preview use "/".
const base = process.env.GITHUB_ACTIONS ? "/nanoACE/" : "/";

export default defineConfig({
  base,
  build: { target: "es2022" },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
