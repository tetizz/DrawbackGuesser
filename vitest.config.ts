import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // UCI integration tests spawn real child processes. Bounding file-level
    // concurrency prevents unrelated CPU-heavy simulation suites from
    // starving their deterministic handshake and shutdown deadlines.
    maxWorkers: 2,
    include: [
      "packages/**/*.test.ts",
      "apps/**/*.test.ts",
      "scripts/**/*.test.mjs",
    ],
    coverage: {
      provider: "v8",
    },
  },
});
