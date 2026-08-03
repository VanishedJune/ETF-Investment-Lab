import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import ts from "typescript";

async function importChartData() {
  const source = fs.readFileSync(path.resolve("src/chart-data.ts"), "utf8");
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const outputPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "investment-lab-chart-")), "chart-data.mjs");
  fs.writeFileSync(outputPath, output, "utf8");
  return import(pathToFileURL(outputPath).href);
}

test("missing technical-indicator values remain chart gaps instead of zero", async () => {
  const { alignIndicatorValues } = await importChartData();

  const values = alignIndicatorValues(
    [{ date: "2026-01-01" }, { date: "2026-01-02" }],
    [{ date: "2026-01-01", ma_5: null }, { date: "2026-01-02", ma_5: "10.25" }],
    "ma_5",
  );

  assert.deepEqual(values, [null, 10.25]);
});
