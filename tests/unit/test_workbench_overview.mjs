import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { statusKind, statusLabel, fmtDuration, fmtNumber, fmtPercent } from "../../src/evaluation/static/workbench/js/core/format.js";

// 1. Status Kind Mapping Unit Tests
assert.equal(statusKind("PASSED"), "success", "PASSED must be success (green)");
assert.equal(statusKind("COMPLETED"), "success", "COMPLETED must be success (green)");
assert.equal(statusKind("APPROVED"), "success", "APPROVED must be success (green)");
assert.equal(statusKind("SCREENED"), "success", "SCREENED must be success (green)");

assert.equal(statusKind("FAILED"), "danger", "FAILED must be danger (red)");
assert.equal(statusKind("REJECTED"), "danger", "REJECTED must be danger (red)");
assert.equal(statusKind("TAMPERED"), "danger", "TAMPERED must be danger (red)");

assert.equal(statusKind("RUNNING"), "running", "RUNNING must be running (animated)");
assert.equal(statusKind("STARTED"), "running", "STARTED must be running (animated)");
assert.equal(statusKind("IN_PROGRESS"), "running", "IN_PROGRESS must be running (animated)");

assert.equal(statusKind("PENDING"), "neutral", "PENDING must be neutral/amber, never green");
assert.equal(statusKind("SKIPPED"), "warning", "SKIPPED must be warning, never green");
assert.equal(statusKind("UNAVAILABLE"), "warning", "UNAVAILABLE must be warning, never green");
assert.equal(statusKind("NOT_EXECUTED"), "warning", "NOT_EXECUTED must be warning, never green");

// 2. fmtDuration Unit Tests
assert.equal(fmtDuration(45), "45s");
assert.equal(fmtDuration(125), "2m 5s");
assert.equal(fmtDuration(3600), "1h 0m");
assert.equal(fmtDuration(null), "Not recorded");
assert.equal(fmtDuration("2026-10-20T10:00:00Z", "2026-10-20T10:04:12Z"), "4m 12s");

// 3. Inspect overview.js source for required operational elements
const overviewPath = path.resolve("src/evaluation/static/workbench/js/pages/overview.js");
const overviewSrc = fs.readFileSync(overviewPath, "utf-8");

// Canonical 8 stages check
const expectedStages = [
  "Dataset Validation",
  "PII Masking",
  "LoRA / Optional DP-LoRA",
  "Adapter Screening",
  "Integrity Signing",
  "Device-Bound Encryption",
  "Deployment Verification",
  "Inference Validation"
];
for (const st of expectedStages) {
  assert.ok(overviewSrc.includes(st), `overview.js must contain canonical stage: ${st}`);
}

// Check summary cards labels
assert.ok(overviewSrc.includes("Total Pipeline Runs"), "Must have Total Pipeline Runs card");
assert.ok(overviewSrc.includes("Successful Runs"), "Must have Successful Runs card");
assert.ok(overviewSrc.includes("Failed Runs"), "Must have Failed Runs card");
assert.ok(overviewSrc.includes("Currently Running"), "Must have Currently Running card");

// Check prominent action
assert.ok(overviewSrc.includes("New Pipeline Run"), "Must have prominent New Pipeline Run action");

// Check evidence disclosure
assert.ok(overviewSrc.includes("Evidence Unavailable"), "Must show Evidence Unavailable when unrecorded");
assert.ok(overviewSrc.includes("Not Evaluated"), "Must show Not Evaluated when unrecorded");

// Check CSS file for animations and reduced motion
const cssPath = path.resolve("src/evaluation/static/workbench/workbench.css");
const cssSrc = fs.readFileSync(cssPath, "utf-8");

assert.ok(cssSrc.includes(".pipeline-grid"), "CSS must contain .pipeline-grid");
assert.ok(cssSrc.includes(".pipeline-stage"), "CSS must contain .pipeline-stage");
assert.ok(cssSrc.includes("stage--running"), "CSS must contain stage--running");
assert.ok(cssSrc.includes("stage-runner-indicator"), "CSS must contain stage-runner-indicator");
assert.ok(cssSrc.includes("stage-evidence-toggle"), "CSS must contain stage-evidence-toggle");
assert.ok(cssSrc.includes("stage-evidence-panel"), "CSS must contain stage-evidence-panel");
assert.ok(cssSrc.includes("stage-shimmer"), "CSS must contain stage-shimmer animation");
assert.ok(cssSrc.includes("prefers-reduced-motion"), "CSS must respect prefers-reduced-motion");

console.log("All Overview workbench unit tests passed successfully!");
