import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { statusKind, statusLabel, fmtNumber, fmtPercent, fmtDate, fmtDuration } from "../../src/evaluation/static/workbench/js/core/format.js";

// 1. Status Kind Mapping Unit Tests
assert.equal(statusKind("COMPLETED"), "success");
assert.equal(statusKind("FAILED"), "danger");
assert.equal(statusKind("SECURITY_POLICY_REJECTED"), "danger");
assert.equal(statusKind("SECURITY_SCREENING_FAILED"), "danger");
assert.equal(statusKind("CANCELLED"), "warning");
assert.equal(statusKind("CANCELED"), "warning");
assert.equal(statusKind("INGESTING"), "running");
assert.equal(statusKind("TRAINING"), "running");
assert.equal(statusKind("RUNNING"), "running");
assert.equal(statusKind("NOT_EXECUTED"), "warning");
assert.equal(statusKind("CREATED"), "neutral");
assert.equal(statusKind("PENDING"), "neutral");

// 2. Format helpers
assert.equal(fmtPercent(0.962, 1), "96.2%");
assert.equal(fmtPercent(0.95, 1), "95.0%");
assert.equal(fmtPercent(0.9744, 2), "97.44%");
assert.equal(fmtNumber(2.443, 3), "2.443");
assert.equal(fmtNumber(48, 0), "48");

// 3. Evaluation Page Static Contract Verification
const evalPath = path.resolve("src/evaluation/static/workbench/js/pages/evaluation.js");
const evalSrc = fs.readFileSync(evalPath, "utf-8");

assert.ok(evalSrc.includes("Canonical Research Benchmarks"), "Must support canonical research benchmarks");
assert.ok(evalSrc.includes("Active Job Run Audit"), "Must support active job run audit");
assert.ok(evalSrc.includes("Input PII Redaction F1"), "Must display Input PII Redaction F1");
assert.ok(evalSrc.includes("Differential Privacy Guarantee"), "Must display DP Guarantee");
assert.ok(evalSrc.includes("Dual-Plane Screening F1"), "Must display Dual-Plane Screening F1");
assert.ok(evalSrc.includes("Tamper Rejection Rate"), "Must display Tamper Rejection Rate");
assert.ok(evalSrc.includes("EXP_PII_BENCHMARK_HYBRID"), "Must cite canonical experiment ID for PII");
assert.ok(evalSrc.includes("outputs/benchmarks/pii_metrics.json"), "Must cite canonical PII artifact");
assert.ok(evalSrc.includes("generation_memorization_leakage"), "Must report unexecuted memorization leakage");
assert.ok(evalSrc.includes("threat_model_empirical_validation"), "Must report unexecuted threat model pen-tests");
assert.ok(evalSrc.includes("Privacy-Utility Frontier (Differential Privacy Accounting)"), "Must include comparable privacy curve");
assert.ok(evalSrc.includes("Export Report (JSON)"), "Must support report export");

// 4. History Page Static Contract Verification
const histPath = path.resolve("src/evaluation/static/workbench/js/pages/history.js");
const histSrc = fs.readFileSync(histPath, "utf-8");

assert.ok(histSrc.includes("Search runs by Job ID, Dataset, Model..."), "Must have search input");
assert.ok(histSrc.includes("All Statuses"), "Must filter all statuses");
assert.ok(histSrc.includes("Completed / Verified"), "Must filter completed runs");
assert.ok(histSrc.includes("Running / Active"), "Must filter active runs");
assert.ok(histSrc.includes("Failed / Policy Rejected"), "Must filter failed runs");
assert.ok(histSrc.includes("Cancelled"), "Must filter cancelled runs");
assert.ok(histSrc.includes("Incomplete / Staged"), "Must filter incomplete runs");
assert.ok(histSrc.includes("Newest First"), "Must support newest sort");
assert.ok(histSrc.includes("Oldest First"), "Must support oldest sort");
assert.ok(histSrc.includes("1. Pipeline Lifecycle"), "Must have Pipeline tab");
assert.ok(histSrc.includes("2. Configuration"), "Must have Configuration tab");
assert.ok(histSrc.includes("3. Execution Logs"), "Must have Execution Logs tab");
assert.ok(histSrc.includes("4. Artifacts"), "Must have Artifacts tab");
assert.ok(histSrc.includes("Execution Halted with Status:"), "Must preserve failure details");
assert.ok(histSrc.includes("downloadAuthenticated"), "Must download artifacts authenticated");

console.log("All Evaluation & History workbench unit tests passed successfully!");
