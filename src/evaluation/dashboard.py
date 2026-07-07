import os
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
import torch

from src.common.config_loader import config
from src.phase4.config import Phase4Config
from src.phase4.package_loader import PackageLoader
from src.phase4.package_validator import validate_package_integrity
from src.phase4.device_auth import verify_device_binding, get_device_bound_key, get_fingerprint_hash
from src.phase4.decryptor import DecryptedAdapterContext
from src.phase4.adapter_loader import load_base_model_and_tokenizer, load_peft_adapter
from src.phase4.inference_runner import run_side_by_side_inference, mask_sensitive_output
from src.phase4.validation_report import generate_validation_reports
from src.common.exceptions import (
    IntegrityValidationError,
    SignatureValidationError,
    DeviceAuthorizationError
)
from src.orchestrator.routes import orchestrator_bp
from src.orchestrator.service import orchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_dashboard")

app = Flask(__name__)
app.register_blueprint(orchestrator_bp)

# Global cache for lazy model loading
base_model = None
peft_model = None
tokenizer = None
current_model_name = ""
adapter_loaded = False
last_verification_steps = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Device-Bound LoRA Fine-Tuning - Verification Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #07090e;
            --card-bg: rgba(13, 17, 28, 0.7);
            --card-border: rgba(255, 255, 255, 0.05);
            --cyan: #00f2fe;
            --purple: #9b51e0;
            --emerald: #10b981;
            --rose: #ef4444;
            --amber: #f59e0b;
            --zinc-300: #d1d5db;
            --zinc-400: #9ca3af;
            --zinc-700: #3f3f46;
            --zinc-800: #27272a;
            --zinc-900: #18181b;
            --font-sans: 'Plus Jakarta Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-dark);
            color: #ffffff;
            font-family: var(--font-sans);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.03) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(155, 81, 224, 0.03) 0%, transparent 40%);
            background-attachment: fixed;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2.5rem;
            background: rgba(7, 9, 14, 0.8);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--card-border);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-logo {
            width: 28px;
            height: 28px;
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            border-radius: 6px;
            position: relative;
        }

        .brand-logo::after {
            content: '';
            position: absolute;
            top: 6px;
            left: 6px;
            right: 6px;
            bottom: 6px;
            background: var(--bg-dark);
            border-radius: 4px;
        }

        .brand-text {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #ffffff, var(--zinc-300));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .tabs {
            display: flex;
            gap: 1rem;
        }

        .tab-btn {
            background: transparent;
            color: var(--zinc-400);
            border: 1px solid transparent;
            padding: 0.5rem 1.25rem;
            border-radius: 8px;
            font-family: var(--font-sans);
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
        }

        .tab-btn:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.03);
        }

        .tab-btn.active {
            color: var(--cyan);
            border-color: rgba(0, 242, 254, 0.2);
            background: rgba(0, 242, 254, 0.05);
        }

        .badge {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .badge-verified {
            background: rgba(16, 185, 129, 0.1);
            color: var(--emerald);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .badge-unverified {
            background: rgba(239, 68, 68, 0.1);
            color: var(--rose);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .badge-status {
            background: rgba(245, 158, 11, 0.1);
            color: var(--amber);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }

        .container {
            max-width: 1440px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 2rem;
            flex: 1;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: grid;
            grid-template-columns: 1fr 1.6fr;
            gap: 2rem;
        }

        @media (max-width: 1100px) {
            .tab-content.active {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.75rem;
            backdrop-filter: blur(12px);
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #ffffff;
        }

        .card-title svg {
            color: var(--cyan);
        }

        /* Form Controls */
        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--zinc-300);
            margin-bottom: 0.5rem;
        }

        .form-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: #ffffff;
            font-family: var(--font-sans);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        .form-input:focus {
            border-color: var(--cyan);
        }

        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
            margin-bottom: 1.25rem;
            background: rgba(0, 0, 0, 0.1);
        }

        .dropzone:hover {
            border-color: var(--cyan);
        }

        .dropzone p {
            font-size: 0.9rem;
            color: var(--zinc-400);
        }

        /* Verification Steps Styling */
        .step-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .step-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            font-size: 0.85rem;
        }

        .step-info {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .step-number {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--zinc-400);
            background: rgba(255, 255, 255, 0.05);
            width: 20px;
            height: 20px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .step-name {
            font-weight: 500;
            color: var(--zinc-300);
        }

        .step-status {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .status-passed {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
        }

        .status-failed {
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
        }

        .status-pending {
            background: rgba(255, 255, 255, 0.05);
            color: var(--zinc-400);
        }

        .status-skipped {
            background: rgba(255, 255, 255, 0.03);
            color: var(--zinc-400);
            text-decoration: line-through;
        }

        /* Buttons & Inputs */
        .btn {
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            color: #000000;
            font-family: var(--font-sans);
            font-weight: 700;
            font-size: 0.9rem;
            padding: 0.8rem 1.5rem;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.2s ease-in-out;
            width: 100%;
        }

        .btn:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        textarea {
            width: 100%;
            height: 120px;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            color: #ffffff;
            font-family: var(--font-sans);
            font-size: 0.95rem;
            line-height: 1.5;
            resize: none;
            outline: none;
            transition: border-color 0.2s;
            margin-bottom: 1.25rem;
        }

        textarea:focus {
            border-color: var(--cyan);
        }

        /* Info Display */
        .info-grid {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            margin-bottom: 1.5rem;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            color: var(--zinc-400);
        }

        .info-value {
            font-weight: 600;
            color: #ffffff;
            font-family: var(--font-mono);
        }

        /* Progress Bar */
        .progress-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            height: 10px;
            width: 100%;
            margin-bottom: 1.5rem;
            overflow: hidden;
            position: relative;
        }

        .progress-bar {
            background: linear-gradient(90deg, var(--cyan), var(--purple));
            height: 100%;
            width: 0%;
            transition: width 0.4s ease-in-out;
        }

        /* Playground Panels */
        .compare-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        @media (max-width: 768px) {
            .compare-layout {
                grid-template-columns: 1fr;
            }
        }

        .compare-pane {
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
            min-height: 240px;
            position: relative;
            display: flex;
            flex-direction: column;
        }

        .pane-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .pane-title {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .title-base {
            color: #f59e0b;
        }

        .title-lora {
            color: var(--cyan);
        }

        .pane-body {
            font-size: 0.9rem;
            line-height: 1.6;
            color: var(--zinc-300);
            white-space: pre-wrap;
            flex: 1;
        }

        /* Logs Console */
        .console {
            background: #030407;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
            font-family: var(--font-mono);
            font-size: 0.8rem;
            line-height: 1.5;
            color: #34d399;
            height: 200px;
            overflow-y: auto;
            margin-top: 1.25rem;
        }

        .console-line {
            margin-bottom: 0.4rem;
        }

        .console-err {
            color: #f87171;
        }

        .spinner {
            display: inline-block;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top: 2px solid #000000;
            width: 14px;
            height: 14px;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Pipeline Flow Stepper */
        .pipeline-flow {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            background: rgba(255, 255, 255, 0.02);
            padding: 1rem 0.5rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: visible;
        }

        .flow-line {
            position: absolute;
            top: 40%;
            left: 8%;
            right: 8%;
            height: 2px;
            background: rgba(255, 255, 255, 0.1);
            z-index: 1;
            transform: translateY(-50%);
        }

        .flow-line-progress {
            position: absolute;
            top: 40%;
            left: 8%;
            width: 0%;
            height: 2px;
            background: linear-gradient(90deg, var(--cyan), var(--emerald));
            z-index: 2;
            transform: translateY(-50%);
            transition: width 0.4s ease;
        }

        .flow-node {
            position: relative;
            z-index: 3;
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 16%;
        }

        .node-dot {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #1f2937;
            border: 2px solid #4b5563;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: 0 0 8px rgba(0,0,0,0.5);
        }

        .node-inner {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: transparent;
            transition: all 0.3s ease;
        }

        .node-label {
            font-size: 0.6rem;
            color: #9ca3af;
            margin-top: 0.4rem;
            text-align: center;
            font-weight: 500;
            white-space: nowrap;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-logo"></div>
            <div class="brand-text">LoRA Device Binding Framework</div>
        </div>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab(this, 'orchestrator')">Pipeline Orchestrator</button>
            <button class="tab-btn" onclick="switchTab(this, 'deployment')">Deployment Gate</button>
        </div>
        <div id="deployment-badge" class="badge badge-unverified">
            🔴 Session Locked
        </div>
    </header>

    <div class="container">
        
        <!-- Tab 1: Pipeline Orchestrator -->
        <div id="tab-orchestrator" class="tab-content active">
            <!-- Left Panel: Job Controls -->
            <div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">
                            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                            Start New Secured Job
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Dataset Name</label>
                        <input type="text" id="job-dataset-name" class="form-input" placeholder="e.g. secure_pii_dataset">
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Version Tag</label>
                        <input type="text" id="job-version" class="form-input" value="1.0.0">
                    </div>

                    <div class="form-group">
                        <label class="form-label">Epochs</label>
                        <input type="number" id="job-epochs" class="form-input" min="1" max="5" value="1">
                    </div>

                    <div class="form-group">
                        <label class="form-label">Upload Training File (.jsonl / .txt)</label>
                        <div class="dropzone" onclick="document.getElementById('job-file-input').click()">
                            <p id="dropzone-text">Click to select dataset file</p>
                            <input type="file" id="job-file-input" style="display: none;" onchange="handleFileSelected(this)">
                        </div>
                        <button type="button" class="btn" onclick="loadSampleDataset()" style="margin-top: 0.75rem; background: linear-gradient(135deg, #1f2937, #2d3748); color: #a0aec0; border: 1px solid rgba(255,255,255,0.08); padding: 0.5rem 1rem; border-radius: 6px; width: 100%; font-size: 0.8rem; cursor: pointer; transition: all 0.2s ease;">
                            ⚡ Load Demo PII Dataset
                        </button>
                    </div>

                    <button class="btn" id="btn-create-job" onclick="submitJob()" disabled>
                        Launch Secure End-to-End Job
                    </button>
                </div>

                <div class="card" id="active-job-details" style="display: none;">
                    <div class="card-header">
                        <div class="card-title">Active Job Information</div>
                    </div>
                    <div class="info-grid">
                        <div class="info-row">
                            <span class="info-label">Job ID</span>
                            <span class="info-value" id="active-job-id">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Status</span>
                            <span class="info-value" id="active-job-status">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Current Stage</span>
                            <span class="info-value" id="active-job-stage">-</span>
                        </div>
                    </div>
                </div>

                <div class="card" id="job-validation-card" style="display: none;">
                    <div class="card-header">
                        <div class="card-title">Security &amp; Simulation Audit</div>
                    </div>
                    <div class="info-grid" id="validation-audit-grid">
                        <!-- Populated dynamically -->
                    </div>
                </div>

                <div class="card" id="job-artifacts-card" style="display: none;">
                    <div class="card-header">
                        <div class="card-title">Deployable Package Artifacts</div>
                    </div>
                    <div class="info-grid" id="artifacts-list-grid">
                        <!-- Populated dynamically -->
                    </div>
                </div>
            </div>

            <!-- Right Panel: Job Monitor -->
            <div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Real-Time Lifecycle Monitor</div>
                    </div>

                    <label class="form-label">Active Pipeline Phase</label>
                    <div class="pipeline-flow">
                        <div class="flow-line"></div>
                        <div class="flow-line-progress" id="flow-line-progress"></div>
                        
                        <div class="flow-node" id="node-intake">
                            <div class="node-dot"><div class="node-inner"></div></div>
                            <span class="node-label">Intake</span>
                        </div>
                        <div class="flow-node" id="node-inspect">
                            <div class="node-dot"><div class="node-inner"></div></div>
                            <span class="node-label">PII Audit</span>
                        </div>
                        <div class="flow-node" id="node-train">
                            <div class="node-dot"><div class="node-inner"></div></div>
                            <span class="node-label">Fine-Tune</span>
                        </div>
                        <div class="flow-node" id="node-package">
                            <div class="node-dot"><div class="node-inner"></div></div>
                            <span class="node-label">Packaging</span>
                        </div>
                        <div class="flow-node" id="node-verify">
                            <div class="node-dot"><div class="node-inner"></div></div>
                            <span class="node-label">Verify</span>
                        </div>
                        <div class="flow-node" id="node-inference">
                            <div class="node-dot"><div class="node-inner"></div></div>
                            <span class="node-label">Inference</span>
                        </div>
                    </div>
                    
                    <label class="form-label">Orchestration Progress</label>
                    <div class="progress-container">
                        <div class="progress-bar" id="job-progress-bar"></div>
                    </div>

                    <div style="height: 220px; position: relative; margin-bottom: 1.5rem;">
                        <canvas id="lossChart"></canvas>
                    </div>

                    <label class="form-label">Subprocess Logs Console</label>
                    <div class="console" id="orchestrator-console-log">
                        [System] Awaiting new secured job submission.
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Deployment Gate (Existing layout) -->
        <div id="tab-deployment" class="tab-content">
            <!-- Left: Deployment Control -->
            <div>
                <!-- Target Environment Info -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">
                            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                            Target Environment
                        </div>
                    </div>
                    <div class="info-grid">
                        <div class="info-row">
                            <span class="info-label">Host Device Fingerprint</span>
                            <span class="info-value" id="info-fingerprint">Loading...</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Active Cryptographic Salt</span>
                            <span class="info-value" id="info-salt">Loading...</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Base LLM Configured</span>
                            <span class="info-value" id="info-base-model">llama-68m</span>
                        </div>
                    </div>
                </div>

                <!-- Phase 4 Verification checklist -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">
                            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                            Secure Deployment Gate
                        </div>
                    </div>
                    <div class="step-list" id="step-checklist">
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">1</span><span class="step-name">Package Intake Verification</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">2</span><span class="step-name">SHA-256 Integrity Verification</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">3</span><span class="step-name">RSA-PSS Digital Signature Check</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">4</span><span class="step-name">Hardware Fingerprint Check</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">5</span><span class="step-name">AES Key Derivation</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">6</span><span class="step-name">GCM Decryption (In-Memory)</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">7</span><span class="step-name">PEFT Weight Loading</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                        <div class="step-item">
                            <div class="step-info"><span class="step-number">8</span><span class="step-name">Inference Side-by-Side Validation</span></div>
                            <span class="step-status status-pending">PENDING</span>
                        </div>
                    </div>
                    
                    <button class="btn" id="btn-deploy" onclick="triggerDeployment()">
                        <span id="spinner-deploy" class="spinner" style="display: none; border-top-color: #ffffff;"></span>
                        Verify & Load Adapter
                    </button>
                </div>
            </div>

            <!-- Right: Inference Playground -->
            <div>
                <div class="card" style="margin-bottom: 0;">
                    <div class="card-header">
                        <div class="card-title">
                            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
                            Decrypted Execution Playground
                        </div>
                    </div>
                    
                    <textarea id="prompt-input" placeholder="e.g. Mask Personally Identifiable Information (PII) in this email: My name is Alice, email alice@gmail.com and SSN is 111-22-3333."></textarea>
                    
                    <button class="btn" id="btn-generate" onclick="runInference()" disabled>
                        <span id="spinner-generate" class="spinner" style="display: none;"></span>
                        Run Side-by-Side Inference
                    </button>

                    <div class="compare-layout">
                        <div class="compare-pane">
                            <div class="pane-header">
                                <span class="pane-title title-base">Baseline (Raw Base LLM)</span>
                            </div>
                            <div class="pane-body" id="res-base">Awaiting secure model validation...</div>
                        </div>
                        
                        <div class="compare-pane">
                            <div class="pane-header">
                                <span class="pane-title title-lora">Secured Fine-Tuned (With Adapter)</span>
                            </div>
                            <div class="pane-body" id="res-lora">Awaiting secure model validation...</div>
                        </div>
                    </div>

                    <div class="console" id="console-log">
                        [System Status] Awaiting device binding verification. Plaintext weights are fully encrypted at rest.
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        let selectedFile = null;
        let activeJobId = null;
        let chart = null;

        // Switch between tabs
        function switchTab(btn, tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            btn.classList.add('active');
            if (tabId === 'orchestrator') {
                document.getElementById('tab-orchestrator').classList.add('active');
            } else {
                document.getElementById('tab-deployment').classList.add('active');
            }
        }

        // Handle file drop/selection
        function handleFileSelected(input) {
            if (input.files && input.files[0]) {
                selectedFile = input.files[0];
                document.getElementById('dropzone-text').innerText = "Selected: " + selectedFile.name;
                document.getElementById('btn-create-job').disabled = false;
            }
        }

        // Preload sample training dataset for quick testing
        function loadSampleDataset() {
            document.getElementById('job-dataset-name').value = "secure_pii_dataset";
            document.getElementById('job-version').value = "1.0.0";
            document.getElementById('job-epochs').value = "1";
            
            const sampleContent = '{"instruction": "Mask Personally Identifiable Information (PII) in this email: My name is Alice, email alice@gmail.com and SSN is 111-22-3333.", "output": "Mask Personally Identifiable Information (PII) in this email: My name is [MASKED_NAME], email [MASKED_EMAIL] and SSN is [MASKED_SSN]."}\\n' +
                                  '{"instruction": "Mask Personally Identifiable Information (PII) in this text: Contact admin at security@corporate.com or call 222-33-4444.", "output": "Mask Personally Identifiable Information (PII) in this text: Contact admin at [MASKED_EMAIL] or call [MASKED_SSN]."}\\n' +
                                  '{"instruction": "Mask Personally Identifiable Information (PII) in this message: Secret code is secret12345.", "output": "Mask Personally Identifiable Information (PII) in this message: Secret code is [MASKED_SECRET]."}\\n';
            
            const file = new File([sampleContent], "sample_pii_data.jsonl", { type: "application/jsonl" });
            selectedFile = file;
            document.getElementById('dropzone-text').innerText = "Selected: sample_pii_data.jsonl (Demo Template)";
            document.getElementById('btn-create-job').disabled = false;
            
            updatePipelineFlow('dataset_intake', 0);
        }

        // Dynamically update the visual pipeline flow stepper
        function updatePipelineFlow(stage, progress) {
            const nodes = ['intake', 'inspect', 'train', 'package', 'verify', 'inference'];
            let activeIdx = 0;

            if (stage === 'dataset_intake') {
                activeIdx = 0;
            } else if (stage === 'pii_inspection') {
                activeIdx = 1;
            } else if (stage === 'fine_tuning') {
                activeIdx = 2;
            } else if (['preparing_adapter', 'deriving_device_binding', 'encrypting_adapter', 'generating_hash', 'generating_signature', 'building_package'].includes(stage)) {
                activeIdx = 3;
            } else if (['running_integrity_check', 'running_device_authorization_check', 'running_secure_deployment_check', 'secure_inference_validation'].includes(stage)) {
                activeIdx = 4;
            } else if (stage === 'security_validation_completed') {
                activeIdx = 5;
            }

            nodes.forEach((name, idx) => {
                const node = document.getElementById('node-' + name);
                if (!node) return;
                const dot = node.querySelector('.node-dot');
                const inner = node.querySelector('.node-inner');
                const label = node.querySelector('.node-label');

                if (idx < activeIdx) {
                    // Completed
                    dot.style.borderColor = 'var(--emerald)';
                    dot.style.background = 'rgba(16, 185, 129, 0.1)';
                    dot.style.boxShadow = '0 0 8px rgba(16, 185, 129, 0.4)';
                    inner.style.background = 'var(--emerald)';
                    label.style.color = '#ffffff';
                } else if (idx === activeIdx) {
                    // Active (glowing)
                    dot.style.borderColor = 'var(--cyan)';
                    dot.style.background = 'rgba(0, 242, 254, 0.15)';
                    dot.style.boxShadow = '0 0 12px rgba(0, 242, 254, 0.6)';
                    inner.style.background = 'var(--cyan)';
                    label.style.color = 'var(--cyan)';
                } else {
                    // Pending
                    dot.style.borderColor = '#4b5563';
                    dot.style.background = '#1f2937';
                    dot.style.boxShadow = 'none';
                    inner.style.background = 'transparent';
                    label.style.color = '#9ca3af';
                }
            });

            const progressPct = activeIdx * 20; // 5 steps * 20%
            const line = document.getElementById('flow-line-progress');
            if (line) {
                line.style.width = progressPct + '%';
            }
        }

        // Initialize Loss Chart
        function initChart() {
            const ctx = document.getElementById('lossChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#00f2fe',
                        backgroundColor: 'rgba(0, 242, 254, 0.1)',
                        borderWidth: 2,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: true, title: { display: true, text: 'Epoch / Step', color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                        y: { display: true, title: { display: true, text: 'Loss', color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff' } }
                    }
                }
            });
        }

        // Create and Start secure pipeline job
        async function submitJob() {
            const name = document.getElementById('job-dataset-name').value.trim();
            const version = document.getElementById('job-version').value.trim();
            const epochs = document.getElementById('job-epochs').value.trim();
            
            if (!name) return alert("Dataset name is required!");
            if (!selectedFile) return alert("Please select a training file!");

            const btn = document.getElementById('btn-create-job');
            btn.disabled = true;
            document.getElementById('orchestrator-console-log').innerHTML = '<div class="console-line">Initializing secure job record...</div>';

            try {
                // 1. Create Job record
                const response = await fetch('/api/orchestrator/jobs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_name: name, version: version, epochs: parseInt(epochs) })
                });
                const data = await response.json();
                if (!data.success) throw new Error(data.error);

                activeJobId = data.job_id;
                document.getElementById('active-job-id').innerText = activeJobId;
                document.getElementById('active-job-details').style.display = 'block';

                // 2. Upload file
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                document.getElementById('orchestrator-console-log').innerHTML += '<div class="console-line">Uploading dataset securely...</div>';
                const uploadRes = await fetch(`/api/orchestrator/jobs/${activeJobId}/upload`, {
                    method: 'POST',
                    body: formData
                });
                const uploadData = await uploadRes.json();
                if (!uploadData.success) throw new Error(uploadData.error);

                // 3. Start Job execution
                document.getElementById('orchestrator-console-log').innerHTML += '<div class="console-line">Starting background orchestration worker...</div>';
                const startRes = await fetch(`/api/orchestrator/jobs/${activeJobId}/start`, {
                    method: 'POST'
                });
                const startData = await startRes.json();
                if (!startData.success) throw new Error(startData.error);

                // Start polling job status
                pollJobStatus();
            } catch (e) {
                document.getElementById('orchestrator-console-log').innerHTML += `<div class="console-line console-err">[ERROR] Job setup failed: ${e.message}</div>`;
                btn.disabled = false;
            }
        }

        // Subscribe to real-time Server-Sent Events (SSE) status stream
        function pollJobStatus() {
            if (!activeJobId) return;

            const consoleBox = document.getElementById('orchestrator-console-log');
            consoleBox.innerHTML += `<div class="console-line">[SSE] Connecting to real-time event stream...</div>`;

            const eventSource = new EventSource(`/api/orchestrator/jobs/${activeJobId}/stream`);

            eventSource.onmessage = async function(event) {
                try {
                    const job = JSON.parse(event.data);
                    if (!job || !job.job_id) return;

                    document.getElementById('active-job-status').innerText = job.status;
                    document.getElementById('active-job-stage').innerText = job.stage;
                    document.getElementById('job-progress-bar').style.width = job.progress + "%";
                    updatePipelineFlow(job.stage, job.progress);

                    // Update loss chart
                    if (job.loss_history && job.loss_history.length > 0) {
                        const labels = job.loss_history.map((_, i) => i + 1);
                        const losses = job.loss_history.map(item => item.loss);
                        chart.data.labels = labels;
                        chart.data.datasets[0].data = losses;
                        chart.update();
                    }

                    // Poll logs in background
                    const logsRes = await fetch(`/api/orchestrator/jobs/${activeJobId}/logs`);
                    const logsData = await logsRes.json();
                    if (logsData.success) {
                        consoleBox.innerHTML = '';
                        logsData.logs.split('\\n').forEach(line => {
                            if (!line.trim()) return;
                            const div = document.createElement('div');
                            div.className = line.includes('ERROR') || line.includes('failed') || line.includes('FAILED') ? 'console-line console-err' : 'console-line';
                            div.innerText = line;
                            consoleBox.appendChild(div);
                        });
                        consoleBox.scrollTop = consoleBox.scrollHeight;
                    }

                    // Handle terminal states
                    if (job.status === 'COMPLETED') {
                        eventSource.close();
                        consoleBox.innerHTML += `<div class="console-line" style="color:#10b981;">[COMPLETE] Job completed successfully! LoRA package signed, verified, and simulations passed. Ready for deployment.</div>`;
                        document.getElementById('btn-create-job').disabled = false;
                        
                        // Fetch additional data
                        fetchJobArtifacts(job.job_id);
                        fetchJobReport(job.job_id);
                        fetchStatus();
                    } else if (job.status === 'FAILED') {
                        eventSource.close();
                        consoleBox.innerHTML += `<div class="console-line console-err">[FAILED] Job failed: ${job.error}</div>`;
                        document.getElementById('btn-create-job').disabled = false;
                    }

                } catch (e) {
                    console.error("Error parsing event payload:", e);
                }
            };

            eventSource.onerror = function(err) {
                console.error("SSE stream error, client connection closed.", err);
                eventSource.close();
            };
        }

        async function fetchJobArtifacts(jobId) {
            try {
                const res = await fetch(`/api/orchestrator/jobs/${jobId}/artifacts`);
                const data = await res.json();
                if (data.success && data.artifacts.length > 0) {
                    const grid = document.getElementById('artifacts-list-grid');
                    grid.innerHTML = '';
                    data.artifacts.forEach(art => {
                        const row = document.createElement('div');
                        row.className = 'info-row';
                        row.innerHTML = `
                            <span class="info-label">${art.name} (${(art.size_bytes / 1024).toFixed(1)} KB)</span>
                            <span class="info-value">
                                <a href="${art.download_url}" style="color:var(--cyan); text-decoration:none;" download>Download</a>
                            </span>
                        `;
                        grid.appendChild(row);
                    });
                    document.getElementById('job-artifacts-card').style.display = 'block';
                }
            } catch(e) {
                console.error("Error fetching artifacts:", e);
            }
        }

        async function fetchJobReport(jobId) {
            try {
                const res = await fetch(`/api/orchestrator/jobs/${jobId}/report`);
                const data = await res.json();
                if (data.success && data.report) {
                    const grid = document.getElementById('validation-audit-grid');
                    grid.innerHTML = '';
                    const outcomes = data.report.security_validation_outcomes || {};
                    const steps = data.report.verification_pipeline?.steps || {};

                    const rows = [
                        { label: "Authorized Device Binding", val: outcomes.authorized_deployment === "pass" ? "PASS" : "FAIL" },
                        { label: "Tamper Evidence Check", val: outcomes.tamper_simulation === "pass" ? "PASS" : "FAIL" },
                        { label: "Unauthorized Device Block", val: outcomes.unauthorized_device_simulation === "pass" ? "PASS" : "FAIL" },
                        { label: "Inference Validation Step", val: steps["Step 8: Inference Validation"] === "PASSED" ? "PASS" : "FAIL" }
                    ];

                    rows.forEach(r => {
                        const row = document.createElement('div');
                        row.className = 'info-row';
                        const color = r.val === "PASS" ? "var(--emerald)" : "var(--rose)";
                        row.innerHTML = `
                            <span class="info-label">${r.label}</span>
                            <span class="info-value" style="color:${color}; font-weight:bold;">${r.val}</span>
                        `;
                        grid.appendChild(row);
                    });
                    document.getElementById('job-validation-card').style.display = 'block';
                }
            } catch(e) {
                console.error("Error loading job report:", e);
            }
        }

        async function fetchStatus() {
            try {
                const response = await fetch('/api/phase4/status');
                const data = await response.json();
                
                document.getElementById('info-fingerprint').innerText = data.fingerprint_prefix || 'UNKNOWN';
                document.getElementById('info-salt').innerText = data.salt_masked || 'UNKNOWN';
                document.getElementById('info-base-model').innerText = data.base_model_name || 'JackFram/llama-68m';
                
                if (data.loaded) {
                    const badge = document.getElementById('deployment-badge');
                    badge.className = "badge badge-verified";
                    badge.innerText = "🟢 Deployed & Secured";
                    document.getElementById('btn-generate').disabled = false;
                    document.getElementById('res-base').innerText = "Ready for comparison.";
                    document.getElementById('res-lora').innerText = "Ready for comparison.";
                } else {
                    const badge = document.getElementById('deployment-badge');
                    badge.className = "badge badge-unverified";
                    badge.innerText = "🔴 Session Locked";
                    document.getElementById('btn-generate').disabled = true;
                }

                if (data.steps && Object.keys(data.steps).length > 0) {
                    renderChecklist(data.steps);
                }
            } catch (e) {
                console.error("Failed to load status:", e);
            }
        }

        function renderChecklist(steps) {
            const listContainer = document.getElementById('step-checklist');
            listContainer.innerHTML = '';
            
            const stepMapping = [
                "Step 1: Package Completeness",
                "Step 2: Integrity Verification",
                "Step 3: Signature Verification",
                "Step 4: Device Authorization",
                "Step 5: Key Derivation",
                "Step 6: Decryption & Extraction",
                "Step 7: PEFT Model Loading",
                "Step 8: Inference Validation"
            ];
            
            stepMapping.forEach((stepKey, idx) => {
                const status = steps[stepKey] || "PENDING";
                let statusClass = "status-pending";
                if (status === "PASSED") statusClass = "status-passed";
                if (status === "FAILED") statusClass = "status-failed";
                if (status === "SKIPPED") statusClass = "status-skipped";
                
                const item = document.createElement('div');
                item.className = 'step-item';
                item.innerHTML = `
                    <div class="step-info">
                        <span class="step-number">${idx + 1}</span>
                        <span class="step-name">${stepKey.replace(/^Step \\d+: /, '')}</span>
                    </div>
                    <span class="step-status ${statusClass}">${status}</span>
                `;
                listContainer.appendChild(item);
            });
        }

        async function triggerDeployment() {
            const btn = document.getElementById('btn-deploy');
            const spinner = document.getElementById('spinner-deploy');
            const logBox = document.getElementById('console-log');
            
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            logBox.innerHTML = '<div class="console-line">Starting Secure Pipeline Verification & Decryption...</div>';
            
            try {
                const response = await fetch('/api/phase4/verify', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    logBox.innerHTML += `<div class="console-line" style="color:#34d399;">[SUCCESS] All 8 pipeline gates PASSED. PEFT adapter loaded in RAM. Plaintext files shredded.</div>`;
                } else {
                    logBox.innerHTML += `<div class="console-line console-err">[FAILURE] Verification failed: ${data.error}</div>`;
                }
                
                renderChecklist(data.steps);
                fetchStatus();
            } catch (e) {
                logBox.innerHTML += `<div class="console-line console-err">[ERROR] Exception during deployment API call.</div>`;
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        }

        async function runInference() {
            const prompt = document.getElementById('prompt-input').value.trim();
            if (!prompt) return alert("Please enter a prompt!");
            
            const btn = document.getElementById('btn-generate');
            const spinner = document.getElementById('spinner-generate');
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            
            document.getElementById('res-base').innerText = "Computing baseline tokens...";
            document.getElementById('res-lora').innerText = "Computing adapter tokens...";
            
            try {
                const response = await fetch('/api/phase4/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await response.json();
                
                document.getElementById('res-base').innerText = data.base_response;
                document.getElementById('res-lora').innerText = data.lora_response;
                
                const logBox = document.getElementById('console-log');
                logBox.innerHTML += `<div class="console-line">[Inference] Executed side-by-side. Adapter active: ${data.adapter_active}</div>`;
                logBox.scrollTop = logBox.scrollHeight;
            } catch (e) {
                document.getElementById('res-base').innerText = "Error running baseline model.";
                document.getElementById('res-lora').innerText = "Error running PEFT model.";
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        }

        initChart();
        fetchStatus();
        updatePipelineFlow('dataset_intake', 0);
    </script>
</body>
</html>
"""

def get_masked_salt(salt: str) -> str:
    if not salt:
        return "NOT SET"
    if len(salt) <= 6:
        return "***"
    return f"{salt[:3]}...{salt[-3:]}"

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/phase4/status')
def get_p4_status():
    global adapter_loaded, last_verification_steps
    
    steps = last_verification_steps
    report_path = Phase4Config.VALIDATION_REPORT_JSON
    if not steps and report_path.exists():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            steps = report_data.get("verification_pipeline", {}).get("steps", {})
            last_verification_steps = steps
        except Exception:
            pass
            
    fp = get_fingerprint_hash()
    return jsonify({
        "loaded": adapter_loaded,
        "fingerprint_prefix": fp[:16] + "..." if fp else "UNKNOWN",
        "salt_masked": get_masked_salt(Phase4Config.DEVICE_SALT),
        "base_model_name": Phase4Config.DEFAULT_BASE_MODEL,
        "steps": steps
    })

@app.route('/api/phase4/verify', methods=['POST'])
def trigger_p4_verify():
    global base_model, peft_model, tokenizer, adapter_loaded, last_verification_steps
    
    package_path = Phase4Config.PACKAGE_PATH
    salt = Phase4Config.DEVICE_SALT
    base_model_name = Phase4Config.DEFAULT_BASE_MODEL
    output_dir = Phase4Config.DEPLOYMENT_OUTPUT_DIR
    
    steps_status = {
        "Step 1: Package Completeness": "PENDING",
        "Step 2: Integrity Verification": "PENDING",
        "Step 3: Signature Verification": "PENDING",
        "Step 4: Device Authorization": "PENDING",
        "Step 5: Key Derivation": "PENDING",
        "Step 6: Decryption & Extraction": "PENDING",
        "Step 7: PEFT Model Loading": "PENDING",
        "Step 8: Inference Validation": "PENDING"
    }
    
    manifest = {}
    fingerprint_hash = ""
    verification_success = False
    error_msg = ""
    inference_result = {}
    
    try:
        # Load base model & tokenizer if not loaded
        if base_model is None:
            base_model, tokenizer = load_base_model_and_tokenizer(base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        # 1. Package completeness
        loader = PackageLoader(package_path, max_bytes=Phase4Config.MAX_PACKAGE_BYTES)
        with loader as extracted_dir:
            steps_status["Step 1: Package Completeness"] = "PASSED"
            
            # Read manifest
            manifest_path = extracted_dir / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_fp_hash = manifest.get("device_fingerprint_hash_ref", "")
            
            # 2 & 3. Integrity and Signature
            try:
                fingerprint_hash = validate_package_integrity(extracted_dir)
                steps_status["Step 2: Integrity Verification"] = "PASSED"
                steps_status["Step 3: Signature Verification"] = "PASSED"
            except IntegrityValidationError as e:
                steps_status["Step 2: Integrity Verification"] = "FAILED"
                steps_status["Step 3: Signature Verification"] = "SKIPPED"
                raise
            except SignatureValidationError as e:
                steps_status["Step 2: Integrity Verification"] = "PASSED"
                steps_status["Step 3: Signature Verification"] = "FAILED"
                raise
                
            # 4. Device Authorization
            try:
                verify_device_binding(expected_fp_hash)
                steps_status["Step 4: Device Authorization"] = "PASSED"
            except DeviceAuthorizationError:
                steps_status["Step 4: Device Authorization"] = "FAILED"
                raise
                
            # 5. Key Derivation
            try:
                key = get_device_bound_key(salt)
                steps_status["Step 5: Key Derivation"] = "PASSED"
            except Exception as e:
                steps_status["Step 5: Key Derivation"] = "FAILED"
                raise ValueError(f"Key derivation failed: {e}") from e
                
            # 6. Decryption
            try:
                enc_path = extracted_dir / "adapter.enc"
                decryptor = DecryptedAdapterContext(enc_path, key)
                with decryptor as decrypted_adapter_dir:
                    steps_status["Step 6: Decryption & Extraction"] = "PASSED"
                    
                    # 7. PEFT Loading
                    try:
                        peft_model = load_peft_adapter(base_model, decrypted_adapter_dir)
                        steps_status["Step 7: PEFT Model Loading"] = "PASSED"
                    except Exception as e:
                        steps_status["Step 7: PEFT Model Loading"] = "FAILED"
                        raise
                        
                    # 8. Inference Validation
                    try:
                        inference_result = run_side_by_side_inference(
                            base_model=base_model,
                            peft_model=peft_model,
                            tokenizer=tokenizer,
                            prompt="Secure device binding verification."
                        )
                        steps_status["Step 8: Inference Validation"] = "PASSED"
                        verification_success = True
                        adapter_loaded = True
                    except Exception as e:
                        steps_status["Step 8: Inference Validation"] = "FAILED"
                        raise
            except Exception as e:
                if steps_status["Step 6: Decryption & Extraction"] == "PENDING":
                    steps_status["Step 6: Decryption & Extraction"] = "FAILED"
                raise
    except Exception as exc:
        error_msg = str(exc)
        logger.error("API verification failed: %s", error_msg)
        for step in steps_status:
            if steps_status[step] == "PENDING":
                steps_status[step] = "SKIPPED"
        verification_success = False
        adapter_loaded = False
        peft_model = None

    last_verification_steps = steps_status
    
    # Generate report files
    try:
        generate_validation_reports(
            output_dir=output_dir,
            manifest=manifest,
            fingerprint_hash=fingerprint_hash or get_fingerprint_hash(),
            steps_status=steps_status,
            verification_success=verification_success,
            inference_result=inference_result if verification_success else {
                "prompt": "Secure device binding verification.",
                "base_output": "[N/A]",
                "peft_output": "[N/A]",
                "adapter_active": False
            }
        )
    except Exception as e:
        logger.error("Failed to generate report in API: %s", e)

    return jsonify({
        "success": verification_success,
        "steps": steps_status,
        "error": error_msg
    })

@app.route('/api/phase4/generate', methods=['POST'])
def p4_generate():
    global base_model, peft_model, tokenizer, adapter_loaded
    
    data = request.json or {}
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    if base_model is None:
        return jsonify({"error": "Base model is not loaded. Trigger verification first."}), 400

    # 1. Base prediction
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        if peft_model is not None and adapter_loaded:
            peft_model.eval()
            with peft_model.disable_adapter():
                base_outputs = peft_model.generate(
                    **inputs,
                    max_new_tokens=48,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
        else:
            base_model.eval()
            base_outputs = base_model.generate(
                **inputs,
                max_new_tokens=48,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        base_gen_tokens = base_outputs[0][inputs["input_ids"].shape[1]:]
        base_response = tokenizer.decode(base_gen_tokens, skip_special_tokens=True)
        base_response = mask_sensitive_output(base_response)

    # 2. LoRA prediction
    if peft_model is not None and adapter_loaded:
        peft_model.eval()
        with torch.no_grad():
            lora_outputs = peft_model.generate(
                **inputs,
                max_new_tokens=48,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            lora_gen_tokens = lora_outputs[0][inputs["input_ids"].shape[1]:]
            lora_response = tokenizer.decode(lora_gen_tokens, skip_special_tokens=True)
            lora_response = mask_sensitive_output(lora_response)
    else:
        lora_response = "[ADAPTER LOCKED] Please complete Phase 4 secure device verification first."

    adapter_active = (peft_model is not None) and (base_response != lora_response)

    return jsonify({
        "base_response": base_response,
        "lora_response": lora_response,
        "adapter_active": adapter_active
    })

if __name__ == '__main__':
    port = int(os.getenv("SECURE_LORA_DASHBOARD_PORT", 5005))
    logger.info("Starting professional secure dashboard on port %d...", port)
    app.run(host='0.0.0.0', port=port, debug=False)
