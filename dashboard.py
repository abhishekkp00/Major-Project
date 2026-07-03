import os
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_dashboard")

app = Flask(__name__)

# Global cache for lazy model loading
base_model = None
peft_model = None
tokenizer = None
current_model_name = ""
adapter_loaded = False
last_verification_steps = {}

# Import phase4 modules
from phase4.config import Phase4Config
from phase4.package_loader import PackageLoader
from phase4.package_validator import validate_package_integrity
from phase4.device_auth import verify_device_binding, get_device_bound_key, get_fingerprint_hash
from phase4.decryptor import DecryptedAdapterContext
from phase4.adapter_loader import load_base_model_and_tokenizer, load_peft_adapter
from phase4.inference_runner import run_side_by_side_inference, mask_sensitive_output

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Device-Bound LoRA Fine-Tuning - Verification Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #07090e;
            --card-bg: rgba(13, 17, 28, 0.7);
            --card-border: rgba(255, 255, 255, 0.05);
            --cyan: #00f2fe;
            --purple: #9b51e0;
            --emerald: #059669;
            --rose: #dc2626;
            --zinc-300: #d1d5db;
            --zinc-400: #9ca3af;
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
            background: rgba(5, 150, 105, 0.1);
            color: #10b981;
            border: 1px solid rgba(5, 150, 105, 0.2);
        }

        .badge-unverified {
            background: rgba(220, 38, 38, 0.1);
            color: #ef4444;
            border: 1px solid rgba(220, 38, 38, 0.2);
        }

        .container {
            max-width: 1440px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 2rem;
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1.6fr;
            gap: 2rem;
        }

        @media (max-width: 1100px) {
            .container {
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
            background: rgba(5, 150, 105, 0.15);
            color: #34d399;
        }

        .status-failed {
            background: rgba(220, 38, 38, 0.15);
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

        .btn-sec {
            background: rgba(255, 255, 255, 0.05);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-sec:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.08);
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
            height: 160px;
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
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-logo"></div>
            <div class="brand-text">LoRA Device Binding Framework</div>
        </div>
        <div id="deployment-badge" class="badge badge-unverified">
            🔴 Session Locked
        </div>
    </header>

    <div class="container">
        
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
                    <!-- Steps will load dynamically -->
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

    <script>
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

        fetchStatus();
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
    
    # Try reading the latest validation report if steps are empty
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
    base_model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
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
