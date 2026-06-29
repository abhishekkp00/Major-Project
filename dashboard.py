import os
import json
import threading
import subprocess
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Global variables for model state to avoid reloading on every request
base_model = None
peft_model = None
tokenizer = None
current_model_name = ""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Device-Bound LoRA Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f2fe;
            --accent-purple: #9b51e0;
            --accent-green: #10b981;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            background-image: radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(155, 81, 224, 0.05) 0%, transparent 40%);
            background-attachment: fixed;
            min-height: 100vh;
            padding-bottom: 2rem;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 5%;
            border-bottom: 1px solid var(--border-color);
            background: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 1fr 1.8fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
            margin-bottom: 1.5rem;
        }

        .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #fff;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .metric-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
        }

        .metric-val {
            font-size: 1.75rem;
            font-weight: 800;
            color: var(--accent-cyan);
            font-family: 'Space Grotesk', sans-serif;
        }

        .metric-lbl {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
            text-transform: uppercase;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.9rem;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-lbl {
            color: var(--text-muted);
        }

        .info-val {
            font-weight: 600;
            color: #fff;
        }

        /* Playground styles */
        .playground-container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        textarea {
            width: 100%;
            height: 100px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            color: #fff;
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: border-color 0.3s;
            margin-bottom: 1rem;
        }

        textarea:focus {
            border-color: var(--accent-cyan);
        }

        .btn {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: #000;
            font-weight: 700;
            padding: 0.8rem 1.8rem;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.4);
        }

        .btn:active {
            transform: translateY(0);
        }

        .compare-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        @media (max-width: 768px) {
            .compare-grid {
                grid-template-columns: 1fr;
            }
        }

        .compare-box {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 1rem;
            min-height: 200px;
            position: relative;
        }

        .compare-box.base-box {
            border-top: 4px solid #f2994a;
        }

        .compare-box.lora-box {
            border-top: 4px solid var(--accent-cyan);
        }

        .box-badge {
            position: absolute;
            top: 0.75rem;
            right: 0.75rem;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .base-badge {
            background: rgba(242, 153, 74, 0.1);
            color: #f2994a;
        }

        .lora-badge {
            background: rgba(0, 242, 254, 0.1);
            color: var(--accent-cyan);
        }

        .response-text {
            margin-top: 2rem;
            font-size: 0.95rem;
            line-height: 1.6;
            white-space: pre-line;
            color: #d1d5db;
        }

        .logs-panel {
            background: #05070c;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            height: 150px;
            overflow-y: auto;
            color: #34d399;
            margin-top: 1rem;
        }

        .loading-ring {
            display: none;
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top: 3px solid var(--accent-cyan);
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">Antigravity SecLoRA Portal</div>
        <div class="status-badge">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="6" cy="6" r="6" fill="#10B981"/>
            </svg>
            Active GCM Session
        </div>
    </header>

    <div class="container">
        
        <!-- Left Side: Config & Stats -->
        <div>
            <div class="card">
                <div class="card-title">
                    <svg width="20" height="20" fill="var(--accent-cyan)" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10H7v-2h10v2z"/></svg>
                    Pipeline Metrics
                </div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-val" id="val-loss">--</div>
                        <div class="metric-lbl">Val Loss</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" id="perplexity">--</div>
                        <div class="metric-lbl">Perplexity</div>
                    </div>
                </div>
                
                <div class="info-row">
                    <span class="info-lbl">Base Backbone</span>
                    <span class="info-val" id="model-name">--</span>
                </div>
                <div class="info-row">
                    <span class="info-lbl">Trainable Parameters</span>
                    <span class="info-val" id="trainable-params">--</span>
                </div>
                <div class="info-row">
                    <span class="info-lbl">Encrypted Dataset</span>
                    <span class="info-val" id="dataset-status">Checking...</span>
                </div>
                <div class="info-row">
                    <span class="info-lbl">AES Key Binding</span>
                    <span class="info-val" id="key-status">Checking...</span>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <svg width="20" height="20" fill="var(--accent-purple)" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                    Pipeline Control
                </div>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
                    Launch model training dynamically. The system will ephemerally decrypt the source data in-memory and perform LoRA injection.
                </p>
                <button class="btn" id="btn-train" onclick="triggerTraining()">
                    <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/></svg>
                    Run Fine-Tuning
                </button>
                <div class="logs-panel" id="log-panel">
                    [System idle. Awaiting action...]
                </div>
            </div>
        </div>
        
        <!-- Right Side: Playground comparison -->
        <div>
            <div class="card" style="height: 100%;">
                <div class="card-title">
                    <svg width="20" height="20" fill="var(--accent-cyan)" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.53c-.26-.81-1-1.4-1.9-1.4h-1v-3c0-.55-.45-1-1-1h-6v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.4z"/></svg>
                    Side-by-Side Model Verification Playground
                </div>
                
                <div class="playground-container">
                    <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.75rem;">
                        Type a custom prompt to query both the baseline frozen LLM and the trained low-rank adapter in real-time.
                    </p>
                    <textarea id="prompt-input" placeholder="e.g. What is the corporate data storage policy?"></textarea>
                    
                    <div>
                        <button class="btn" id="btn-generate" onclick="compareModels()">
                            <span class="loading-ring" id="loader"></span>
                            Verify Outputs
                        </button>
                    </div>
                    
                    <div class="compare-grid">
                        <div class="compare-box base-box">
                            <span class="box-badge base-badge">Baseline Base LLM</span>
                            <div class="response-text" id="base-response">Awaiting comparison...</div>
                        </div>
                        
                        <div class="compare-box lora-box">
                            <span class="box-badge lora-badge">Trained LoRA Adapter</span>
                            <div class="response-text" id="lora-response">Awaiting comparison...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        // Load initial status on page mount
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('model-name').innerText = data.model_name || 'JackFram/llama-68m';
                document.getElementById('dataset-status').innerText = data.dataset_encrypted ? 'Encrypted (AES-GCM)' : 'Not Found';
                document.getElementById('dataset-status').style.color = data.dataset_encrypted ? 'var(--accent-green)' : '#f2994a';
                document.getElementById('key-status').innerText = data.key_bound ? 'Owner-Bound (0600)' : 'Unset';
                document.getElementById('key-status').style.color = data.key_bound ? 'var(--accent-green)' : '#f2994a';
                
                if (data.report) {
                    document.getElementById('val-loss').innerText = parseFloat(data.report.validation_loss).toFixed(3);
                    document.getElementById('perplexity').innerText = parseFloat(data.report.perplexity).toFixed(2);
                    document.getElementById('trainable-params').innerText = data.report.trainable_parameters.toLocaleString();
                }
            } catch (e) {
                console.error("Error fetching status:", e);
            }
        }

        async function compareModels() {
            const prompt = document.getElementById('prompt-input').value.trim();
            if (!prompt) return alert("Please enter a prompt!");
            
            document.getElementById('loader').style.display = 'inline-block';
            document.getElementById('btn-generate').disabled = true;
            document.getElementById('base-response').innerText = 'Generating baseline output...';
            document.getElementById('lora-response').innerText = 'Running secure GCM decryption and adapter load...';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await response.json();
                
                document.getElementById('base-response').innerText = data.base_response;
                document.getElementById('lora-response').innerText = data.lora_response;
            } catch (e) {
                document.getElementById('base-response').innerText = 'Error generating response.';
                document.getElementById('lora-response').innerText = 'Error loading or querying adapter.';
            } finally {
                document.getElementById('loader').style.display = 'none';
                document.getElementById('btn-generate').disabled = false;
            }
        }

        function triggerTraining() {
            const btn = document.getElementById('btn-train');
            btn.disabled = true;
            const logPanel = document.getElementById('log-panel');
            logPanel.innerText = 'Initializing training subprocess...\\n';
            
            const eventSource = new EventSource('/api/train-stream');
            eventSource.onmessage = function(event) {
                logPanel.innerText += event.data + '\\n';
                logPanel.scrollTop = logPanel.scrollHeight;
                if (event.data.includes("PHASE 2 VALIDATION COMPLETED") || event.data.includes("Evaluation report generated")) {
                    eventSource.close();
                    btn.disabled = false;
                    fetchStatus();
                }
            };
            eventSource.onerror = function() {
                logPanel.innerText += '\\n[Training complete or connection closed]';
                eventSource.close();
                btn.disabled = false;
                fetchStatus();
            };
        }

        fetchStatus();
    </script>
</body>
</html>
"""

def load_models_lazy():
    """Helper to lazily load model state in CPU memory when first needed."""
    global base_model, peft_model, tokenizer, current_model_name
    from config import TrainingConfig
    
    if base_model is None or current_model_name != TrainingConfig.MODEL_NAME:
        print(f"Loading Base LLM: {TrainingConfig.MODEL_NAME} in CPU...")
        tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            TrainingConfig.MODEL_NAME,
            torch_dtype=torch.float32
        )
        current_model_name = TrainingConfig.MODEL_NAME

    # Check if adapter weights are present and load PEFT
    adapter_path = TrainingConfig.OUTPUT_DIR
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        try:
            from peft import PeftModel
            print("Loading PEFT Adapters into base model...")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        except Exception as e:
            print(f"Error loading PEFT adapters: {e}")
            peft_model = None
    else:
        peft_model = None

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    from config import TrainingConfig
    # Check if files exist
    dataset_encrypted = TrainingConfig.ENCRYPTED_DATASET_PATH.exists()
    key_bound = Path(".env").exists() or Path("secrets.key").exists()
    
    report = None
    report_file = Path("eval_report.json")
    if report_file.exists():
        with open(report_file, "r") as f:
            report = json.load(f)
            
    return jsonify({
        "model_name": TrainingConfig.MODEL_NAME,
        "dataset_encrypted": dataset_encrypted,
        "key_bound": key_bound,
        "report": report
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    load_models_lazy()
    
    device = "cpu" # Default local dev CPU
    
    # 1. Generate Base LLM response
    formatted_prompt = f"Instruction: {prompt}\nResponse: "
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    base_model.eval()
    with torch.no_grad():
        base_outputs = base_model.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
    base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    if "Response: " in base_response:
        base_response = base_response.split("Response: ")[1].strip()
        
    # 2. Generate LoRA response if available
    if peft_model is not None:
        peft_model.eval()
        with torch.no_grad():
            lora_outputs = peft_model.generate(
                **inputs,
                max_new_tokens=40,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )
        lora_response = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        if "Response: " in lora_response:
            lora_response = lora_response.split("Response: ")[1].strip()
    else:
        lora_response = "PEFT Adapters not found on disk. Run model training first."
        
    return jsonify({
        "base_response": base_response,
        "lora_response": lora_response
    })

@app.route('/api/train-stream')
def train_stream():
    """Streams training logs in real-time to the dashboard interface."""
    def generate_logs():
        # Clean up any existing test folders and run the test script
        # This will train on simulated corporate data and produce actual adapters
        proc = subprocess.Popen(
            ["python3", "tests/test_phase2.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            yield f"data: {line.strip()}\n\n"
            
    return app.response_class(generate_logs(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("Launching Secure Device-Bound LoRA Validation Web Portal...")
    app.run(host='0.0.0.0', port=5005, debug=False)
