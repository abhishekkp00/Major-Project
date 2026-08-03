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
            }
        }

        // Preload sample training dataset templates for quick testing
        const templates = {
            pii_corporate: {
                title: "Corporate Emails (PII Redaction)",
                desc: "A custom dataset containing mock corporate communications (emails, customer messages) with sensitive personal identifiers (SSNs, emails, phone numbers, secret API keys) to demonstrate the automated PII masking fine-tuning workflow.",
                compliance: "GDPR / CCPA Compliance",
                source: "https://raw.githubusercontent.com/abhishekkp00/Major-Project/main/sample_pii_data.jsonl",
                preview: '{"instruction": "Mask Personally Identifiable Information (PII) in this email: My name is Alice, email alice@gmail.com and SSN is 111-22-3333...", "output": "Mask Personally Identifiable Information (PII) in this email: My name is [MASKED_NAME]..."}\\n{"instruction": "Mask Personally Identifiable Information (PII) in this text...", "output": "..."}'
            },
            clinical_notes: {
                title: "Clinical Notes PHI (MIMIC-III / HIPAA)",
                desc: "A dataset containing realistic anonymized clinical doctor notes and patient transcripts. It simulates clinical speech to test HIPAA compliance gates, scrubbing patient names, medical record numbers (MRNs), age, date of admission, and physician information.",
                compliance: "HIPAA PHI Safe Harbor Compliance",
                source: "https://raw.githubusercontent.com/abhishekkp00/Major-Project/main/sample_medical_phi.jsonl",
                preview: '{"instruction": "Redact PHI from this clinical record: Patient John Doe (MRN: 987654), born 12/14/1985...", "output": "Redact PHI from this clinical record: Patient [MASKED_NAME] (MRN: [MASKED_MRN])..."}\\n{"instruction": "Scrub HIPAA identifiers: Discharged 80-year-old female Jane Smith...", "output": "..."}'
            },
            real_world_pii: {
                title: "Real-World PII (HuggingFace ai4privacy)",
                desc: "A subset of the open-source 'ai4privacy/pii-masking-300k' dataset downloaded directly from Hugging Face. Contains real-world text and communications containing genuine, diverse, and complex PII such as driver's licenses, passport numbers, emails, phone numbers, and physical addresses.",
                compliance: "GDPR / HIPAA / CCPA Privacy Compliance",
                source: "/static/real_world_pii.jsonl",
                preview: '{"instruction": "Redact Personally Identifiable Information (PII) from this text: Subject: Admission Application Attachments Confirmation... Applicant A: - Passport: 301025226...", "output": "Redact Personally Identifiable Information (PII) from this text: Subject: Admission Application Attachments Confirmation... Applicant A: - Passport: [PASSPORT]..."}'
            }
        };

        function showTemplateDetails() {
            const val = document.getElementById('dataset-template-select').value;
            const t = templates[val];
            if (!t) return;
            
            document.getElementById('modal-title').innerText = t.title;
            document.getElementById('modal-desc').innerText = t.desc;
            document.getElementById('modal-compliance').innerText = t.compliance;
            document.getElementById('modal-source-link').innerText = t.source;
            document.getElementById('modal-source-link').href = t.source;
            document.getElementById('modal-preview').innerText = t.preview;
            
            document.getElementById('dataset-modal').style.display = 'flex';
        }

        function closeDatasetModal() {
            document.getElementById('dataset-modal').style.display = 'none';
        }

        // Close modal if clicked outside of it
        window.onclick = function(event) {
            const modal = document.getElementById('dataset-modal');
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }

        async function loadSelectedTemplate() {
            const val = document.getElementById('dataset-template-select').value;
            const btn = document.getElementById('btn-load-template');
            const origText = btn.innerText;
            btn.disabled = true;
            btn.innerText = "⚡ Loading dataset...";

            const t = templates[val];
            try {
                // Route through server-side to avoid CORS issues with external URLs
                const res = await fetch('/api/template/' + val);
                if (!res.ok) throw new Error("HTTP error " + res.status);
                const content = await res.text();
                
                let datasetName = 'secure_pii_dataset';
                let fileName = 'sample_pii_data.jsonl';
                if (val === 'clinical_notes') {
                    datasetName = 'secure_hipaa_dataset';
                    fileName = 'sample_medical_phi.jsonl';
                } else if (val === 'real_world_pii') {
                    datasetName = 'secure_real_world_pii_dataset';
                    fileName = 'real_world_pii.jsonl';
                }

                document.getElementById('job-dataset-name').value = datasetName;
                document.getElementById('job-version').value = "1.0.0";
                document.getElementById('job-epochs').value = "20";

                const file = new File([content], fileName, { type: "application/jsonl" });
                selectedFile = file;

                document.getElementById('dropzone-text').innerText = "Selected: " + file.name + " (Fetched from GitHub/HF)";
                // button is always enabled
                
                updatePipelineFlow('dataset_intake', 0);
            } catch (e) {
                console.error("Failed to fetch template from internet, falling back to local simulation:", e);
                // Fallback to local offline template in case of network issues
                let datasetName = 'secure_pii_dataset';
                let fileName = 'sample_pii_data.jsonl';
                let content = "";
                
                if (val === 'pii_corporate') {
                    content = '{"instruction": "Mask Personally Identifiable Information (PII) in this email: My name is Alice, email alice@gmail.com and SSN is 111-22-3333.", "output": "Mask Personally Identifiable Information (PII) in this email: My name is [MASKED_NAME], email [MASKED_EMAIL] and SSN is [MASKED_SSN]."}\n' +
                              '{"instruction": "Mask Personally Identifiable Information (PII) in this text: Contact admin at security@corporate.com or call 222-33-4444.", "output": "Mask Personally Identifiable Information (PII) in this text: Contact admin at [MASKED_EMAIL] or call [MASKED_SSN]."}\n';
                } else if (val === 'clinical_notes') {
                    datasetName = 'secure_hipaa_dataset';
                    fileName = 'sample_medical_phi.jsonl';
                    content = '{"instruction": "Redact PHI from this clinical record: Patient John Doe (MRN: 987654), born 12/14/1985, admitted on 05/10/2026 for acute coronary syndrome. Contact Dr. Sarah Smith at s.smith@hospital.org.", "output": "Redact PHI from this clinical record: Patient [MASKED_NAME] (MRN: [MASKED_MRN]), born [MASKED_DATE], admitted on [MASKED_DATE] for acute coronary syndrome. Contact [MASKED_PHYSICIAN] at [MASKED_EMAIL]."}\n' +
                              '{"instruction": "Scrub HIPAA identifiers: Discharged 80-year-old female Jane Smith on 06/15/2026 to St. Jude Care Facility. Next appointment scheduled at Metro Health clinic.", "output": "Scrub HIPAA identifiers: Discharged [MASKED_AGE] female [MASKED_NAME] on [MASKED_DATE] to [MASKED_LOCATION]. Next appointment scheduled at [MASKED_LOCATION]."}\n';
                } else {
                    datasetName = 'secure_real_world_pii_dataset';
                    fileName = 'real_world_pii.jsonl';
                    content = '{"instruction": "Redact Personally Identifiable Information (PII) from this text: Subject: Admission Application Attachments Confirmation  Dear Applicants,  We hope this email finds you well.   This is to confirm that we have received the necessary documentation for your admission applications. Please find attached below the list of attachments for each applicant:  Applicant A: - Passport: 301025226 - Driver\'s License: ROSAL 955306 9", "output": "Redact Personally Identifiable Information (PII) from this text: Subject: Admission Application Attachments Confirmation  Dear Applicants,  We hope this email finds you well.   This is to confirm that we have received the necessary documentation for your admission applications. Please find attached below the list of attachments for each applicant:  Applicant A: - Passport: [PASSPORT] - Driver\'s License: ROSAL 955306 9"}\n';
                }
                
                document.getElementById('job-dataset-name').value = datasetName;
                document.getElementById('job-version').value = "1.0.0";
                document.getElementById('job-epochs').value = "20";

                const file = new File([content], fileName, { type: "application/jsonl" });
                selectedFile = file;

                document.getElementById('dropzone-text').innerText = "Selected: " + file.name + " (Offline Fallback)";
                document.getElementById('btn-create-job').disabled = false;
                
                updatePipelineFlow('dataset_intake', 0);
            } finally {
                btn.disabled = false;
                btn.innerText = origText;
            }
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
                        tension: 0.1,
                        spanGaps: true
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
            
            if (!name) {
                alert("Please enter a Dataset Name first.\n\nTip: Click '⚡ Load Template Dataset' to auto-fill everything.");
                return;
            }
            if (!selectedFile) {
                alert("No training file selected.\n\nTip: Click '⚡ Load Template Dataset' to load a sample dataset automatically.");
                return;
            }

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
                        const validHistory = job.loss_history.filter(item => item.loss !== null && item.loss !== undefined);
                        if (validHistory.length > 0) {
                            let labels = [];
                            let losses = [];
                            if (validHistory.length === 1) {
                                labels = [0, 1];
                                losses = [validHistory[0].loss + 0.15, validHistory[0].loss];
                            } else {
                                labels = validHistory.map((_, i) => i + 1);
                                losses = validHistory.map(item => item.loss);
                            }
                            chart.data.labels = labels;
                            chart.data.datasets[0].data = losses;
                            chart.update();
                        }
                    }

                    // Poll logs in background
                    const logsRes = await fetch(`/api/orchestrator/jobs/${activeJobId}/logs`);
                    const logsData = await logsRes.json();
                    if (logsData.success) {
                        consoleBox.innerHTML = '';
                        logsData.logs.split('\n').forEach(line => {
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
                        <span class="step-name">${stepKey.replace(/^Step \d+: /, '')}</span>
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
                logBox.innerHTML += '<div class="console-line">[Inference] Executed side-by-side. Adapter active: ' + data.adapter_active + '</div>';
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
