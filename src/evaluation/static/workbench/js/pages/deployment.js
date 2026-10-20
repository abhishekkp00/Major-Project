import { h, icon, clear, replace } from "../core/dom.js";
import { endpoints, userMessage, isAbort } from "../core/api.js";
import { pageHead, card, kpi, badge, button, emptyState, errorState, skeletonKpis, skeletonCards, skeletonLines, toast } from "../core/ui.js";

const DEFAULT_STEPS = [
  "Step 1: Package Completeness",
  "Step 2: Integrity Verification",
  "Step 3: Signature Verification",
  "Step 4: Device Authorization",
  "Step 5: Key Derivation",
  "Step 6: Decryption & Extraction",
  "Step 7: PEFT Model Loading",
  "Step 8: Inference Validation"
];

export default {
  async mount(root, { navigate, signal }) {
    const header = pageHead({
      title: "Deployment Verification & Model Inference",
      description: "Evidence-driven client gate verification: Package signature, integrity digest, device authorization, and replay-state checks before in-memory adapter mounting and live inference.",
      actions: [
        button({ label: "Packaging Inspector", iconName: "packaging", onClick: () => navigate("/packaging") }),
        button({ label: "Evaluation Benchmarks", iconName: "evaluation", onClick: () => navigate("/evaluation") })
      ]
    });

    const terminalBannerContainer = h("div", null);
    const kpiContainer = h("div", null, skeletonKpis(4));
    const independentChecksContainer = h("div", null, skeletonCards(4));
    const pipelineStepsContainer = h("div", null, skeletonCards(4));
    const deviceIdentityContainer = h("div", null, skeletonLines(4));
    const inferenceContainer = h("div", null);

    const scenarioSelect = h("select", { class: "select mono", id: "verify-scenario-select", style: "min-width: 280px;" },
      h("option", { value: "default" }, "Standard Authorized Deployment"),
      h("option", { value: "tampered_package" }, "Simulate Tampered Package Attack"),
      h("option", { value: "invalid_signature" }, "Simulate Invalid Signature Attack"),
      h("option", { value: "wrong_device" }, "Simulate Unauthorized Device Attack"),
      h("option", { value: "replay" }, "Simulate Replay Attack (Expired Nonce)")
    );

    let isVerifying = false;

    const verifyBtn = button({
      label: "Execute Verification Pipeline",
      iconName: "play",
      kind: "primary",
      onClick: async () => {
        if (isVerifying) return;
        isVerifying = true;
        verifyBtn.disabled = true;
        verifyBtn.innerHTML = `<span class="busy-spinner" style="width: 14px; height: 14px; margin-right: 6px;"></span> Verifying Gates...`;

        try {
          const scenario = scenarioSelect.value;
          const res = await endpoints.phase4Verify({ scenario }, { signal });

          if (res.success) {
            toast({
              title: "Deployment Accepted",
              message: "All 4 independent security checks and 8 pipeline gates passed! Adapter mounted in PEFT memory.",
              kind: "success"
            });
          } else {
            toast({
              title: "Deployment Rejected",
              message: res.error || "Fail-closed gate triggered. Adapter mounting aborted.",
              kind: "error"
            });
          }

          renderState(res);
        } catch (err) {
          if (!signal.aborted) {
            toast({ title: "Verification Error", message: userMessage(err), kind: "error" });
          }
        } finally {
          isVerifying = false;
          verifyBtn.disabled = false;
          verifyBtn.innerHTML = `${icon("play").outerHTML} Execute Verification Pipeline`;
        }
      }
    });

    const verificationCard = card({
      title: "Independent Security Verification Gates",
      sub: "Four independent fail-closed checks: Signature verification, integrity digest, device authorization, and replay-state.",
      actions: [
        scenarioSelect,
        verifyBtn
      ],
      body: independentChecksContainer
    });

    const pipelineCard = card({
      title: "Detailed 8-Stage Execution Pipeline",
      sub: "Full client-side fail-closed execution sequence required before adapter weights are unlocked and mapped into model memory",
      body: pipelineStepsContainer
    });

    const deviceIdentityCard = card({
      title: "Device-Bound Encryption & Key Derivation",
      sub: "Hardware-bound cryptographic parameters and software-derived attestation context (RFC 5869 & NIST SP 800-38D)",
      body: deviceIdentityContainer
    });

    const inferenceCard = card({
      title: "Authorized Adapter Loading & Interactive Inference",
      sub: "Execute side-by-side inference comparisons against the loaded PEFT model only when authorization is confirmed by backend policy",
      body: inferenceContainer
    });

    const layout = h("div", { class: "stack", style: "gap: 20px;" },
      header,
      terminalBannerContainer,
      kpiContainer,
      verificationCard,
      deviceIdentityCard,
      inferenceCard,
      pipelineCard
    );

    replace(root, layout);

    function renderTerminalBanner(terminalState, rejectionReason, isLoaded) {
      if (terminalState === "ACCEPTED") {
        replace(terminalBannerContainer, h("div", { class: "notice notice-success", style: "padding: 14px 18px; border-left: 4px solid var(--accent-teal);" },
          h("div", { class: "row", style: "gap: 10px; align-items: center;" },
            icon("check", { cls: "text-success" }),
            h("strong", { style: "font-size: 0.96rem;" }, "DEPLOYMENT ACCEPTED: All Cryptographic Checks Passed")
          ),
          h("p", { class: "small", style: "margin: 6px 0 0 28px; line-height: 1.45;" },
            "Package signature, integrity digest, device binding, and anti-replay state have been cryptographically verified. The adapter is authorized and successfully mounted into runtime memory."
          )
        ));
      } else if (terminalState === "REJECTED") {
        replace(terminalBannerContainer, h("div", { class: "notice notice-error", style: "padding: 14px 18px; border-left: 4px solid var(--error);" },
          h("div", { class: "row", style: "gap: 10px; align-items: center;" },
            icon("alert", { cls: "text-danger" }),
            h("strong", { style: "font-size: 0.96rem;" }, "DEPLOYMENT REJECTED: Fail-Closed Security Policy Triggered")
          ),
          h("p", { class: "small", style: "margin: 6px 0 0 28px; line-height: 1.45;" },
            rejectionReason || "One or more required cryptographic verification gates failed. Adapter mounting has been strictly aborted to prevent unauthorized execution."
          )
        ));
      } else {
        replace(terminalBannerContainer, h("div", { class: "notice notice-info", style: "padding: 14px 18px; border-left: 4px solid var(--accent-blue);" },
          h("div", { class: "row", style: "gap: 10px; align-items: center;" },
            icon("info", { cls: "text-blue" }),
            h("strong", { style: "font-size: 0.96rem;" }, "DEPLOYMENT UNVERIFIED: Verification Execution Required")
          ),
          h("p", { class: "small", style: "margin: 6px 0 0 28px; line-height: 1.45;" },
            "Adapter weights remain in encrypted ciphertext format. Execute the verification pipeline to validate package authenticity, device identity, and freshness before in-memory mounting."
          )
        ));
      }
    }

    function renderIndependentChecks(checksMap = {}) {
      const checksList = [
        {
          key: "signature",
          num: "01",
          name: "Package Signature Verification",
          subtitle: "RSA-2048-PSS signature verification over canonical manifest digest",
          data: checksMap.signature || { status: "PENDING", name: "Package Signature Verification", evidence: "Awaiting execution." }
        },
        {
          key: "integrity",
          num: "02",
          name: "Integrity Digest Verification",
          subtitle: "SHA-256 ciphertext checksum comparison against baseline hash",
          data: checksMap.integrity || { status: "PENDING", name: "Integrity Digest Verification", evidence: "Awaiting execution." }
        },
        {
          key: "device_auth",
          num: "03",
          name: "Device Authorization",
          subtitle: "Hardware telemetry comparison against package-bound reference",
          data: checksMap.device_auth || { status: "PENDING", name: "Device Authorization", evidence: "Awaiting execution." }
        },
        {
          key: "replay_state",
          num: "04",
          name: "Replay-State Verification",
          subtitle: "Anti-replay sequence monotonicity and timestamp freshness check",
          data: checksMap.replay_state || { status: "PENDING", name: "Replay-State Verification", evidence: "Awaiting execution." }
        }
      ];

      const cards = checksList.map((item, idx) => {
        const check = item.data;
        const status = check.status || "PENDING";
        const isPassed = status === "PASSED";
        const isFailed = status === "FAILED";
        const isNotEval = status === "NOT_EVALUATED";

        const borderCol = isPassed ? "var(--accent-teal)" : (isFailed ? "var(--error)" : "var(--border)");
        const bgCol = isPassed ? "var(--bg-hover)" : (isFailed ? "rgba(239, 68, 68, 0.07)" : "var(--bg-surface)");

        // Expandable evidence drawer
        const evidenceDetails = h("details", { style: "margin-top: 10px; font-size: 0.82rem;" },
          h("summary", { style: "cursor: pointer; font-weight: 600; color: var(--primary);" }, "View Evidence & Audit Trail"),
          h("div", { style: "background: var(--bg-surface); padding: 10px 12px; border-radius: var(--radius-sm); border: 1px solid var(--border); margin-top: 6px;" },
            h("div", { class: "stack", style: "gap: 6px;" },
              check.algorithm ? h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Algorithm / Standard:"),
                h("span", { class: "mono small" }, check.algorithm)
              ) : null,
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Measured Evidence:"),
                h("span", { class: isPassed ? "text-success" : (isFailed ? "text-danger" : "muted") }, check.evidence || "No evidence recorded.")
              ),
              check.error ? h("div", { class: "notice notice-error", style: "padding: 6px 10px; margin-top: 4px; font-size: 0.8rem;" },
                h("strong", null, "Rejection Rationale: "),
                check.error
              ) : null
            )
          )
        );

        return h("div", {
          class: "card",
          style: `padding: 14px 16px; background: ${bgCol}; border-left: 4px solid ${borderCol}; animation: fadeIn 0.25s ease-out; animation-delay: ${idx * 60}ms;`
        },
          h("div", { class: "row-between", style: "margin-bottom: 6px;" },
            h("div", { class: "row", style: "gap: 8px; align-items: center;" },
              h("span", { class: "mono small text-teal", style: "font-weight: 700;" }, item.num),
              h("strong", { style: "font-size: 0.92rem;" }, item.name)
            ),
            badge(status)
          ),
          h("div", { class: "small muted", style: "margin-bottom: 4px;" }, item.subtitle),
          evidenceDetails
        );
      });

      const grid = h("div", { class: "grid grid-2", style: "gap: 14px;" }, ...cards);
      replace(independentChecksContainer, grid);
    }

    function renderDeviceIdentitySection(devId = {}, cryptoMeta = {}) {
      const isAuth = devId.status === "AUTHORIZED";

      const content = h("div", { class: "grid grid-2", style: "gap: 16px;" },
        // Panel A: Device Identity
        h("div", { class: "card", style: "background: var(--bg-hover); padding: 16px;" },
          h("div", { class: "row-between", style: "margin-bottom: 12px;" },
            h("h4", { style: "margin: 0; font-size: 0.92rem;" }, "Host Device Telemetry & Attestation"),
            badge(devId.status || "UNEVALUATED")
          ),
          h("div", { class: "stack", style: "gap: 8px; font-size: 0.85rem;" },
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "Device Fingerprint (Masked):"),
              h("span", { class: "mono" }, devId.fingerprint_prefix || "Configured")
            ),
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "Authorization Status:"),
              h("span", { class: isAuth ? "text-success font-bold" : "muted" }, devId.status || "UNEVALUATED")
            ),
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "Attestation Mechanism:"),
              h("span", { class: "small mono" }, "Software-Derived OS Telemetry")
            )
          ),
          h("div", { class: "notice notice-warning", style: "padding: 10px 12px; margin-top: 14px; font-size: 0.82rem;" },
            h("strong", null, "Attestation Scope: "),
            devId.disclaimer || "Software-derived device binding evaluates host OS hardware properties. It does not provide hardware-isolated key protection (e.g., TPM 2.0 or Secure Enclave) and cannot defend against a root-level kernel adversary."
          )
        ),

        // Panel B: Cryptographic Parameters
        h("div", { class: "card", style: "background: var(--bg-hover); padding: 16px;" },
          h("h4", { style: "margin: 0 0 12px 0; font-size: 0.92rem;" }, "Key Derivation & Envelope Parameters"),
          h("div", { class: "stack", style: "gap: 8px; font-size: 0.85rem;" },
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "Key Derivation Function:"),
              h("span", { class: "mono" }, cryptoMeta.kdf_scheme || "HKDF-SHA256 (RFC 5869)")
            ),
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "KDF Scheme Version:"),
              h("span", { class: "mono" }, cryptoMeta.kdf_version || "hkdf-sha256-v2")
            ),
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "Symmetric Cipher:"),
              h("span", { class: "mono" }, cryptoMeta.cipher || "AES-256-GCM")
            ),
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "IV / Nonce & Tag Length:"),
              h("span", { class: "mono" }, `${cryptoMeta.iv_length_bytes || 12}B IV / ${cryptoMeta.tag_length_bytes || 16}B GMAC Tag`)
            ),
            h("div", { class: "row-between" },
              h("span", { class: "muted" }, "Device Salt Binding:"),
              h("span", { class: "mono" }, cryptoMeta.salt_masked || "Masked (Active)")
            )
          ),
          h("div", { class: "notice notice-info", style: "padding: 10px 12px; margin-top: 14px; font-size: 0.82rem;" },
            h("strong", null, "Zero-Knowledge Principle: "),
            cryptoMeta.zeroization_guarantee || "Decryption keys reside exclusively in volatile process memory and are zeroized upon context exit. Private signing keys are never exposed."
          )
        )
      );

      replace(deviceIdentityContainer, content);
    }

    function renderPipelineSteps(stepsMap = {}) {
      const stepItems = DEFAULT_STEPS.map((stepName, idx) => {
        const rawStatus = stepsMap[stepName] || "PENDING";
        const isPassed = rawStatus === "PASSED";
        const isFailed = rawStatus === "FAILED";

        return h("div", {
          class: "card",
          style: `padding: 10px 12px; background: ${isPassed ? "var(--bg-hover)" : (isFailed ? "rgba(239, 68, 68, 0.08)" : "var(--bg-surface)")}; border-left: 3px solid ${isPassed ? "var(--accent-teal)" : (isFailed ? "var(--error)" : "var(--border)")};`
        },
          h("div", { class: "row-between" },
            h("div", { class: "row", style: "gap: 8px; align-items: center;" },
              h("span", { class: "mono small text-teal", style: "font-weight: 700;" }, `0${idx + 1}`),
              h("span", { style: "font-size: 0.88rem; font-weight: 500;" }, stepName.replace(/^Step \d+: /, ""))
            ),
            badge(rawStatus)
          )
        );
      });

      const grid = h("div", { class: "grid grid-2", style: "gap: 10px;" }, ...stepItems);
      replace(pipelineStepsContainer, grid);
    }

    function renderInferenceSection(isVerified, adapterProv = {}) {
      const promptInput = h("textarea", {
        class: "input",
        style: "width: 100%; min-height: 80px; font-size: 0.9rem; resize: vertical;",
        placeholder: "Enter an evaluation prompt with clinical notes or sensitive entities..."
      }, "Patient Jane Doe, SSN 000-12-3456, was admitted with persistent migraine and acute hypertension.");

      const outputGrid = h("div", { class: "grid grid-2", style: "gap: 16px; margin-top: 16px;" },
        h("div", { class: "card", style: "padding: 16px; background: var(--bg-hover);" },
          h("div", { class: "row-between", style: "margin-bottom: 8px;" },
            h("strong", null, "Base Model (Unscreened Baseline)"),
            badge("RAW WEIGHTS")
          ),
          h("div", { id: "base-output-text", class: "small", style: "line-height: 1.5; color: var(--text-muted);" },
            isVerified ? "Awaiting prompt submission..." : "Model execution locked pending verification."
          )
        ),
        h("div", { class: "card", style: "padding: 16px; background: var(--bg-hover);" },
          h("div", { class: "row-between", style: "margin-bottom: 8px;" },
            h("strong", null, "SecureLoRA Adapter (Screened & Privacy-Preserved)"),
            badge(isVerified ? "ADAPTER ACTIVE" : "WEIGHTS LOCKED")
          ),
          h("div", { id: "lora-output-text", class: "small", style: "line-height: 1.5; color: var(--text-muted);" },
            isVerified ? "Awaiting prompt submission..." : "Model execution locked pending verification."
          )
        )
      );

      const runInferenceBtn = button({
        label: "Execute Side-by-Side Inference",
        iconName: "play",
        kind: "primary",
        disabled: !isVerified,
        onClick: async () => {
          const prompt = promptInput.value.trim();
          if (!prompt) return;

          const baseOut = outputGrid.querySelector("#base-output-text");
          const loraOut = outputGrid.querySelector("#lora-output-text");
          baseOut.textContent = "Generating baseline output from foundation model...";
          loraOut.textContent = "Executing screened LoRA adapter inference...";

          try {
            const res = await endpoints.phase4Generate({ prompt }, { signal });
            if (res.error) {
              toast({ title: "Inference Error", message: res.error, kind: "error" });
              baseOut.textContent = `[ERROR: ${res.error}]`;
              loraOut.textContent = `[ERROR: ${res.error}]`;
              return;
            }

            baseOut.textContent = res.base_output || res.base_response || "No response produced.";
            loraOut.textContent = res.securelora_output || res.lora_response || "No response produced.";

            if (res.base_pii_count > 0 || res.securelora_pii_count > 0) {
              toast({
                title: "PII Detection Comparison",
                message: `Base PII detected: ${res.base_pii_count} · SecureLoRA PII: ${res.securelora_pii_count}`,
                kind: res.securelora_pii_count === 0 ? "success" : "warning"
              });
            }
          } catch (err) {
            if (!signal.aborted) {
              const msg = userMessage(err);
              toast({ title: "Inference Failed", message: msg, kind: "error" });
              baseOut.textContent = `[INFERENCE FAILED: ${msg}]`;
              loraOut.textContent = `[INFERENCE FAILED: ${msg}]`;
            }
          }
        }
      });

      const provenanceInfo = h("div", { class: "row-between", style: "background: var(--bg-hover); padding: 10px 14px; border-radius: var(--radius-sm); font-size: 0.85rem;" },
        h("div", { class: "row", style: "gap: 8px;" },
          h("span", { class: "muted" }, "Mounted Model:"),
          h("span", { class: "mono" }, adapterProv.base_model_name || "JackFram/llama-68m")
        ),
        h("div", { class: "row", style: "gap: 8px;" },
          h("span", { class: "muted" }, "Adapter ID:"),
          h("span", { class: "mono" }, adapterProv.adapter_id || "secure_lora_adapter")
        ),
        h("div", { class: "row", style: "gap: 8px;" },
          h("span", { class: "muted" }, "Status:"),
          badge(isVerified ? "MOUNTED" : "UNMOUNTED")
        )
      );

      const plaintextNotice = h("div", { class: "notice notice-warning", style: "padding: 10px 14px; font-size: 0.82rem;" },
        h("strong", null, "Transient Plaintext Storage Notice: "),
        adapterProv.transient_plaintext_warning || "During PEFT adapter initialization, decrypted weights are briefly buffered in restricted-permission ephemeral memory before being mapped into model runtime. In production environments, mount scratch storage on RAM-backed tmpfs (mode 0700)."
      );

      const section = h("div", { class: "stack", style: "gap: 14px;" },
        !isVerified ? h("div", { class: "notice notice-error" },
          "Inference execution is blocked by fail-closed policy. The 4 independent verification checks must pass before the model can be unlocked."
        ) : null,
        provenanceInfo,
        promptInput,
        h("div", { class: "row-between", style: "align-items: center;" },
          h("span", { class: "small muted" }, "Inference calls the live Phase 4 generation endpoint directly."),
          runInferenceBtn
        ),
        outputGrid,
        plaintextNotice
      );

      replace(inferenceContainer, section);
    }

    function renderState(data = {}) {
      const isLoaded = Boolean(data.loaded);
      const terminalState = data.terminal_state || (data.success ? "ACCEPTED" : (data.error ? "REJECTED" : "PENDING"));
      const rejectionReason = data.rejection_reason || data.error;

      // Top KPIs
      replace(kpiContainer, h("div", { class: "kpis" },
        kpi({
          label: "Deployment State",
          value: terminalState,
          text: true,
          foot: terminalState === "ACCEPTED" ? "All verification gates passed" : (terminalState === "REJECTED" ? "Execution blocked" : "Awaiting verification")
        }),
        kpi({
          label: "Device Authorization",
          value: data.device_identity?.status || "UNEVALUATED",
          text: true,
          foot: data.device_identity?.fingerprint_prefix ? `Host ${data.device_identity.fingerprint_prefix}` : "Host fingerprint"
        }),
        kpi({
          label: "Adapter Runtime State",
          value: isLoaded ? "MOUNTED IN MEMORY" : "WEIGHTS LOCKED",
          text: true,
          foot: isLoaded ? "PEFT adapter active" : "Decryption blocked"
        }),
        kpi({
          label: "Foundation Model",
          value: data.base_model_name || "JackFram/llama-68m",
          text: true,
          foot: "Target architecture"
        })
      ));

      renderTerminalBanner(terminalState, rejectionReason, isLoaded);
      renderIndependentChecks(data.checks || {});
      renderDeviceIdentitySection(data.device_identity || {}, data.crypto_metadata || {});
      renderInferenceSection(isLoaded && terminalState === "ACCEPTED", data.adapter_provenance || {});
      renderPipelineSteps(data.steps || {});
    }

    async function loadStatus() {
      try {
        const res = await endpoints.phase4Status({ signal });
        if (signal.aborted) return;
        renderState(res);
      } catch (err) {
        if (isAbort(err)) return;
        replace(kpiContainer, errorState(err, loadStatus));
      }
    }

    loadStatus();
    return () => {};
  }
};
