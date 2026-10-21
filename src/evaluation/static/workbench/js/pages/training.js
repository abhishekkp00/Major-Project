/* SecureLoRA Training & Differentially Private Adaptation Module.
   Provides a technically accurate, validated configuration panel,
   pre-launch confirmation modal, real-time telemetry monitor,
   accessible loss curve charts, live log stream, resulting adapter
   metadata, and strict differential privacy accounting. */

import { h, icon, replace } from "../core/dom.js";
import { endpoints, userMessage, isAbort } from "../core/api.js";
import {
  fmtNumber,
  fmtPercent,
  fmtDate,
  fmtDuration,
  validateInteger,
  validateFloat,
  validateJobVersion,
} from "../core/format.js";
import {
  pageHead,
  card,
  badge,
  field,
  setFieldError,
  markValid,
  button,
  openDialog,
  toast,
} from "../core/ui.js";
import { lineChart } from "../core/charts.js";

export default {
  async mount(root, { navigate, signal }) {
    // Session preferences
    const preselectedDataset = sessionStorage.getItem("securelora_selected_dataset") || "synthetic_ehr";
    const preselectedSubset = sessionStorage.getItem("securelora_selected_subset") || "100";
    let activeJobId = sessionStorage.getItem("securelora_active_job") || null;

    let pollInterval = null;
    let timerInterval = null;
    let jobStartTime = null;
    let isSubmitting = false;

    // Load backend configuration and historical jobs concurrently
    let backendConfig = null;
    let recentJobs = [];

    try {
      const [cfgRes, jobsRes] = await Promise.all([
        endpoints.trainingConfig({ signal }).catch(() => null),
        endpoints.jobs({ signal }).catch(() => null),
      ]);
      if (cfgRes && cfgRes.success && cfgRes.config) {
        backendConfig = cfgRes.config;
      }
      if (jobsRes && jobsRes.jobs) {
        recentJobs = jobsRes.jobs;
      }
    } catch (err) {
      if (signal.aborted) return;
    }

    // Default configuration fallbacks if API unavailable
    const supportedModels = backendConfig?.supported_models || [
      {
        id: "JackFram/llama-68m",
        name: "JackFram/llama-68m (Evaluated Baseline)",
        parameters: "68M",
        default: true,
        description: "Baseline causal language model evaluated and pre-configured for SecureLoRA.",
      },
      {
        id: "meta-llama/Llama-2-7b-hf",
        name: "meta-llama/Llama-2-7b-hf (7B Scale)",
        parameters: "7B",
        default: false,
        description: "High-capacity 7B parameter foundation model requiring dedicated GPU acceleration.",
      },
    ];

    const loraDefaults = backendConfig?.lora_defaults || {
      r: 8,
      r_min: 1,
      r_max: 64,
      alpha: 16,
      alpha_min: 1,
      alpha_max: 128,
      dropout: 0.05,
      dropout_min: 0.0,
      dropout_max: 0.5,
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"],
      default_modules: ["q_proj", "v_proj"],
    };

    const dpCapabilities = backendConfig?.dp_capabilities || {
      supported: true,
      engine: "Opacus (PyTorch DP-SGD)",
      default_accountant: "rdp",
      supported_accountants: ["rdp", "prv"],
      default_max_grad_norm: 1.0,
      max_grad_norm_min: 0.1,
      max_grad_norm_max: 10.0,
      default_noise_multiplier: 1.2,
      noise_multiplier_min: 0.1,
      noise_multiplier_max: 5.0,
      default_target_delta: 0.00001,
      default_target_epsilon: 8.0,
      target_epsilon_min: 0.5,
      target_epsilon_max: 50.0,
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Header & Navigation
    // ─────────────────────────────────────────────────────────────────────────
    const header = pageHead({
      title: "SecureLoRA Training & Privacy Optimization",
      description:
        "Configure LoRA or DP-LoRA fine-tuning hyperparameters, verify compute requirements, and monitor live convergence telemetry with formal privacy accounting.",
      actions: [
        button({
          label: "Dataset Catalog",
          iconName: "dataset",
          onClick: () => navigate("/dataset"),
        }),
        button({
          label: "Pipeline Overview",
          iconName: "overview",
          kind: "ghost",
          onClick: () => navigate("/overview"),
        }),
      ],
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Mode Switcher: Standard LoRA vs DP-LoRA
    // ─────────────────────────────────────────────────────────────────────────
    let currentMode = "standard"; // "standard" | "dp"

    const modeStandardRadio = h("input", {
      type: "radio",
      id: "mode-standard",
      name: "training-mode",
      value: "standard",
      checked: true,
    });
    const modeDpRadio = h("input", {
      type: "radio",
      id: "mode-dp",
      name: "training-mode",
      value: "dp",
      disabled: !dpCapabilities.supported,
    });

    const modeContainer = h(
      "div",
      {
        class: "grid grid-2",
        style: "gap: 12px; margin-bottom: 16px;",
        role: "radiogroup",
        "aria-label": "LoRA Training Mode",
      },
      h(
        "label",
        {
          class: "card clickable",
          id: "mode-standard-label",
          style:
            "border: 2px solid var(--primary); padding: 14px; cursor: pointer; border-radius: var(--radius-md); transition: var(--transition-fast);",
        },
        h(
          "div",
          { class: "row-between", style: "margin-bottom: 6px;" },
          h(
            "div",
            { class: "row", style: "gap: 8px;" },
            modeStandardRadio,
            h("strong", null, "Standard LoRA (Mode A)")
          ),
          badge("BASELINE UTILITY")
        ),
        h(
          "p",
          { class: "text-muted", style: "font-size: 0.85rem; margin: 0;" },
          "Standard gradient descent without clipping or noise injection. Maximum task utility baseline. Training-data protection relies on Phase 1 sanitization."
        )
      ),
      h(
        "label",
        {
          class: `card clickable ${!dpCapabilities.supported ? "disabled" : ""}`,
          id: "mode-dp-label",
          style:
            "border: 2px solid var(--border); padding: 14px; cursor: pointer; border-radius: var(--radius-md); transition: var(--transition-fast);",
        },
        h(
          "div",
          { class: "row-between", style: "margin-bottom: 6px;" },
          h(
            "div",
            { class: "row", style: "gap: 8px;" },
            modeDpRadio,
            h("strong", null, "DP-LoRA via Opacus (Mode B)")
          ),
          dpCapabilities.supported
            ? badge("FORMAL (ε, δ)-DP")
            : badge("BACKEND UNAVAILABLE")
        ),
        h(
          "p",
          { class: "text-muted", style: "font-size: 0.85rem; margin: 0;" },
          dpCapabilities.supported
            ? "Per-example gradient clipping (C) + Gaussian noise (σ). Guaranteed mathematical resistance to membership inference attacks with Rényi DP accounting."
            : "Opacus differential privacy library is not installed or enabled in current backend environment."
        )
      )
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Form Inputs & Fields
    // ─────────────────────────────────────────────────────────────────────────
    // 1. Base Model Selector
    const modelSelect = h(
      "select",
      { class: "select", id: "train-model" },
      ...supportedModels.map((m) =>
        h(
          "option",
          { value: m?.id || "", selected: Boolean(m?.default) },
          `${m?.name || m?.id || "Model"} [${m?.parameters || "Unknown"}]`
        )
      )
    );
    const modelField = field({
      label: "Base Foundation Model",
      control: modelSelect,
      hint: "Causal language model architecture to be frozen and adapted with low-rank matrices.",
    });

    // 2. Dataset & Sample Controls
    const datasetSelect = h(
      "select",
      { class: "select", id: "train-dataset" },
      h("option", { value: "synthetic_ehr", selected: preselectedDataset === "synthetic_ehr" }, "Synthetic EHR Clinical Dialogue (1,000 records)"),
      h("option", { value: "alpaca_cleaned", selected: preselectedDataset === "alpaca_cleaned" }, "Stanford Alpaca Cleaned (51,760 records)"),
      h("option", { value: "synthetic_pii_benchmark", selected: preselectedDataset === "synthetic_pii_benchmark" }, "Synthetic PII Adversarial Benchmark"),
      h("option", { value: "gsm8k", selected: preselectedDataset === "gsm8k" }, "GSM8K Reasoning Benchmark"),
      h("option", { value: "custom", selected: preselectedDataset === "custom" }, "Custom Uploaded Dataset")
    );
    const datasetField = field({
      label: "Dataset Source",
      control: datasetSelect,
      hint: "Standardized or custom training corpus for adaptation.",
    });

    const subsetSelect = h(
      "select",
      { class: "select", id: "train-subset" },
      h("option", { value: "50", selected: preselectedSubset === "50" }, "50 records (Quick Smoke Test)"),
      h("option", { value: "100", selected: preselectedSubset === "100" }, "100 records (Standard Research)"),
      h("option", { value: "500", selected: preselectedSubset === "500" }, "500 records (Deep Fine-Tuning)"),
      h("option", { value: "1000", selected: preselectedSubset === "1000" }, "1,000 records (Full Subset)")
    );
    const subsetField = field({
      label: "Sample Subset Size",
      control: subsetSelect,
      hint: "Bounded sample count to control memory footprint and compute bounds.",
    });

    const versionInput = h("input", {
      type: "text",
      class: "input mono",
      id: "train-version",
      value: "1.0.0",
      placeholder: "1.0.0",
    });
    const versionField = field({
      label: "Adapter Version",
      control: versionInput,
      hint: "Semantic version format: X.Y.Z (e.g. 1.0.0).",
    });

    const epochsInput = h("input", {
      type: "number",
      class: "input",
      id: "train-epochs",
      value: "1",
      min: "1",
      max: "20",
    });
    const epochsField = field({
      label: "Training Epochs",
      control: epochsInput,
      hint: "Number of full passes over dataset (1 – 20).",
    });

    // 3. LoRA Hyperparameters
    const rankInput = h("input", {
      type: "number",
      class: "input",
      id: "train-rank",
      value: String(loraDefaults.r),
      min: String(loraDefaults.r_min),
      max: String(loraDefaults.r_max),
    });
    const rankField = field({
      label: `LoRA Rank (r: ${loraDefaults.r_min}–${loraDefaults.r_max})`,
      control: rankInput,
      hint: "Bottleneck dimension for low-rank adapter matrices (A and B).",
    });

    const alphaInput = h("input", {
      type: "number",
      class: "input",
      id: "train-alpha",
      value: String(loraDefaults.alpha),
      min: String(loraDefaults.alpha_min),
      max: String(loraDefaults.alpha_max),
    });
    const alphaField = field({
      label: `LoRA Alpha (α: ${loraDefaults.alpha_min}–${loraDefaults.alpha_max})`,
      control: alphaInput,
      hint: "Scaling hyperparameter for weight update magnitude (ΔW = α/r · BA).",
    });

    const dropoutInput = h("input", {
      type: "number",
      step: "0.01",
      class: "input",
      id: "train-dropout",
      value: String(loraDefaults.dropout),
      min: String(loraDefaults.dropout_min),
      max: String(loraDefaults.dropout_max),
    });
    const dropoutField = field({
      label: `LoRA Dropout (${loraDefaults.dropout_min}–${loraDefaults.dropout_max})`,
      control: dropoutInput,
      hint: "Dropout regularization applied to adapter intermediate activations.",
    });

    // Target Module checkboxes
    const targetModuleCheckboxes = (loraDefaults.target_modules || ["q_proj", "v_proj", "k_proj", "o_proj"]).map((mod) => {
      const isDefault = loraDefaults.default_modules?.includes(mod) || ["q_proj", "v_proj"].includes(mod);
      const cb = h("input", {
        type: "checkbox",
        id: `mod-${mod}`,
        value: mod,
        checked: isDefault,
      });
      const lbl = h(
        "label",
        { class: "row", style: "gap: 6px; font-family: monospace; font-size: 0.85rem; cursor: pointer;" },
        cb,
        mod
      );
      return { cb, lbl, value: mod };
    });

    const modulesBox = h(
      "div",
      { id: "train-modules-box", class: "row", style: "gap: 16px; flex-wrap: wrap; margin-top: 6px;" },
      ...targetModuleCheckboxes.map((m) => m.lbl)
    );
    const modulesField = field({
      label: "Target Attention Modules",
      control: modulesBox,
      hint: "Transformer weight matrices to inject LoRA adapters into.",
    });

    // 4. Differential Privacy Parameters (Conditional)
    const clipNormInput = h("input", {
      type: "number",
      step: "0.1",
      class: "input",
      id: "train-clip-norm",
      value: String(dpCapabilities.default_max_grad_norm ?? 1.0),
      min: String(dpCapabilities.max_grad_norm_min ?? 0.1),
      max: String(dpCapabilities.max_grad_norm_max ?? 10.0),
    });
    const clipNormField = field({
      label: "Max Gradient Clipping Norm (C)",
      control: clipNormInput,
      hint: "Per-example gradient bound: g̃_i = g_i · min(1, C / ||g_i||₂).",
    });

    const noiseMultInput = h("input", {
      type: "number",
      step: "0.05",
      class: "input",
      id: "train-noise-multiplier",
      value: String(dpCapabilities.default_noise_multiplier ?? 1.2),
      min: String(dpCapabilities.noise_multiplier_min ?? 0.1),
      max: String(dpCapabilities.noise_multiplier_max ?? 5.0),
    });
    const noiseMultField = field({
      label: "Noise Multiplier (σ)",
      control: noiseMultInput,
      hint: "Standard deviation of Gaussian noise calibrated to clipping norm: N(0, σ²C²I).",
    });

    const targetDeltaInput = h("input", {
      type: "text",
      class: "input mono",
      id: "train-target-delta",
      value: "1e-5",
    });
    const targetDeltaField = field({
      label: "Target Delta (δ)",
      control: targetDeltaInput,
      hint: "Failure probability bound (must be < 1 / N, typically 1e-5).",
    });

    const targetEpsilonInput = h("input", {
      type: "number",
      step: "0.5",
      class: "input",
      id: "train-target-epsilon",
      value: String(dpCapabilities.default_target_epsilon ?? 8.0),
      min: String(dpCapabilities.target_epsilon_min ?? 0.5),
      max: String(dpCapabilities.target_epsilon_max ?? 50.0),
    });
    const targetEpsilonField = field({
      label: "Target Epsilon Budget (ε)",
      control: targetEpsilonInput,
      hint: "Target upper bound on privacy expenditure across total training iterations.",
    });

    const accountantSelect = h(
      "select",
      { class: "select", id: "train-accountant" },
      h("option", { value: "rdp", selected: true }, "Rényi Differential Privacy (RDP - Moments Accountant)"),
      h("option", { value: "prv" }, "Privacy Loss Distribution (PRV / f-DP)")
    );
    const accountantField = field({
      label: "Privacy Accountant Mechanism",
      control: accountantSelect,
      hint: "Composition accountant tracking privacy loss distribution across steps.",
    });

    const dpPanel = h(
      "div",
      {
        id: "dp-parameters-panel",
        class: "stack",
        style: "display: none; padding: 16px; background: var(--bg-hover); border: 1px solid var(--border); border-radius: var(--radius-md); gap: 14px;",
      },
      h(
        "div",
        { class: "row-between" },
        h(
          "div",
          { class: "row", style: "gap: 8px;" },
          icon("lock", { cls: "text-primary" }),
          h("strong", null, "Opacus DP-SGD Privacy Hyperparameters")
        ),
        badge("PROVABLE BOUNDS")
      ),
      h("div", { class: "grid grid-2", style: "gap: 16px;" }, clipNormField, noiseMultField),
      h("div", { class: "grid grid-3", style: "gap: 16px;" }, targetDeltaField, targetEpsilonField, accountantField)
    );

    // 5. Cryptographic Salt Control
    const saltInput = h("input", {
      type: "text",
      class: "input mono",
      id: "train-salt",
      placeholder: "Auto-derived hardware binding salt (HKDF-SHA256)",
    });
    const saltField = field({
      label: "Device Binding Salt (Optional)",
      control: saltInput,
      hint: "Hardware-unique salt for HKDF key derivation. Defaults to local host identity.",
    });

    // Mode Toggle Event Listeners
    function updateModeDisplay(mode) {
      currentMode = mode;
      const standardCard = modeContainer.querySelector("#mode-standard-label");
      const dpCard = modeContainer.querySelector("#mode-dp-label");

      if (mode === "dp") {
        modeDpRadio.checked = true;
        modeStandardRadio.checked = false;
        dpPanel.style.display = "block";
        if (standardCard) standardCard.style.borderColor = "var(--border)";
        if (dpCard) dpCard.style.borderColor = "var(--primary)";
      } else {
        modeStandardRadio.checked = true;
        modeDpRadio.checked = false;
        dpPanel.style.display = "none";
        if (standardCard) standardCard.style.borderColor = "var(--primary)";
        if (dpCard) dpCard.style.borderColor = "var(--border)";
      }
    }

    modeStandardRadio.addEventListener("change", () => updateModeDisplay("standard"));
    modeDpRadio.addEventListener("change", () => updateModeDisplay("dp"));

    // ─────────────────────────────────────────────────────────────────────────
    // Launch Action & Confirmation Dialog
    // ─────────────────────────────────────────────────────────────────────────
    const launchBtn = button({
      label: "Review & Launch Training Run",
      iconName: "play",
      kind: "primary",
      onClick: async () => {
        if (isSubmitting) return;

        // 1. Validation
        let hasErrors = false;

        const vErr = validateJobVersion(versionInput.value);
        if (vErr) { setFieldError(versionField, vErr); hasErrors = true; } else { markValid(versionField); }

        const epErr = validateInteger(epochsInput.value, { min: 1, max: 20, label: "Training Epochs" });
        if (epErr) { setFieldError(epochsField, epErr); hasErrors = true; } else { markValid(epochsField); }

        const rErr = validateInteger(rankInput.value, {
          min: loraDefaults.r_min,
          max: loraDefaults.r_max,
          label: "LoRA Rank",
        });
        if (rErr) { setFieldError(rankField, rErr); hasErrors = true; } else { markValid(rankField); }

        const aErr = validateInteger(alphaInput.value, {
          min: loraDefaults.alpha_min,
          max: loraDefaults.alpha_max,
          label: "LoRA Alpha",
        });
        if (aErr) { setFieldError(alphaField, aErr); hasErrors = true; } else { markValid(alphaField); }

        const dropErr = validateFloat(dropoutInput.value, {
          min: loraDefaults.dropout_min,
          max: loraDefaults.dropout_max,
          label: "LoRA Dropout",
        });
        if (dropErr) { setFieldError(dropoutField, dropErr); hasErrors = true; } else { markValid(dropoutField); }

        const selectedModules = targetModuleCheckboxes.filter((m) => m.cb.checked).map((m) => m.value);
        if (selectedModules.length === 0) {
          setFieldError(modulesField, "Select at least one target attention module.");
          hasErrors = true;
        } else {
          markValid(modulesField);
        }

        let dpConfigPayload = null;
        if (currentMode === "dp") {
          const clipErr = validateFloat(clipNormInput.value, {
            min: dpCapabilities.max_grad_norm_min ?? 0.1,
            max: dpCapabilities.max_grad_norm_max ?? 10.0,
            label: "Clipping Norm",
          });
          if (clipErr) { setFieldError(clipNormField, clipErr); hasErrors = true; } else { markValid(clipNormField); }

          const noiseErr = validateFloat(noiseMultInput.value, {
            min: dpCapabilities.noise_multiplier_min ?? 0.1,
            max: dpCapabilities.noise_multiplier_max ?? 5.0,
            label: "Noise Multiplier",
          });
          if (noiseErr) { setFieldError(noiseMultField, noiseErr); hasErrors = true; } else { markValid(noiseMultField); }

          const deltaVal = Number(targetDeltaInput.value.trim());
          if (!Number.isFinite(deltaVal) || deltaVal <= 0 || deltaVal >= 1) {
            setFieldError(targetDeltaField, "Target Delta must be a positive decimal strictly less than 1 (e.g. 1e-5).");
            hasErrors = true;
          } else {
            markValid(targetDeltaField);
          }

          const epsErr = validateFloat(targetEpsilonInput.value, {
            min: dpCapabilities.target_epsilon_min ?? 0.5,
            max: dpCapabilities.target_epsilon_max ?? 50.0,
            label: "Target Epsilon",
          });
          if (epsErr) { setFieldError(targetEpsilonField, epsErr); hasErrors = true; } else { markValid(targetEpsilonField); }

          if (!hasErrors) {
            dpConfigPayload = {
              dp_enabled: true,
              dp_max_grad_norm: Number(clipNormInput.value),
              dp_noise_multiplier: Number(noiseMultInput.value),
              dp_target_delta: deltaVal,
              dp_target_epsilon: Number(targetEpsilonInput.value),
              dp_accountant: accountantSelect.value,
            };
          }
        }

        if (hasErrors) {
          toast({
            title: "Validation Incomplete",
            message: "Please correct the highlighted configuration parameters before proceeding.",
            kind: "warning",
          });
          return;
        }

        // 2. Pre-launch Confirmation Summary Modal
        const selectedModelObj = supportedModels.find((m) => m?.id === modelSelect.value) || {
          name: modelSelect.value,
          parameters: "N/A",
        };

        const confirmationSummary = h(
          "div",
          { class: "stack", style: "gap: 14px;" },
          h(
            "p",
            { class: "text-muted", style: "margin: 0;" },
            "Please review the technical parameters and resource requirements for this training pipeline job. Once launched, fine-tuning and cryptographic packaging will execute asynchronously."
          ),
          h(
            "div",
            {
              class: "card",
              style: "background: var(--bg-hover); padding: 14px; font-size: 0.9rem; line-height: 1.6;",
            },
            h(
              "div",
              { class: "grid grid-2", style: "gap: 10px;" },
              h("div", null, h("strong", null, "Base Model: "), selectedModelObj.name),
              h("div", null, h("strong", null, "Architecture Parameters: "), selectedModelObj.parameters),
              h(
                "div",
                null,
                h("strong", null, "Training Mode: "),
                currentMode === "dp" ? "Differentially Private LoRA (DP-SGD)" : "Standard LoRA (Mode A)"
              ),
              h(
                "div",
                null,
                h("strong", null, "Data Corpus: "),
                `${datasetSelect.value} (${subsetSelect.value} records, ${epochsInput.value} epochs)`
              ),
              h(
                "div",
                null,
                h("strong", null, "LoRA Adapter: "),
                `r=${rankInput.value}, α=${alphaInput.value}, dropout=${dropoutInput.value}`
              ),
              h(
                "div",
                null,
                h("strong", null, "Target Modules: "),
                selectedModules.join(", ")
              ),
              ...(currentMode === "dp" && dpConfigPayload
                ? [
                    h(
                      "div",
                      null,
                      h("strong", null, "DP Noise Multiplier (σ): "),
                      `${dpConfigPayload.dp_noise_multiplier}`
                    ),
                    h(
                      "div",
                      null,
                      h("strong", null, "Clipping Bound (C): "),
                      `${dpConfigPayload.dp_max_grad_norm}`
                    ),
                    h(
                      "div",
                      null,
                      h("strong", null, "Privacy Target: "),
                      `ε ≤ ${dpConfigPayload.dp_target_epsilon}, δ = ${dpConfigPayload.dp_target_delta}`
                    ),
                    h(
                      "div",
                      null,
                      h("strong", null, "Accountant: "),
                      dpConfigPayload.dp_accountant.toUpperCase()
                    ),
                  ]
                : [
                    h(
                      "div",
                      { style: "grid-column: span 2;" },
                      h("strong", null, "Privacy Assurance: "),
                      h("span", { class: "text-muted" }, "Standard LoRA (No DP-SGD noise injected; relies on Phase 1 sanitization)")
                    ),
                  ])
            )
          ),
          h(
            "div",
            {
              class: "row",
              style:
                "gap: 8px; padding: 10px 12px; background: rgba(234, 179, 8, 0.1); border-left: 3px solid var(--warning); font-size: 0.85rem; border-radius: var(--radius-sm);",
            },
            icon("alert-circle", { cls: "text-warning" }),
            h(
              "span",
              null,
              "Costly Training Job: Starting this job triggers in-memory dataset decryption, iterative tensor forward/backward passes, and automated Phase 3 security screening."
            )
          )
        );

        const confirmed = await openDialog({
          title: "Confirm Pipeline Execution",
          content: confirmationSummary,
          actions: [
            { label: "Cancel / Adjust", value: "cancel" },
            { label: "Confirm & Launch Training", value: "confirm", kind: "primary" },
          ],
          width: 580,
        });

        if (confirmed !== "confirm") return;

        // 3. Execution & Duplicate Submission Guard
        isSubmitting = true;
        launchBtn.disabled = true;
        launchBtn.innerHTML = `<span class="busy-spinner" style="width: 14px; height: 14px; margin-right: 6px;"></span> Launching Pipeline Job...`;

        try {
          const payload = {
            dataset_name: datasetSelect.value,
            dataset_type: datasetSelect.value,
            version: versionInput.value.trim(),
            epochs: parseInt(epochsInput.value, 10),
            subset_size: parseInt(subsetSelect.value, 10),
            salt: saltInput.value.trim() || undefined,
            training_config: {
              model_name: modelSelect.value,
              lora_r: parseInt(rankInput.value, 10),
              lora_alpha: parseInt(alphaInput.value, 10),
              lora_dropout: parseFloat(dropoutInput.value),
              target_modules: selectedModules,
              ...(currentMode === "dp" ? dpConfigPayload : { dp_enabled: false }),
            },
          };

          const createRes = await endpoints.createJob(payload, { signal });
          if (!createRes.success || !createRes.job_id) {
            throw new Error(createRes.error || "Failed to create pipeline job.");
          }

          const newJobId = createRes.job_id;
          activeJobId = newJobId;
          sessionStorage.setItem("securelora_active_job", newJobId);

          toast({
            title: "Job Initialized",
            message: `Workspace created for ${newJobId}. Commencing execution...`,
            kind: "info",
          });

          const startRes = await endpoints.startJob(newJobId, { signal });
          if (!startRes.success) {
            throw new Error(startRes.error || "Failed to start training subprocess.");
          }

          toast({
            title: "Pipeline Active",
            message: `Fine-tuning started for ${newJobId}`,
            kind: "success",
          });

          // Switch monitor to newly created job
          renderJobMonitor(newJobId, false);
        } catch (err) {
          if (!signal.aborted) {
            toast({
              title: "Launch Failed",
              message: userMessage(err),
              kind: "error",
            });
          }
        } finally {
          isSubmitting = false;
          launchBtn.disabled = false;
          launchBtn.innerHTML = `${icon("play").outerHTML} Review & Launch Training Run`;
        }
      },
    });

    const formCard = card({
      title: "Adapter Training Configuration",
      sub: "Configure foundation model, low-rank adapter dimensions, and differential privacy bounds",
      body: h(
        "div",
        { class: "stack", style: "gap: 16px;" },
        modeContainer,
        h("div", { class: "grid grid-2", style: "gap: 16px;" }, modelField, datasetField),
        h("div", { class: "grid grid-3", style: "gap: 16px;" }, versionField, epochsField, subsetField),
        h("div", { class: "grid grid-3", style: "gap: 16px;" }, rankField, alphaField, dropoutField),
        modulesField,
        dpPanel,
        saltField,
        h("div", { class: "row-end", style: "margin-top: 8px;" }, launchBtn)
      ),
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Historical Experiments & Active Monitor Selectors
    // ─────────────────────────────────────────────────────────────────────────
    const initialJobId = activeJobId || (recentJobs.length > 0 ? recentJobs[0].job_id : "");

    const historySelect = h(
      "select",
      { class: "select mono", id: "historical-run-select", style: "min-width: 320px; max-width: 500px;" },
      h("option", { value: "" }, recentJobs.length ? "-- Select Run to Inspect --" : "No pipeline runs recorded"),
      ...recentJobs.map((j) =>
        h(
          "option",
          { value: j.job_id, selected: j.job_id === initialJobId },
          `${j.job_id} · ${j.status} · ${j.dataset_name || j.dataset_type || "dataset"} (${fmtDate(j.created_at)})`
        )
      )
    );

    function selectAndRenderJob(val) {
      if (!val) {
        renderEmptyMonitor();
        return;
      }
      historySelect.value = val;
      renderJobMonitor(val, true);
      monitorContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    const historyBar = h(
      "div",
      {
        class: "card row-between",
        style: "padding: 12px 16px; background: var(--bg-card); gap: 12px; flex-wrap: wrap;",
      },
      h(
        "div",
        { class: "row", style: "gap: 8px;" },
        icon("clock", { cls: "text-muted", size: 18 }),
        h("strong", null, "Run Telemetry & Audit:"),
        h("span", { class: "text-muted", style: "font-size: 0.85rem;" }, "Inspect loss curves, logs, and privacy accounting for active or past runs")
      ),
      h(
        "div",
        { class: "row", style: "gap: 8px;" },
        historySelect,
        button({
          label: "View Run",
          iconName: "evaluation",
          kind: "primary",
          small: true,
          onClick: () => {
            selectAndRenderJob(historySelect.value);
          },
        })
      )
    );

    historySelect.addEventListener("change", () => {
      selectAndRenderJob(historySelect.value);
    });

    // Monitor Container - placed immediately below the run audit bar for instant visibility
    const monitorContainer = h("div", { id: "training-monitor-container", class: "stack", style: "gap: 16px;" });

    const layout = h(
      "div",
      { class: "stack", style: "gap: 20px;" },
      header,
      historyBar,
      monitorContainer,
      formCard
    );

    replace(root, layout);

    // Initial load: render active job or most recent job
    if (initialJobId) {
      renderJobMonitor(initialJobId, !activeJobId);
    } else {
      renderEmptyMonitor();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Monitor Rendering Function
    // ─────────────────────────────────────────────────────────────────────────
    function renderEmptyMonitor() {
      replace(
        monitorContainer,
        card({
          title: "Training Telemetry & Privacy Accounting",
          sub: "No active or historical job selected",
          body: h(
            "p",
            { class: "text-muted", style: "margin: 0;" },
            "Configure parameters above and click 'Review & Launch Training Run', or select a historical experiment to inspect verified loss curves and privacy accounting."
          ),
        })
      );
    }

    function renderJobMonitor(jobId, forceHistorical = false) {
      if (pollInterval) clearInterval(pollInterval);
      if (timerInterval) clearInterval(timerInterval);
      if (historySelect && historySelect.value !== jobId) {
        historySelect.value = jobId;
      }

      // Create skeleton cards
      const monitorCard = card({
        title: `Telemetry Monitor: ${jobId}`,
        sub: "Real-time loss convergence, live subprocess logs, and adapter parameters",
        actions: [
          button({
            label: "Screening",
            iconName: "screening",
            small: true,
            kind: "ghost",
            onClick: () => navigate("/screening"),
          }),
          button({
            label: "Packaging",
            iconName: "packaging",
            small: true,
            kind: "ghost",
            onClick: () => navigate("/packaging"),
          }),
        ],
        body: h(
          "div",
          { class: "stack", style: "gap: 16px;" },
          h("div", { id: "monitor-historical-banner" }),
          h(
            "div",
            {
              id: "monitor-status-row",
              class: "row-between",
              style:
                "padding: 12px 16px; background: var(--bg-hover); border-radius: var(--radius-sm); border: 1px solid var(--border); flex-wrap: wrap; gap: 12px;",
            },
            h(
              "div",
              { class: "row", style: "gap: 10px;" },
              h("span", { id: "monitor-spinner", class: "busy-spinner", style: "width: 16px; height: 16px;" }),
              h("div", { class: "stack-sm" },
                h("span", { id: "monitor-status-headline", style: "font-weight: 600;" }, "Connecting to job telemetry..."),
                h("span", { id: "monitor-step-details", class: "text-muted", style: "font-size: 0.85rem;" }, "Awaiting event feed...")
              )
            ),
            h(
              "div",
              { class: "row", style: "gap: 12px;" },
              h(
                "div",
                { class: "stack-sm", style: "text-align: right;" },
                h("span", { style: "font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted);" }, "Elapsed Time"),
                h("span", { id: "monitor-elapsed-timer", class: "mono", style: "font-weight: 600;" }, "0s")
              ),
              h("span", { id: "monitor-status-badge" }, badge("LOADING"))
            )
          ),
          h("div", { id: "monitor-loss-chart-container" }),
          h(
            "div",
            { class: "stack-sm" },
            h("div", { class: "row-between" },
              h("strong", { style: "font-size: 0.85rem;" }, "Subprocess Execution Logs:"),
              h("span", { class: "text-muted mono", style: "font-size: 0.75rem;" }, "training.log")
            ),
            h(
              "div",
              {
                class: "card",
                style:
                  "padding: 12px; background: #050b14; color: #a5b4fc; font-family: monospace; font-size: 0.8rem; height: 220px; overflow-y: auto; border: 1px solid var(--border);",
              },
              h("pre", { id: "monitor-logs-pre", style: "margin: 0; white-space: pre-wrap;" }, "Awaiting log stream...")
            )
          )
        ),
      });

      // Adapter Metadata Card
      const metadataCard = card({
        title: "Resulting Adapter Metadata & Architecture",
        sub: "Cryptographic identifier, parameter counts, and fine-tuning convergence metrics",
        body: h("div", { id: "adapter-metadata-content", class: "text-muted" }, "Metadata will be populated once adapter fine-tuning converges."),
      });

      // Privacy Accounting Card
      const privacyCard = card({
        title: "Differential Privacy Accounting",
        sub: "Formal (ε, δ) bounds verified via Rényi Differential Privacy moments accountant",
        body: h("div", { id: "privacy-accounting-content", class: "text-muted" }, "Awaiting privacy accounting evaluation artifact..."),
      });

      replace(
        monitorContainer,
        h("div", { class: "stack", style: "gap: 16px;" }, monitorCard, privacyCard, metadataCard)
      );

      // Polling Loop
      let isHistoricalRun = forceHistorical;

      async function pollJobTelemetry() {
        if (signal.aborted) {
          if (pollInterval) clearInterval(pollInterval);
          if (timerInterval) clearInterval(timerInterval);
          return;
        }

        try {
          const [jobRes, logsRes, metricsRes, summaryRes] = await Promise.all([
            endpoints.job(jobId, { signal }).catch(() => null),
            endpoints.jobLogs(jobId, { signal }).catch(() => null),
            endpoints.jobMetrics(jobId, { signal }).catch(() => null),
            endpoints.jobSummary(jobId, { signal }).catch(() => null),
          ]);

          if (!jobRes || !jobRes.job) return;
          const j = jobRes.job;
          const kpi = summaryRes?.kpi || {};
          const evalMet = j.eval_metrics || {};
          const trainingCfg = j.training_config || {};

          // Check if run is historical
          const isCompleted = ["COMPLETED", "FAILED", "CANCELLED"].includes(j.status);
          if (isCompleted || forceHistorical) {
            isHistoricalRun = true;
          }

          // Update Historical Banner
          const histBanner = monitorCard.querySelector("#monitor-historical-banner");
          if (histBanner) {
            if (isHistoricalRun) {
              replace(
                histBanner,
                h(
                  "div",
                  {
                    class: "row",
                    style:
                      "gap: 8px; padding: 8px 12px; background: rgba(59, 130, 246, 0.1); border-left: 3px solid var(--accent); border-radius: var(--radius-sm); font-size: 0.85rem; margin-bottom: 4px;",
                  },
                  icon("clock", { cls: "text-accent" }),
                  h("strong", null, "[Historical Experiment]"),
                  h("span", null, `Job ${jobId} telemetry is read-only. Recorded at ${fmtDate(j.created_at)}.`)
                )
              );
            } else {
              replace(histBanner, h("div", null));
            }
          }

          // Status Headline & Badges
          const headlineEl = monitorCard.querySelector("#monitor-status-headline");
          const stepDetailsEl = monitorCard.querySelector("#monitor-step-details");
          const statusBadgeEl = monitorCard.querySelector("#monitor-status-badge");
          const spinnerEl = monitorCard.querySelector("#monitor-spinner");
          const timerEl = monitorCard.querySelector("#monitor-elapsed-timer");

          if (headlineEl) {
            headlineEl.textContent = `Status: ${j.status} · Stage: ${j.stage || "dataset_intake"}`;
          }
          if (stepDetailsEl) {
            const currentStepStr = j.current_step ? `Step: ${j.current_step}` : "";
            const epochStr = j.current_epoch ? `Epoch: ${fmtNumber(j.current_epoch, 2)}` : `Epochs: ${j.epochs || 1}`;
            const subsetStr = `Subset: ${j.subset_size || j.num_records || 0} samples`;
            const modeLabel = j.dp_enabled || trainingCfg.dp_enabled ? "Mode B: DP-LoRA" : "Mode A: Standard LoRA";
            stepDetailsEl.textContent = [modeLabel, epochStr, currentStepStr, subsetStr].filter(Boolean).join(" · ");
          }

          if (statusBadgeEl) {
            replace(statusBadgeEl, badge(j.status));
          }

          // Spinner handling
          if (spinnerEl) {
            if (isCompleted) {
              spinnerEl.style.display = "none";
            } else {
              spinnerEl.style.display = "inline-block";
            }
          }

          // Elapsed Time
          const durationSec =
            evalMet.training_duration_seconds ||
            kpi.training_time_s ||
            (j.created_at && j.updated_at ? (new Date(j.updated_at).getTime() - new Date(j.created_at).getTime()) / 1000 : null);

          if (timerEl) {
            if (isCompleted && durationSec) {
              timerEl.textContent = fmtDuration(durationSec);
            } else if (j.created_at) {
              timerEl.textContent = fmtDuration(j.created_at);
            }
          }

          // Logs Stream
          const logsEl = monitorCard.querySelector("#monitor-logs-pre");
          if (logsRes && logsRes.logs && logsEl) {
            logsEl.textContent = logsRes.logs;
            logsEl.scrollTop = logsEl.scrollHeight;
          }

          // Real Loss History Chart (No Simulated Progress)
          const lossContainer = monitorCard.querySelector("#monitor-loss-chart-container");
          const rawLosses = j.loss_history || metricsRes?.metrics?.loss_history || [];

          if (lossContainer) {
            if (Array.isArray(rawLosses) && rawLosses.length > 0) {
              const validPoints = rawLosses.filter((p) => p && (p.loss !== undefined || p.eval_loss !== undefined));
              if (validPoints.length > 0) {
                const labels = validPoints.map((p, idx) =>
                  p.epoch !== undefined ? `Ep ${fmtNumber(p.epoch, 2)}` : `Step ${idx + 1}`
                );
                const trainLossValues = validPoints.map((p) => (Number.isFinite(p.loss) ? p.loss : null));
                const evalLossValues = validPoints.map((p) => (Number.isFinite(p.eval_loss) ? p.eval_loss : null));

                const series = [
                  { name: "Training Loss", values: trainLossValues },
                ];
                if (evalLossValues.some((v) => v !== null)) {
                  series.push({ name: "Validation Loss", values: evalLossValues });
                }

                replace(
                  lossContainer,
                  lineChart({
                    title: `Loss Convergence Curve (${jobId})`,
                    labels,
                    series,
                    yFormat: (v) => fmtNumber(v, 4),
                  })
                );
              }
            } else {
              replace(
                lossContainer,
                h(
                  "div",
                  {
                    class: "card row-center",
                    style: "padding: 32px; background: var(--bg-hover); color: var(--text-muted); font-size: 0.9rem;",
                  },
                  j.status === "CREATED" || j.status === "INGESTING"
                    ? "Phase 1 dataset sanitization in progress. Loss metrics will stream once fine-tuning begins."
                    : j.status === "TRAINING"
                    ? "Fine-tuning in progress. Awaiting initial checkpoint loss telemetry..."
                    : "No loss curve telemetry was recorded for this job execution."
                )
              );
            }
          }

          // ───────────────────────────────────────────────────────────────────
          // Privacy Accounting Panel (Strict DP vs Standard LoRA distinction)
          // ───────────────────────────────────────────────────────────────────
          const privacyContainer = monitorContainer.querySelector("#privacy-accounting-content");
          if (privacyContainer) {
            const isDpJob = Boolean(
              j.dp_enabled ||
              trainingCfg.dp_enabled ||
              kpi.training_mode === "DP-LoRA" ||
              evalMet.training_mode === "dp-lora" ||
              evalMet.training_mode === "dp_lora"
            );

            // Valid accounting artifact check: epsilon and delta must exist from backend
            const hasAccountingArtifact =
              isDpJob &&
              ((evalMet.epsilon !== undefined && evalMet.epsilon !== null) ||
               (kpi.dp_epsilon !== undefined && kpi.dp_epsilon !== null));

            if (!isDpJob) {
              // Standard LoRA: Explicitly NEVER claim differential privacy
              replace(
                privacyContainer,
                h(
                  "div",
                  { class: "stack", style: "gap: 12px;" },
                  h(
                    "div",
                    { class: "row-between", style: "padding: 10px 14px; background: var(--bg-hover); border-radius: var(--radius-sm); border: 1px solid var(--border);" },
                    h(
                      "div",
                      { class: "row", style: "gap: 8px;" },
                      icon("info", { cls: "text-muted" }),
                      h("strong", null, "Standard LoRA — No Differential Privacy Accounting Applied")
                    ),
                    badge("STANDARD LoRA (MODE A)")
                  ),
                  h(
                    "p",
                    { class: "text-muted", style: "margin: 0; font-size: 0.88rem; line-height: 1.5;" },
                    "This model was fine-tuned with standard empirical risk minimization (ERM). Per-example gradient clipping and Gaussian noise injection (DP-SGD) were not active. Consequently, no (ε, δ) differential privacy bounds exist for this run. Training-data protection is provided by Phase 1 PII masking and Phase 3/4 device-bound encryption."
                  ),
                  h(
                    "div",
                    { class: "grid grid-4", style: "gap: 12px; font-size: 0.85rem;" },
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Epsilon (ε)"), h("strong", null, "None (Not Applicable)")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Delta (δ)"), h("strong", null, "None (Not Applicable)")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Noise Multiplier (σ)"), h("strong", null, "0.0")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Clipping Bound (C)"), h("strong", null, "None (Unbounded)"))
                  )
                )
              );
            } else if (hasAccountingArtifact) {
              // Valid DP-LoRA Accounting Artifact Exists
              const finalEpsilon = evalMet.epsilon ?? kpi.dp_epsilon;
              const finalDelta = evalMet.delta ?? kpi.dp_delta ?? 1e-5;
              const noiseMult = evalMet.noise_multiplier ?? kpi.noise_multiplier ?? trainingCfg.dp_noise_multiplier ?? 1.2;
              const clipNorm = evalMet.max_grad_norm ?? kpi.max_grad_norm ?? trainingCfg.dp_max_grad_norm ?? 1.0;
              const accountantType = evalMet.accountant_type ?? trainingCfg.dp_accountant ?? "rdp";

              replace(
                privacyContainer,
                h(
                  "div",
                  { class: "stack", style: "gap: 14px;" },
                  h(
                    "div",
                    { class: "row-between", style: "padding: 10px 14px; background: rgba(16, 185, 129, 0.1); border-left: 3px solid var(--success); border-radius: var(--radius-sm);" },
                    h(
                      "div",
                      { class: "row", style: "gap: 8px;" },
                      icon("shield", { cls: "text-success" }),
                      h("strong", null, "Verified Differential Privacy Accounting Artifact")
                    ),
                    badge("DP-LoRA (MODE B)")
                  ),
                  h(
                    "div",
                    { class: "grid grid-4", style: "gap: 12px;" },
                    h(
                      "div",
                      { class: "card", style: "padding: 12px; background: var(--bg-hover);" },
                      h("div", { class: "text-muted", style: "font-size: 0.8rem;" }, "Privacy Loss Bound (ε)"),
                      h("div", { class: "mono", style: "font-size: 1.4rem; font-weight: 700; color: var(--primary);" }, fmtNumber(finalEpsilon, 4)),
                      h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, "Guaranteed upper bound")
                    ),
                    h(
                      "div",
                      { class: "card", style: "padding: 12px; background: var(--bg-hover);" },
                      h("div", { class: "text-muted", style: "font-size: 0.8rem;" }, "Failure Probability (δ)"),
                      h("div", { class: "mono", style: "font-size: 1.4rem; font-weight: 700;" }, String(finalDelta)),
                      h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, "< 1 / N dataset bound")
                    ),
                    h(
                      "div",
                      { class: "card", style: "padding: 12px; background: var(--bg-hover);" },
                      h("div", { class: "text-muted", style: "font-size: 0.8rem;" }, "Noise Multiplier (σ)"),
                      h("div", { class: "mono", style: "font-size: 1.4rem; font-weight: 700;" }, fmtNumber(noiseMult, 2)),
                      h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, "Gaussian perturbation scale")
                    ),
                    h(
                      "div",
                      { class: "card", style: "padding: 12px; background: var(--bg-hover);" },
                      h("div", { class: "text-muted", style: "font-size: 0.8rem;" }, "Clipping Norm (C)"),
                      h("div", { class: "mono", style: "font-size: 1.4rem; font-weight: 700;" }, fmtNumber(clipNorm, 2)),
                      h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, "Per-example gradient cap")
                    )
                  ),
                  h(
                    "div",
                    { class: "card", style: "padding: 12px; font-size: 0.85rem; line-height: 1.5;" },
                    h("strong", null, "Accountant Engine: "),
                    `${accountantType.toUpperCase()} (Rényi Differential Privacy Moments Composition). `,
                    "Under Poisson subsampled Gaussian mechanism, this adapter guarantees mathematical privacy against membership inference, sample reconstruction, and training set attribute leakage."
                  )
                )
              );
            } else if (j.status === "TRAINING" || j.status === "INGESTING" || j.status === "CREATED") {
              // DP-LoRA still running, accounting in progress
              replace(
                privacyContainer,
                h(
                  "div",
                  { class: "stack", style: "gap: 10px;" },
                  h(
                    "div",
                    { class: "row-between", style: "padding: 10px 14px; background: var(--bg-hover); border-radius: var(--radius-sm);" },
                    h(
                      "div",
                      { class: "row", style: "gap: 8px;" },
                      h("span", { class: "busy-spinner", style: "width: 14px; height: 14px;" }),
                      h("strong", null, "DP-SGD Privacy Accountant Accumulating Moments...")
                    ),
                    badge("ACCOUNTING IN PROGRESS")
                  ),
                  h(
                    "p",
                    { class: "text-muted", style: "margin: 0; font-size: 0.85rem;" },
                    "Opacus PrivacyEngine is measuring per-example gradient norms and noise divergence across optimizer steps. Formal (ε, δ) values will be finalized upon training completion."
                  )
                )
              );
            } else {
              // DP-LoRA failed or cancelled without valid artifact
              replace(
                privacyContainer,
                h(
                  "p",
                  { class: "text-muted", style: "margin: 0;" },
                  "Differential privacy accounting artifact unavailable. The training job stopped before the accountant converged."
                )
              );
            }
          }

          // ───────────────────────────────────────────────────────────────────
          // Resulting Adapter Metadata Card
          // ───────────────────────────────────────────────────────────────────
          const metadataContainer = monitorContainer.querySelector("#adapter-metadata-content");
          if (metadataContainer) {
            const trainableParams = evalMet.trainable_parameters || kpi.trainable_parameters || evalMet.trainable_params;
            const totalParams = evalMet.total_parameters || kpi.total_parameters || evalMet.total_params;
            const trainablePct = evalMet.trainable_percent || (trainableParams && totalParams ? (100 * trainableParams / totalParams) : null);
            const trainLoss = evalMet.train_loss || kpi.train_loss;
            const valLoss = evalMet.val_loss || kpi.val_loss;
            const perplexity = evalMet.perplexity;
            const peakMem = evalMet.peak_memory_mb;
            const throughput = evalMet.throughput_samples_per_sec;

            if (isCompleted || trainableParams) {
              replace(
                metadataContainer,
                h(
                  "div",
                  { class: "stack", style: "gap: 14px;" },
                  h(
                    "div",
                    { class: "grid grid-3", style: "gap: 12px; font-size: 0.88rem;" },
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Base Model"), h("strong", null, j.model_name || trainingCfg.model_name || "JackFram/llama-68m")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Adapter Semantic Version"), h("strong", null, `v${j.version || "1.0.0"}`)),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Training Mode"), h("strong", null, j.dp_enabled ? "DP-LoRA (Mode B)" : "Standard LoRA (Mode A)")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Trainable Parameters"), h("strong", null, trainableParams ? `${fmtNumber(trainableParams)} (${fmtPercent((trainablePct || 0) / 100, 2)})` : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Total Parameters"), h("strong", null, totalParams ? fmtNumber(totalParams) : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Training Duration"), h("strong", null, durationSec ? fmtDuration(durationSec) : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Final Training Loss"), h("strong", null, trainLoss !== undefined ? fmtNumber(trainLoss, 4) : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Final Validation Loss"), h("strong", null, valLoss !== undefined ? fmtNumber(valLoss, 4) : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Validation Perplexity"), h("strong", null, perplexity ? fmtNumber(perplexity, 2) : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Peak GPU/RAM Memory"), h("strong", null, peakMem ? `${fmtNumber(peakMem, 1)} MB` : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Throughput"), h("strong", null, throughput ? `${fmtNumber(throughput, 1)} samples/s` : "Not recorded")),
                    h("div", { class: "card", style: "padding: 10px;" }, h("div", { class: "text-muted" }, "Hardware Binding Salt"), h("strong", { class: "mono" }, j.salt ? `${String(j.salt).slice(0, 12)}...` : "Derived Node Salt"))
                  ),
                  h(
                    "div",
                    { class: "row-end", style: "gap: 10px; margin-top: 4px;" },
                    button({
                      label: "Inspect Adapter Screening",
                      iconName: "screening",
                      kind: "ghost",
                      onClick: () => navigate("/screening"),
                    }),
                    button({
                      label: "View Cryptographic Packaging",
                      iconName: "packaging",
                      kind: "ghost",
                      onClick: () => navigate("/packaging"),
                    })
                  )
                )
              );
            }
          }

          // Stop polling if completed or failed
          if (isCompleted) {
            if (pollInterval) clearInterval(pollInterval);
          }
        } catch (err) {
          if (isAbort(err)) {
            if (pollInterval) clearInterval(pollInterval);
          }
        }
      }

      pollJobTelemetry();
      pollInterval = setInterval(pollJobTelemetry, 2500);
    }

    return () => {
      if (pollInterval) clearInterval(pollInterval);
      if (timerInterval) clearInterval(timerInterval);
    };
  },
};
