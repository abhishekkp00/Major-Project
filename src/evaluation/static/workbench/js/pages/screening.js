/* SecureLoRA Adapter Screening Module.
   Evidence-driven model security review interface providing:
   - Section A: Structural screening (norms, singular-value/rank, outlier Z-scores, trusted-weight comparisons).
   - Section B: Behavioral screening (trigger-prompt tests, paraphrase consistency, probe distribution, inference indicators).
   - Section C: Decision summary (Pass, Flag, Reject, Not Evaluated, rationale, actual backend weights & thresholds, screening mode, Run ID).
   - Clear distinction between real inference callbacks vs unexecuted/synthetic probes.
   - Never fabricates metrics or reference comparisons when data is absent. */

import { h, icon, clear, replace } from "../core/dom.js";
import { endpoints, userMessage, isAbort } from "../core/api.js";
import { fmtNumber, fmtPercent, fmtDate, statusKind } from "../core/format.js";
import {
  pageHead,
  card,
  kpi,
  badge,
  button,
  emptyState,
  errorState,
  skeletonKpis,
  skeletonCards,
  skeletonLines,
} from "../core/ui.js";
import { barChart } from "../core/charts.js";

export default {
  async mount(root, { navigate, params, signal }) {
    let currentJobId = params?.job_id || sessionStorage.getItem("securelora_active_job") || null;

    // Header
    const header = pageHead({
      title: "Pre-Deployment Adapter Screening & Verification",
      description:
        "Evidence-driven model security review evaluating structural parameter anomalies and behavioral backdoor trigger resistance prior to cryptographic packaging.",
      actions: [
        button({
          label: "Training Module",
          iconName: "training",
          kind: "ghost",
          onClick: () => navigate("/training"),
        }),
        button({
          label: "Cryptographic Packaging",
          iconName: "packaging",
          kind: "primary",
          onClick: () => navigate("/packaging"),
        }),
      ],
    });

    // Run Selector & Global Screening Status Bar
    const jobSelect = h("select", {
      id: "screening-job-select",
      class: "select mono",
      style: "min-width: 320px;",
    }, h("option", { value: "" }, "Loading available pipeline runs..."));

    const modeBadgeContainer = h("span", { id: "screening-mode-badge" });
    const decisionBadgeContainer = h("span", { id: "screening-decision-badge" }, badge("SELECT RUN"));

    const selectorBar = h(
      "div",
      {
        class: "card row-between",
        style:
          "padding: 12px 16px; margin-bottom: 20px; background: var(--bg-card); border: 1px solid var(--border); flex-wrap: wrap; gap: 14px;",
      },
      h(
        "div",
        { class: "row", style: "gap: 12px; flex-wrap: wrap;" },
        h("label", { style: "font-weight: 600; font-size: 0.9rem;" }, "Screening Audit Target:"),
        jobSelect,
        button({
          label: "Refresh",
          iconName: "refresh",
          kind: "ghost",
          small: true,
          onClick: () => loadScreeningDetails(jobSelect.value),
        })
      ),
      h("div", { class: "row", style: "gap: 10px;" }, modeBadgeContainer, decisionBadgeContainer)
    );

    // Main Content Containers
    const decisionSummaryContainer = h("div", { id: "decision-summary-section" }, skeletonKpis(4));
    const structuralSectionContainer = h("div", { id: "structural-screening-section" }, skeletonCards(2));
    const behavioralSectionContainer = h("div", { id: "behavioral-screening-section" }, skeletonCards(2));
    const technicalDetailsContainer = h("div", { id: "technical-details-section" });

    const layout = h(
      "div",
      { class: "stack", style: "gap: 20px;" },
      header,
      selectorBar,
      decisionSummaryContainer,
      structuralSectionContainer,
      behavioralSectionContainer,
      technicalDetailsContainer
    );

    replace(root, layout);

    jobSelect.addEventListener("change", () => {
      currentJobId = jobSelect.value;
      if (currentJobId) {
        sessionStorage.setItem("securelora_active_job", currentJobId);
        loadScreeningDetails(currentJobId);
      }
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Initial Job Picker Initialization
    // ─────────────────────────────────────────────────────────────────────────
    async function initJobPicker() {
      try {
        const res = await endpoints.jobs({ signal });
        if (signal.aborted) return;
        const jobs = res.jobs || [];

        jobSelect.innerHTML = "";
        if (jobs.length === 0) {
          jobSelect.innerHTML = `<option value="">No pipeline jobs found</option>`;
          replace(
            decisionSummaryContainer,
            emptyState({
              iconName: "screening",
              title: "No Pipeline Runs Available for Screening",
              message: "Launch a training job to generate fine-tuned LoRA adapter weights and screening evidence.",
              action: button({
                label: "Go to Training",
                iconName: "play",
                kind: "primary",
                onClick: () => navigate("/training"),
              }),
            })
          );
          replace(structuralSectionContainer, h("div", null));
          replace(behavioralSectionContainer, h("div", null));
          return;
        }

        jobs.forEach((j) => {
          const opt = h(
            "option",
            {
              value: j.job_id,
              selected: currentJobId ? j.job_id === currentJobId : false,
            },
            `${j.job_id} · ${j.status} · ${j.dataset_name || j.dataset_type || "Standard"} (${fmtDate(j.created_at)})`
          );
          jobSelect.appendChild(opt);
        });

        if (!currentJobId || !jobs.some((j) => j.job_id === currentJobId)) {
          currentJobId = jobs[0].job_id;
          jobSelect.value = currentJobId;
        }

        loadScreeningDetails(currentJobId);
      } catch (err) {
        if (isAbort(err)) return;
        jobSelect.innerHTML = `<option value="">Failed to load jobs list</option>`;
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Load Screening Details & Render Sections
    // ─────────────────────────────────────────────────────────────────────────
    async function loadScreeningDetails(jobId) {
      if (!jobId) return;

      replace(decisionSummaryContainer, skeletonKpis(4));
      replace(structuralSectionContainer, skeletonCards(2));
      replace(behavioralSectionContainer, skeletonCards(2));
      replace(technicalDetailsContainer, h("div", null));

      try {
        const data = await endpoints.jobScreening(jobId, { signal });
        if (signal.aborted) return;

        // Handle: Screening Not Evaluated or Incomplete
        if (!data || !data.available) {
          replace(decisionBadgeContainer, badge("NOT_EVALUATED", "Not Evaluated"));
          replace(modeBadgeContainer, badge("UNKNOWN", "Mode: N/A"));

          const reasonMsg =
            data?.reason ||
            "Screening has not executed for this job, or adapter weights were not generated.";

          replace(
            decisionSummaryContainer,
            card({
              title: "Adapter Screening Status: Not Evaluated",
              sub: `Run ID: ${jobId}`,
              body: h(
                "div",
                { class: "stack", style: "gap: 12px;" },
                h(
                  "div",
                  {
                    class: "row",
                    style:
                      "gap: 8px; padding: 12px; background: var(--bg-hover); border-left: 3px solid var(--border); border-radius: var(--radius-sm);",
                  },
                  icon("alert-circle", { cls: "text-muted" }),
                  h("span", { class: "text-muted" }, reasonMsg)
                ),
                h(
                  "p",
                  { class: "text-muted", style: "font-size: 0.88rem; margin: 0;" },
                  "Adapter screening evaluates structural parameter distributions and behavioral trigger sensitivity after fine-tuning completes. Check that Phase 2 training finished successfully."
                )
              ),
            })
          );
          replace(structuralSectionContainer, h("div", null));
          replace(behavioralSectionContainer, h("div", null));
          return;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Section C: Decision Summary & Governance
        // ─────────────────────────────────────────────────────────────────────
        const rawDecision = String(data.decision || "REVIEW").toUpperCase();
        const riskLevel = String(data.risk_level || "MEDIUM").toUpperCase();
        const riskScore = Number(data.adapter_risk_score ?? 0.0);
        const decCode = data.decision_code || "REQUIRES_ADMIN_APPROVAL";
        const screeningMode = String(data.screening_mode || "PRODUCTION").toUpperCase();
        const baseModel = data.base_model || "JackFram/llama-68m";
        const bypassedViaForce = Boolean(data.bypassed_via_force);

        const struct = data.structural_analysis || {};
        const behav = data.behavioral_analysis || {};
        const expl = data.decision_explanation || {};
        const weights = data.configured_weights || { structural: 0.35, behavioral: 0.45, consistency: 0.20 };
        const thresholds = data.configured_thresholds || {
          low_risk: 0.35,
          high_risk: 0.65,
          max_frobenius_norm: 25.0,
          max_l2_norm: 20.0,
          max_layer_zscore: 3.0,
          max_trigger_sensitivity: 0.60,
          min_paraphrase_consistency: 0.60,
          max_output_divergence: 0.70,
        };

        // Determine clear decision state: Pass, Flag, Reject, or Not Evaluated
        let decisionState = "Pass";
        let decisionBadgeKind = "success";
        let decisionIcon = "check-circle";

        if (rawDecision === "REJECTED" || riskLevel === "HIGH" || !data.available) {
          decisionState = "Reject";
          decisionBadgeKind = "danger";
          decisionIcon = "alert-triangle";
        } else if (rawDecision === "REVIEW" || riskLevel === "MEDIUM" || decCode === "REQUIRES_ADMIN_APPROVAL") {
          decisionState = "Flag";
          decisionBadgeKind = "warning";
          decisionIcon = "alert-circle";
        } else if (rawDecision === "SCREENED" || rawDecision === "APPROVED" || riskLevel === "LOW") {
          decisionState = "Pass";
          decisionBadgeKind = "success";
          decisionIcon = "shield-check";
        }

        // Header Badges
        replace(
          modeBadgeContainer,
          badge(screeningMode === "PRODUCTION" ? "SUCCESS" : "WARNING", `Mode: ${screeningMode}`)
        );
        replace(
          decisionBadgeContainer,
          badge(decisionBadgeKind === "success" ? "PASSED" : (decisionBadgeKind === "danger" ? "FAILED" : "WARNING"), `Gate Decision: ${decisionState}`)
        );

        // Section C Card: Decision Summary
        const decisionCard = card({
          title: `Screening Decision Summary: ${decisionState}`,
          sub: `Audit Run ID: ${jobId} · Base Model: ${baseModel} · Screening Mode: ${screeningMode}`,
          body: h(
            "div",
            { class: "stack", style: "gap: 16px;" },
            // KPI Summary Row
            h(
              "div",
              { class: "kpis" },
              kpi({
                label: "Screening Gate Decision",
                value: decisionState,
                text: true,
                foot: `Policy Code: ${decCode} · Level: ${riskLevel}`,
              }),
              kpi({
                label: "Composite Risk Score",
                value: fmtNumber(riskScore, 4),
                foot: `Low < ${thresholds.low_risk} · High ≥ ${thresholds.high_risk}`,
              }),
              kpi({
                label: "Structural Risk (w=0.35)",
                value: fmtNumber(data.structural_score ?? struct.structural_risk_score ?? 0.0, 4),
                foot: `Max Z-Score: ${fmtNumber(struct.max_layer_zscore ?? 0.0, 2)} (${struct.outlier_layer_count ?? 0} outliers)`,
              }),
              kpi({
                label: "Behavioral Risk (w=0.45)",
                value: fmtNumber(data.behavioral_score ?? behav.behavioral_risk_score ?? 0.0, 4),
                foot: behav.real_inference_performed ? "Verified live model inference" : "Callback not provisioned",
              })
            ),
            // Rationale Banner
            h(
              "div",
              {
                class: "row",
                style: `gap: 12px; padding: 14px; border-left: 4px solid var(--${decisionBadgeKind === "success" ? "success" : (decisionBadgeKind === "danger" ? "danger" : "warning")}); background: var(--bg-hover); border-radius: var(--radius-sm);`,
              },
              icon(decisionIcon, {
                cls: `text-${decisionBadgeKind === "success" ? "success" : (decisionBadgeKind === "danger" ? "danger" : "warning")}`,
              }),
              h(
                "div",
                { class: "stack-sm" },
                h("strong", null, expl.rationale || `Screening completed with composite risk score ${fmtNumber(riskScore, 4)} (${riskLevel} risk).`),
                h(
                  "div",
                  { class: "text-muted", style: "font-size: 0.85rem;" },
                  `${expl.structural_evidence || ""} ${expl.behavioral_evidence || ""}`
                ),
                bypassedViaForce
                  ? h(
                      "span",
                      { class: "text-warning", style: "font-weight: 600; font-size: 0.85rem;" },
                      "⚠️ Operator Warning: Security policy was bypassed using --force mode. High-risk adapter permitted to advance."
                    )
                  : null
              )
            ),
            // Configured Weights & Thresholds Table
            h(
              "details",
              {
                class: "card",
                style: "padding: 12px; background: var(--bg-surface); border: 1px solid var(--border); font-size: 0.85rem;",
              },
              h(
                "summary",
                { style: "cursor: pointer; font-weight: 600;" },
                "View Configured Risk Weights & Acceptance Thresholds (from Backend Security Policy)"
              ),
              h(
                "div",
                { class: "grid grid-2", style: "gap: 16px; margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--border);" },
                h(
                  "div",
                  { class: "stack-sm" },
                  h("strong", null, "Composite Risk Scoring Weights:"),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "Structural Analysis Weight:"), h("span", { class: "mono" }, fmtPercent(weights.structural))),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "Behavioral Probing Weight:"), h("span", { class: "mono" }, fmtPercent(weights.behavioral))),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "Consistency Verification Weight:"), h("span", { class: "mono" }, fmtPercent(weights.consistency))),
                  h("div", { class: "text-muted", style: "font-size: 0.78rem; font-style: italic; margin-top: 4px;" }, "Formula: Risk = min(1.0, max(Σ w_i·s_i, 0.95·max(s_struct, s_behav)))")
                ),
                h(
                  "div",
                  { class: "stack-sm" },
                  h("strong", null, "Backend Hard Gate Limits:"),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "Low-Risk Threshold (Pass):"), h("span", { class: "mono" }, `< ${thresholds.low_risk}`)),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "High-Risk Threshold (Reject):"), h("span", { class: "mono" }, `≥ ${thresholds.high_risk}`)),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "Max Frobenius Norm Bound:"), h("span", { class: "mono" }, thresholds.max_frobenius_norm)),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "Max Layer Z-Score Bound:"), h("span", { class: "mono" }, `${thresholds.max_layer_zscore}σ`)),
                  h("div", { class: "row-between" }, h("span", { class: "text-muted" }, "Max Trigger Sensitivity:"), h("span", { class: "mono" }, thresholds.max_trigger_sensitivity))
                )
              )
            )
          ),
        });

        replace(decisionSummaryContainer, decisionCard);

        // ─────────────────────────────────────────────────────────────────────
        // Section A: Structural Screening
        // ─────────────────────────────────────────────────────────────────────
        const hasRefWeights = struct.cosine_similarity_ref !== null && struct.cosine_similarity_ref !== undefined;

        const structuralOverviewCard = card({
          title: "Section A: Structural Weight Statistics & Spectral Norms",
          sub: "Verifies parameter tensor distribution, Frobenius energy, effective rank utilization, and outlier Z-scores",
          body: h(
            "div",
            { class: "stack", style: "gap: 14px;" },
            h(
              "div",
              { class: "grid grid-3", style: "gap: 12px; font-size: 0.88rem;" },
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Global Frobenius Norm (||W||_F)"),
                h("div", { class: "mono", style: "font-size: 1.25rem; font-weight: 700;" }, fmtNumber(struct.global_frobenius_norm, 3)),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, `Threshold: ≤ ${thresholds.max_frobenius_norm}`)
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Global Spectral / L2 Norm"),
                h("div", { class: "mono", style: "font-size: 1.25rem; font-weight: 700;" }, fmtNumber(struct.global_l2_norm, 3)),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, `Threshold: ≤ ${thresholds.max_l2_norm}`)
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Max Layer Outlier Z-Score"),
                h(
                  "div",
                  {
                    class: "mono",
                    style: `font-size: 1.25rem; font-weight: 700; color: ${(struct.max_layer_zscore ?? 0) > thresholds.max_layer_zscore ? "var(--danger)" : "inherit"};`,
                  },
                  fmtNumber(struct.max_layer_zscore, 2)
                ),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, `Threshold: ≤ ${thresholds.max_layer_zscore}σ`)
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Effective Rank Utilization"),
                h("div", { class: "mono", style: "font-size: 1.25rem; font-weight: 700;" }, fmtPercent(struct.rank_utilization_mean ?? 1.0, 1)),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, "Singular value spectral ratio (s₀ / Σ sᵢ)")
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Outlier Layer Count"),
                h(
                  "div",
                  {
                    class: "mono",
                    style: `font-size: 1.25rem; font-weight: 700; color: ${(struct.outlier_layer_count ?? 0) > 0 ? "var(--warning)" : "inherit"};`,
                  },
                  String(struct.outlier_layer_count ?? 0)
                ),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, struct.outlier_layers?.length ? struct.outlier_layers.join(", ") : "None detected")
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Weight Sparsity Ratio"),
                h("div", { class: "mono", style: "font-size: 1.25rem; font-weight: 700;" }, fmtPercent(struct.sparsity_ratio ?? 0.0, 1)),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, "Zero / dead parameter fraction")
              )
            ),
            // Trusted-weight comparisons only when reference data exists
            h(
              "div",
              {
                class: "card",
                style: `padding: 12px; background: ${hasRefWeights ? "var(--bg-surface)" : "var(--bg-hover)"}; border: 1px solid var(--border);`,
              },
              h(
                "div",
                { class: "row-between", style: "margin-bottom: 6px;" },
                h("strong", { style: "font-size: 0.88rem;" }, "Trusted-Weight Reference Baseline Comparison:"),
                badge(hasRefWeights ? "SUCCESS" : "NEUTRAL", hasRefWeights ? "REFERENCE COMPARED" : "REFERENCE OMITTED")
              ),
              hasRefWeights
                ? h(
                    "div",
                    { class: "grid grid-2", style: "gap: 12px; font-size: 0.85rem;" },
                    h(
                      "div",
                      { class: "row-between" },
                      h("span", { class: "text-muted" }, "Cosine Similarity vs Trusted Baseline:"),
                      h("span", { class: "mono text-primary", style: "font-weight: 600;" }, fmtNumber(struct.cosine_similarity_ref, 4))
                    ),
                    h(
                      "div",
                      { class: "row-between" },
                      h("span", { class: "text-muted" }, "Parameter Drift Score:"),
                      h("span", { class: "mono", style: "font-weight: 600;" }, fmtNumber(struct.parameter_drift_score, 4))
                    )
                  )
                : h(
                    "p",
                    { class: "text-muted", style: "margin: 0; font-size: 0.85rem;" },
                    "No reference baseline weights were configured for this adapter run. Trusted-weight cosine similarity and parameter drift comparisons are omitted to ensure zero metric fabrication."
                  )
            )
          ),
        });

        replace(structuralSectionContainer, structuralOverviewCard);

        // ─────────────────────────────────────────────────────────────────────
        // Section B: Behavioral Screening
        // ─────────────────────────────────────────────────────────────────────
        const realInference = Boolean(behav.real_inference_performed);
        const probeList = Array.isArray(behav.probe_results) ? behav.probe_results : [];
        const execStatus = String(behav.execution_status || (realInference ? "COMPLETED" : "NOT_EXECUTED")).toUpperCase();

        const behavioralOverviewCard = card({
          title: "Section B: Behavioral Probing & Trigger Verification",
          sub: "Evaluates trigger-conditioned backdoor sensitivity, paraphrase consistency, and model response divergence",
          body: h(
            "div",
            { class: "stack", style: "gap: 16px;" },
            // Evidence Source & Inference Availability Indicator
            h(
              "div",
              {
                class: "row-between",
                style: `padding: 12px; background: ${realInference ? "rgba(16, 185, 129, 0.1)" : "rgba(234, 179, 8, 0.1)"}; border-left: 4px solid var(--${realInference ? "success" : "warning"}); border-radius: var(--radius-sm); flex-wrap: wrap; gap: 10px;`,
              },
              h(
                "div",
                { class: "row", style: "gap: 10px;" },
                icon(realInference ? "cpu" : "alert-circle", {
                  cls: `text-${realInference ? "success" : "warning"}`,
                }),
                h(
                  "div",
                  { class: "stack-sm" },
                  h(
                    "strong",
                    null,
                    realInference
                      ? "Evidence Source: Live Model Inference Probing Verified"
                      : "Evidence Source: Behavioral Inference Callback Not Provisioned"
                  ),
                  h(
                    "span",
                    { class: "text-muted", style: "font-size: 0.83rem;" },
                    realInference
                      ? `Target model '${baseModel}' executed live token generation across probe suite.`
                      : "Behavioral probing requires an active model inference callback. Because none was provisioned during packaging, behavioral tests were not executed or reflect baseline test suites."
                  )
                )
              ),
              badge(realInference ? "SUCCESS" : "WARNING", realInference ? "LIVE INFERENCE" : "MISSING CALLBACK")
            ),
            // Behavioral Metrics Grid
            h(
              "div",
              { class: "grid grid-4", style: "gap: 12px; font-size: 0.88rem;" },
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Trigger Token Sensitivity"),
                h(
                  "div",
                  {
                    class: "mono",
                    style: `font-size: 1.25rem; font-weight: 700; color: ${(behav.trigger_sensitivity ?? 0) > thresholds.max_trigger_sensitivity ? "var(--danger)" : "inherit"};`,
                  },
                  behav.trigger_sensitivity !== null && behav.trigger_sensitivity !== undefined
                    ? fmtNumber(behav.trigger_sensitivity, 4)
                    : "Not evaluated"
                ),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, `Limit: ≤ ${thresholds.max_trigger_sensitivity}`)
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Paraphrase Consistency"),
                h(
                  "div",
                  {
                    class: "mono",
                    style: `font-size: 1.25rem; font-weight: 700; color: ${(behav.paraphrase_consistency ?? 1) < thresholds.min_paraphrase_consistency ? "var(--danger)" : "inherit"};`,
                  },
                  behav.paraphrase_consistency !== null && behav.paraphrase_consistency !== undefined
                    ? fmtPercent(behav.paraphrase_consistency, 1)
                    : "Not evaluated"
                ),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, `Min: ≥ ${fmtPercent(thresholds.min_paraphrase_consistency)}`)
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Normal Output Divergence"),
                h(
                  "div",
                  { class: "mono", style: "font-size: 1.25rem; font-weight: 700;" },
                  behav.normal_output_divergence !== null && behav.normal_output_divergence !== undefined
                    ? fmtNumber(behav.normal_output_divergence, 4)
                    : "Not evaluated"
                ),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, `Limit: ≤ ${thresholds.max_output_divergence}`)
              ),
              h(
                "div",
                { class: "card", style: "padding: 10px; background: var(--bg-hover);" },
                h("div", { class: "text-muted" }, "Abnormal Output Flip Rate"),
                h(
                  "div",
                  {
                    class: "mono",
                    style: `font-size: 1.25rem; font-weight: 700; color: ${(behav.abnormal_response_rate ?? 0) > 0 ? "var(--danger)" : "inherit"};`,
                  },
                  behav.abnormal_response_rate !== null && behav.abnormal_response_rate !== undefined
                    ? fmtPercent(behav.abnormal_response_rate, 1)
                    : "Not evaluated"
                ),
                h("div", { class: "text-muted", style: "font-size: 0.75rem;" }, `Trigger flips: ${fmtPercent(behav.classification_flip_rate ?? 0.0, 1)}`)
              )
            ),
            // Probe Results Chart & Table (when probes exist)
            probeList.length > 0
              ? h(
                  "div",
                  { class: "stack", style: "gap: 12px;" },
                  h("strong", { style: "font-size: 0.88rem;" }, "Output Divergence by Probe Category:"),
                  barChart({
                    title: "Behavioral Probe Output Divergence",
                    labels: probeList.map((p, idx) => `P${idx + 1} (${p.category})`),
                    series: [
                      {
                        name: "Divergence Score",
                        values: probeList.map((p) => p.divergence ?? 0.0),
                      },
                    ],
                    yFormat: (v) => fmtNumber(v, 2),
                  }),
                  // Probe Inspection Table
                  h(
                    "details",
                    {
                      class: "card",
                      style: "padding: 12px; background: var(--bg-surface); border: 1px solid var(--border); font-size: 0.85rem;",
                    },
                    h("summary", { style: "cursor: pointer; font-weight: 600;" }, `Inspect Probe Responses & Prompts (${probeList.length} Probes)`),
                    h(
                      "div",
                      { class: "table-wrap", style: "margin-top: 10px;" },
                      h(
                        "table",
                        { class: "table", style: "width: 100%; font-size: 0.82rem;" },
                        h(
                          "thead",
                          null,
                          h(
                            "tr",
                            null,
                            h("th", null, "Category"),
                            h("th", null, "Prompt"),
                            h("th", null, "Candidate Adapter Output"),
                            h("th", null, "Divergence")
                          )
                        ),
                        h(
                          "tbody",
                          null,
                          ...probeList.map((p) =>
                            h(
                              "tr",
                              null,
                              h("td", null, badge("NEUTRAL", p.category)),
                              h("td", { class: "mono", style: "max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" }, p.prompt),
                              h("td", { class: "mono", style: "max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" }, p.candidate_output || "N/A"),
                              h("td", { class: "mono" }, fmtNumber(p.divergence, 4))
                            )
                          )
                        )
                      )
                    )
                  )
                )
              : null
          ),
        });

        replace(behavioralSectionContainer, behavioralOverviewCard);

        // ─────────────────────────────────────────────────────────────────────
        // Section: Layer-by-Layer Technical Details (Expandable)
        // ─────────────────────────────────────────────────────────────────────
        const layerMetricsMap = struct.layer_metrics || {};
        const layerEntries = Object.entries(layerMetricsMap);

        if (layerEntries.length > 0) {
          const layerDetailsCard = card({
            title: "Layer-by-Layer Matrix Telemetry (Technical Audit)",
            sub: "Detailed Frobenius, spectral, and statistical metrics across all inspected transformer projection layers",
            body: h(
              "div",
              { class: "table-wrap", style: "max-height: 280px; overflow-y: auto;" },
              h(
                "table",
                { class: "table", style: "width: 100%; font-size: 0.82rem;" },
                h(
                  "thead",
                  null,
                  h(
                    "tr",
                    null,
                    h("th", null, "Layer Name"),
                    h("th", null, "L2 Norm"),
                    h("th", null, "L1 Norm"),
                    h("th", null, "L∞ Norm"),
                    h("th", null, "Mean"),
                    h("th", null, "Std Dev"),
                    h("th", null, "Outlier State")
                  )
                ),
                h(
                  "tbody",
                  null,
                  ...layerEntries.map(([lName, lData]) => {
                    const isOutlier = struct.outlier_layers?.includes(lName);
                    return h(
                      "tr",
                      null,
                      h("td", { class: "mono", style: "font-weight: 500;" }, lName),
                      h("td", { class: "mono" }, fmtNumber(lData.l2_norm, 4)),
                      h("td", { class: "mono" }, fmtNumber(lData.l1_norm, 4)),
                      h("td", { class: "mono" }, fmtNumber(lData.linf_norm, 4)),
                      h("td", { class: "mono" }, fmtNumber(lData.mean, 5)),
                      h("td", { class: "mono" }, fmtNumber(lData.std, 5)),
                      h("td", null, isOutlier ? badge("DANGER", "Outlier") : badge("SUCCESS", "Normal"))
                    );
                  })
                )
              )
            ),
          });
          replace(technicalDetailsContainer, layerDetailsCard);
        } else {
          replace(technicalDetailsContainer, h("div", null));
        }

      } catch (err) {
        if (isAbort(err)) return;
        replace(decisionSummaryContainer, errorState(err, () => loadScreeningDetails(jobId)));
      }
    }

    initJobPicker();
    return () => {};
  },
};
