import { h, icon, clear, replace } from "../core/dom.js";
import { endpoints, userMessage, isAbort } from "../core/api.js";
import { statusKind, statusLabel, fmtNumber, fmtPercent, fmtDate, fmtDuration, isNil } from "../core/format.js";
import { pageHead, card, kpi, badge, dataTable, button, emptyState, errorState, skeletonKpis, skeletonCards, skeletonLines, toast } from "../core/ui.js";

// Canonical 8-stage sequence requested by security architecture
const CANONICAL_8_STAGES = [
  { id: "dataset",      backendId: "dataset",      name: "Dataset Validation",       shortDesc: "Schema inspection & file sanitization" },
  { id: "pii_masking",  backendId: "pii_audit",    name: "PII Masking",              shortDesc: "In-memory hybrid entity redaction" },
  { id: "training",     backendId: "training",     name: "LoRA / Optional DP-LoRA",  shortDesc: "Fine-tuning with privacy accounting" },
  { id: "screening",    backendId: "screening",    name: "Adapter Screening",        shortDesc: "Dual-plane structural & behavioral gate" },
  { id: "signing",      backendId: "provenance",   name: "Integrity Signing",        shortDesc: "SHA-256 digest & RSA-PSS signature" },
  { id: "encryption",   backendId: "packaging",    name: "Device-Bound Encryption",  shortDesc: "AES-256-GCM sealed with HKDF key" },
  { id: "deployment",   backendId: "deployment",   name: "Deployment Verification",  shortDesc: "Client-side 8-gate fail-closed audit" },
  { id: "inference",    backendId: "inference",    name: "Inference Validation",     shortDesc: "Live side-by-side verification pass" }
];

export default {
  async mount(root, { navigate, signal }) {
    let pollTimer = null;
    let selectedJobId = sessionStorage.getItem("securelora_active_job") || null;
    let cachedJobs = [];

    // 1. Page Header with Prominent "New Pipeline Run" Action
    const header = pageHead({
      title: "Security Operations Overview",
      description: "Real-time posture of the SecureLoRA adapter security lifecycle, active verification gates, and evaluation provenance.",
      actions: [
        button({ label: "New Pipeline Run", iconName: "play", kind: "primary", onClick: () => navigate("/training") }),
        button({ label: "Deployment Verification", iconName: "deployment", onClick: () => navigate("/deployment") })
      ]
    });

    // 2. Summary Cards Container
    const summaryCardsContainer = h("div", null, skeletonKpis(4));

    // 3. Eight-Stage Pipeline Section
    const pipelineSelector = h("select", { class: "select mono", style: "min-width: 260px; font-size: 0.85rem;" },
      h("option", { value: "" }, "Loading pipeline runs...")
    );

    const pipelineGridContainer = h("div", null, skeletonCards(4));

    const pipelineCard = card({
      title: "Eight-Stage Security Pipeline Visualization",
      sub: "Real-time execution status, duration, and cryptographic evidence across all 8 security gates",
      actions: [
        h("div", { class: "row", style: "gap: 8px;" },
          h("label", { class: "small muted", style: "align-self: center;" }, "Inspecting Run:"),
          pipelineSelector
        )
      ],
      body: pipelineGridContainer
    });

    // 4. Recent Runs Table & Evaluation Panel
    const recentRunsContainer = h("div", null, skeletonLines(4));
    const recentRunsCard = card({
      title: "Recent Pipeline Executions",
      sub: "Historical runs with dataset mode, status, duration, and inspector shortcuts",
      actions: [
        button({ label: "View All Runs", iconName: "history", small: true, kind: "ghost", onClick: () => navigate("/history") })
      ],
      body: recentRunsContainer
    });

    const evalPanelContainer = h("div", null, skeletonLines(4));
    const evalPanelCard = card({
      title: "Verified Research Evaluation",
      sub: "Canonical benchmark metrics, tamper resistance, and device binding posture",
      actions: [
        button({ label: "Detailed Reports", iconName: "evaluation", small: true, kind: "ghost", onClick: () => navigate("/evaluation") })
      ],
      body: evalPanelContainer
    });

    const layout = h("div", { class: "stack", style: "gap: 20px;" },
      header,
      summaryCardsContainer,
      pipelineCard,
      h("div", { class: "grid grid-2", style: "gap: 20px;" },
        recentRunsCard,
        evalPanelCard
      )
    );

    replace(root, layout);

    // Event listener on pipeline job selector
    pipelineSelector.addEventListener("change", (e) => {
      selectedJobId = e.target.value;
      if (selectedJobId) {
        sessionStorage.setItem("securelora_active_job", selectedJobId);
        loadPipelineStages(selectedJobId);
      }
    });

    // Map backend stage list into the canonical 8 stages
    function mapToCanonicalStages(backendStages = [], job = {}) {
      const stageMap = new Map();
      for (const bs of backendStages) {
        stageMap.set(bs.id, bs);
      }

      return CANONICAL_8_STAGES.map((def, idx) => {
        const bs = stageMap.get(def.backendId) || {};
        let rawStatus = bs.status || "PENDING";
        
        // Never turn a stage green merely because overall job is COMPLETED
        // If check was not executed or metrics are empty for verification stages, mark as UNAVAILABLE or PENDING
        if (def.backendId === "device_auth" || def.backendId === "deployment" || def.backendId === "inference") {
          if (!bs.status || bs.status === "PENDING") {
            rawStatus = job.status === "COMPLETED" && bs.status !== "PASSED" ? "UNAVAILABLE" : "PENDING";
          }
        }

        // Duration formatting
        let durationStr = null;
        if (bs.metrics && bs.metrics.training_time_s) {
          durationStr = fmtDuration(bs.metrics.training_time_s);
        } else if (def.backendId === "training" && job.training_time_s) {
          durationStr = fmtDuration(job.training_time_s);
        }

        return {
          stepNum: idx + 1,
          id: def.id,
          name: def.name,
          shortDesc: def.shortDesc,
          status: rawStatus,
          duration: durationStr,
          purpose: bs.purpose || "Stage verification check.",
          securitySignificance: bs.security_significance || "Security policy invariant check.",
          result: bs.result || "Evidence Unavailable",
          metrics: bs.metrics || {}
        };
      });
    }

    // Render the 8-stage visualizer
    function renderPipelineStages(stages, job = {}) {
      if (!stages || stages.length === 0) {
        replace(pipelineGridContainer, emptyState({
          iconName: "screening",
          title: "No Pipeline Stages Available",
          message: "Select a pipeline run or launch a new training job to inspect live stages."
        }));
        return;
      }

      const stageCards = stages.map((st) => {
        const kind = statusKind(st.status);
        let statusClass = "stage--pending";
        if (kind === "success") statusClass = "stage--success";
        else if (kind === "danger") statusClass = "stage--danger";
        else if (kind === "running") statusClass = "stage--running";
        else if (kind === "warning") statusClass = "stage--warning";

        const isRunning = kind === "running";
        const hasEvidence = st.metrics && Object.keys(st.metrics).length > 0;

        // Metric chips for summary view
        const keyMetrics = Object.entries(st.metrics)
          .filter(([k, v]) => !isNil(v) && typeof v !== "object")
          .slice(0, 3)
          .map(([k, v]) => h("div", { class: "row-between small", style: "padding: 2px 0;" },
            h("span", { class: "muted" }, k.replace(/_/g, " ")),
            h("span", { class: "mono", style: "font-size: 11px;" }, String(v))
          ));

        // Evidence panel (expandable)
        const evidencePanel = h("div", {
          class: "stage-evidence-panel",
          id: `evidence-${st.id}`,
          "data-open": "false"
        },
          h("div", { class: "stack", style: "gap: 6px;" },
            h("div", { class: "small" },
              h("strong", null, "Purpose: "),
              h("span", { class: "muted" }, st.purpose)
            ),
            h("div", { class: "small" },
              h("strong", null, "Security Significance: "),
              h("span", { class: "muted" }, st.securitySignificance)
            ),
            h("div", { class: "small" },
              h("strong", null, "Result Evidence: "),
              h("span", { class: "mono text-teal" }, st.result)
            ),
            hasEvidence ? h("div", { class: "card", style: "background: var(--surface); padding: 8px; margin-top: 4px;" },
              h("div", { class: "small", style: "font-weight: 600; margin-bottom: 4px;" }, "Stage Telemetry:"),
              ...Object.entries(st.metrics).map(([k, v]) =>
                h("div", { class: "row-between small", style: "padding: 2px 0; border-bottom: 1px dotted var(--border);" },
                  h("span", { class: "muted" }, k),
                  h("span", { class: "mono" }, typeof v === "object" ? JSON.stringify(v) : String(v ?? "None"))
                )
              )
            ) : null
          )
        );

        const toggleBtn = h("button", {
          type: "button",
          class: "stage-evidence-toggle",
          "aria-expanded": "false",
          onClick: (e) => {
            const isOpen = evidencePanel.getAttribute("data-open") === "true";
            evidencePanel.setAttribute("data-open", (!isOpen).toString());
            toggleBtn.setAttribute("aria-expanded", (!isOpen).toString());
            const textSpan = toggleBtn.querySelector(".toggle-text");
            if (textSpan) textSpan.textContent = isOpen ? "View Evidence" : "Hide Evidence";
          }
        },
          h("span", { class: "toggle-text" }, "View Evidence"),
          icon("chevron-down")
        );

        return h("div", { class: `pipeline-stage ${statusClass}`, id: `pipeline-stage-${st.id}` },
          h("div", { class: "row-between", style: "align-items: flex-start;" },
            h("div", { class: "row", style: "gap: 6px; align-items: center;" },
              isRunning ? h("span", { class: "stage-runner-indicator" }) : null,
              h("span", { class: "mono text-teal small", style: "font-weight: 700;" }, `0${st.stepNum}`),
              h("strong", { style: "font-size: 0.9rem;" }, st.name)
            ),
            badge(st.status)
          ),
          h("div", { class: "muted small", style: "font-size: 11px;" }, st.shortDesc),
          st.duration ? h("div", { class: "small text-teal mono", style: "font-size: 11px;" }, `Duration: ${st.duration}`) : null,
          keyMetrics.length > 0 ? h("div", { style: "margin: 4px 0;" }, ...keyMetrics) : null,
          toggleBtn,
          evidencePanel
        );
      });

      replace(pipelineGridContainer, h("div", { class: "pipeline-grid" }, ...stageCards));
    }

    // Load pipeline stages for a specific job
    async function loadPipelineStages(jobId) {
      if (!jobId) {
        replace(pipelineGridContainer, emptyState({
          iconName: "inbox",
          title: "No Run Selected",
          message: "Select a pipeline run above or launch a new run."
        }));
        return;
      }

      try {
        const [summaryRes, jobRes] = await Promise.all([
          endpoints.jobSummary(jobId, { signal }).catch(() => null),
          endpoints.job(jobId, { signal }).catch(() => null)
        ]);

        if (signal.aborted) return;
        const job = jobRes?.job || {};
        const backendStages = summaryRes?.stages || [];

        const canonicalStages = mapToCanonicalStages(backendStages, job);
        renderPipelineStages(canonicalStages, job);

      } catch (err) {
        if (isAbort(err)) return;
        replace(pipelineGridContainer, h("div", { class: "notice notice-error" }, userMessage(err)));
      }
    }

    // Main data fetch & state loader
    async function loadOverviewData() {
      try {
        const [jobsRes, p4Res, summaryRes] = await Promise.all([
          endpoints.jobs({ signal }).catch(() => ({ jobs: [] })),
          endpoints.phase4Status({ signal }).catch(() => ({ loaded: false })),
          endpoints.research("summary", { signal }).catch(() => ({ available: false }))
        ]);

        if (signal.aborted) return;

        cachedJobs = jobsRes.jobs || [];
        const totalRuns = cachedJobs.length;
        const successfulRuns = cachedJobs.filter(j => j.status === "COMPLETED").length;
        const failedRuns = cachedJobs.filter(j => j.status === "FAILED").length;
        const runningRuns = cachedJobs.filter(j => j.status === "RUNNING" || j.status === "STARTED" || j.status === "TRAINING").length;

        // 2. Render Summary Cards from Actual Backend Data Only
        replace(summaryCardsContainer, h("div", { class: "kpis" },
          kpi({
            label: "Total Pipeline Runs",
            value: fmtNumber(totalRuns, 0),
            foot: totalRuns > 0 ? "Cumulative orchestration jobs" : "No jobs initialized yet"
          }),
          kpi({
            label: "Successful Runs",
            value: fmtNumber(successfulRuns, 0),
            foot: totalRuns > 0 ? `${fmtPercent(successfulRuns / (totalRuns || 1), 1)} success rate` : "No completed jobs"
          }),
          kpi({
            label: "Failed Runs",
            value: fmtNumber(failedRuns, 0),
            foot: failedRuns > 0 ? "Blocked by security gates or errors" : "Zero fail-closed rejections"
          }),
          kpi({
            label: "Currently Running",
            value: fmtNumber(runningRuns, 0),
            foot: runningRuns > 0 ? "Active pipeline execution in progress" : "No active fine-tuning jobs"
          })
        ));

        // Update Job Selector for 8-Stage Pipeline
        pipelineSelector.innerHTML = "";
        if (cachedJobs.length === 0) {
          pipelineSelector.innerHTML = `<option value="">No pipeline runs found</option>`;
          replace(pipelineGridContainer, emptyState({
            iconName: "inbox",
            title: "No Pipeline Runs Found",
            message: "Launch your first privacy-screened fine-tuning run to populate pipeline stages.",
            action: button({ label: "New Pipeline Run", iconName: "play", kind: "primary", onClick: () => navigate("/training") })
          }));
        } else {
          cachedJobs.forEach(j => {
            const opt = h("option", {
              value: j.job_id,
              selected: selectedJobId ? j.job_id === selectedJobId : false
            }, `${j.job_id} (${j.dataset_name || j.dataset_type || "Standard"} — ${j.status})`);
            pipelineSelector.appendChild(opt);
          });

          if (!selectedJobId || !cachedJobs.some(j => j.job_id === selectedJobId)) {
            selectedJobId = cachedJobs[0].job_id;
            pipelineSelector.value = selectedJobId;
          }

          loadPipelineStages(selectedJobId);
        }

        // 5. Render Recent Runs Table
        const recentRunsBody = recentRunsCard.querySelector(".card-body");
        if (cachedJobs.length === 0) {
          replace(recentRunsBody, emptyState({
            iconName: "inbox",
            title: "No Recent Runs",
            message: "Executed runs will appear here with timing and dataset metadata."
          }));
        } else {
          const recentList = cachedJobs.slice(0, 6);
          const table = dataTable({
            columns: [
              { key: "job_id", label: "Run ID", render: r => h("strong", { class: "mono" }, r.job_id) },
              { key: "dataset", label: "Dataset", render: r => r.dataset_name || r.dataset_type || "Standard" },
              {
                key: "mode",
                label: "Training Mode",
                render: r => {
                  const dp = r.training_config?.dp_enabled || r.dp_enabled;
                  const eps = r.training_config?.dp_epsilon || r.dp_epsilon;
                  return dp ? h("span", { class: "mono text-teal" }, `DP-LoRA (ε=${fmtNumber(eps || 2.44, 2)})`) : "LoRA (Standard)";
                }
              },
              { key: "status", label: "Status", render: r => badge(r.status) },
              { key: "created_at", label: "Start Time", render: r => fmtDate(r.created_at) },
              {
                key: "duration",
                label: "Duration",
                render: r => {
                  if (r.status === "RUNNING" || r.status === "STARTED") return h("span", { class: "text-teal" }, "In Progress");
                  return fmtDuration(r.created_at, r.updated_at);
                }
              },
              {
                key: "actions",
                label: "Actions",
                render: r => h("div", { class: "row", style: "gap: 6px;" },
                  button({
                    label: "Stages",
                    iconName: "screening",
                    small: true,
                    kind: "ghost",
                    onClick: () => {
                      selectedJobId = r.job_id;
                      pipelineSelector.value = r.job_id;
                      sessionStorage.setItem("securelora_active_job", r.job_id);
                      loadPipelineStages(r.job_id);
                      pipelineCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
                    }
                  }),
                  button({
                    label: "Inspect",
                    iconName: "chevron-right",
                    small: true,
                    kind: "secondary",
                    onClick: () => navigate("/history", { job_id: r.job_id })
                  })
                )
              }
            ],
            rows: recentList
          });
          replace(recentRunsBody, table);
        }

        // 6. Render Recent Evaluation Panel (Using Real Metric Data When Available)
        const evalBody = evalPanelCard.querySelector(".card-body");
        const sec = summaryRes.available && summaryRes.security ? summaryRes.security : null;
        const perf = summaryRes.available && summaryRes.performance ? summaryRes.performance : null;

        const evalRows = [
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Screening F1 Score"),
            sec && sec.screening_f1_score !== undefined
              ? h("strong", { class: "mono text-teal" }, fmtPercent(sec.screening_f1_score, 1))
              : h("span", { class: "muted small" }, "Evidence Unavailable")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Cross-Device Rejection Rate"),
            sec && sec.cross_device_rejection_rate !== undefined
              ? h("strong", { class: "mono text-teal" }, fmtPercent(sec.cross_device_rejection_rate, 1))
              : h("span", { class: "muted small" }, "Evidence Unavailable")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Tamper Rejection Rate"),
            sec && sec.tamper_rejection_rate !== undefined
              ? h("strong", { class: "mono text-teal" }, fmtPercent(sec.tamper_rejection_rate, 1))
              : h("span", { class: "muted small" }, "Evidence Unavailable")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Deployment Verification"),
            p4Res.loaded
              ? badge("VERIFIED", "Adapter Active in Memory")
              : badge("UNVERIFIED", "Weights Locked / Unverified")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Target Base Architecture"),
            h("span", { class: "mono" }, p4Res.base_model_name || "JackFram/llama-68m")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Device Salt Fingerprint"),
            p4Res.salt_masked
              ? h("span", { class: "mono" }, p4Res.salt_masked)
              : h("span", { class: "muted small" }, "Not Evaluated")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0;" },
            h("span", { class: "muted" }, "Hardware Attestation Model"),
            h("span", { class: "small muted", style: "max-width: 260px; text-align: right;" },
              "Software-derived platform binding. Does not require or assert hardware TPM / enclave backing."
            )
          )
        ];

        replace(evalBody, h("div", { class: "stack", style: "gap: 4px;" }, ...evalRows));

        // Setup background polling if any job is currently running
        if (runningRuns > 0 && !pollTimer && !signal.aborted) {
          pollTimer = setInterval(async () => {
            if (signal.aborted) {
              clearInterval(pollTimer);
              return;
            }
            try {
              const freshRes = await endpoints.jobs({ signal });
              if (freshRes.jobs) {
                cachedJobs = freshRes.jobs;
                const stillRunning = cachedJobs.some(j => j.status === "RUNNING" || j.status === "STARTED" || j.status === "TRAINING");
                if (selectedJobId) {
                  loadPipelineStages(selectedJobId);
                }
                if (!stillRunning) {
                  clearInterval(pollTimer);
                  pollTimer = null;
                  loadOverviewData();
                }
              }
            } catch (_) {}
          }, 3000);
        }

      } catch (err) {
        if (isAbort(err)) return;
        replace(summaryCardsContainer, errorState(err, loadOverviewData));
      }
    }

    loadOverviewData();

    return () => {
      if (pollTimer) clearInterval(pollTimer);
    };
  }
};
