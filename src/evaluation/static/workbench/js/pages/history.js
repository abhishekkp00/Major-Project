import { h, icon, clear, replace } from "../core/dom.js";
import { endpoints, userMessage, isAbort, downloadAuthenticated } from "../core/api.js";
import { fmtNumber, fmtBytes, fmtDate, fmtDuration, statusKind } from "../core/format.js";
import { pageHead, card, badge, button, alertBox, dataTable, emptyState, errorState, skeletonLines, skeletonCards, notifySuccess, notifyError } from "../core/ui.js";

export default {
  async mount(root, { navigate, params, signal }) {
    let allJobs = [];
    let selectedJob = null;
    let activeInspectorTab = "pipeline"; // "pipeline" | "config" | "logs" | "artifacts"
    const initialJobId = params?.job_id || sessionStorage.getItem("securelora_active_job") || null;

    const header = pageHead({
      title: "Orchestration Run History & Audit Log",
      description: "Searchable audit trail of all LoRA/DP-LoRA fine-tuning runs, adapter screening evaluations, cryptographic packaging, and deployment verification outcomes.",
      actions: [
        button({ label: "Launch New Run", iconName: "play", kind: "primary", onClick: () => navigate("/training") }),
        button({ label: "Evaluation Benchmarks", iconName: "evaluation", onClick: () => navigate("/evaluation") })
      ]
    });

    const searchInput = h("input", {
      type: "search",
      class: "input",
      placeholder: "Search runs by Job ID, Dataset, Model...",
      style: "max-width: 300px;"
    });

    const statusFilter = h("select", { class: "select", style: "min-width: 170px;" },
      h("option", { value: "ALL" }, "All Statuses"),
      h("option", { value: "COMPLETED" }, "Completed / Verified"),
      h("option", { value: "RUNNING" }, "Running / Active"),
      h("option", { value: "FAILED" }, "Failed / Policy Rejected"),
      h("option", { value: "CANCELLED" }, "Cancelled"),
      h("option", { value: "INCOMPLETE" }, "Incomplete / Staged")
    );

    const sortFilter = h("select", { class: "select", style: "min-width: 140px;" },
      h("option", { value: "NEWEST" }, "Newest First"),
      h("option", { value: "OLDEST" }, "Oldest First")
    );

    const filterBar = h("div", {
      class: "row-between",
      style: "background: var(--bg-surface); padding: 12px 16px; border: 1px solid var(--border); border-radius: var(--radius-md); gap: 12px; flex-wrap: wrap;"
    },
      h("div", { class: "row", style: "gap: 12px; flex-wrap: wrap;" },
        searchInput,
        statusFilter,
        sortFilter
      ),
      h("div", { id: "history-count-badge", class: "small muted" }, "Loading runs...")
    );

    const tableContainer = h("div", null, skeletonLines(6));
    const detailDrawer = h("div", {
      class: "card",
      id: "history-detail-drawer",
      style: "display: none; padding: 20px; background: var(--bg-surface);"
    });

    const layout = h("div", { class: "stack" },
      header,
      filterBar,
      h("div", { class: "grid grid-2", style: "grid-template-columns: 1fr;" },
        card({
          title: "Pipeline Executions",
          sub: "Select any run to open its complete pipeline stages, configurations, live logs, and cryptographic artifacts.",
          body: tableContainer
        }),
        detailDrawer
      )
    );

    replace(root, layout);

    function updateTable() {
      const q = searchInput.value.toLowerCase().trim();
      const stFilter = statusFilter.value;
      const order = sortFilter.value;

      let filtered = allJobs.filter(j => {
        const s = (j.status || "").toUpperCase();
        if (stFilter === "COMPLETED" && s !== "COMPLETED") return false;
        if (stFilter === "RUNNING" && !["RUNNING", "STARTED", "STARTING", "INGESTING", "TRAINING"].includes(s)) return false;
        if (stFilter === "FAILED" && !["FAILED", "SECURITY_POLICY_REJECTED", "SECURITY_SCREENING_FAILED"].includes(s)) return false;
        if (stFilter === "CANCELLED" && !["CANCELLED", "CANCELED"].includes(s)) return false;
        if (stFilter === "INCOMPLETE" && !["CREATED", "PENDING", "STAGED"].includes(s)) return false;

        if (q) {
          const matchId = (j.job_id || "").toLowerCase().includes(q);
          const matchDs = (j.dataset_name || j.dataset_type || "").toLowerCase().includes(q);
          const matchMod = (j.base_model_name || "").toLowerCase().includes(q);
          return matchId || matchDs || matchMod;
        }
        return true;
      });

      // Sort by created_at date
      filtered.sort((a, b) => {
        const tA = new Date(a.created_at || 0).getTime();
        const tB = new Date(b.created_at || 0).getTime();
        return order === "OLDEST" ? tA - tB : tB - tA;
      });

      const countBadge = filterBar.querySelector("#history-count-badge");
      if (countBadge) countBadge.textContent = `Showing ${filtered.length} of ${allJobs.length} runs`;

      if (filtered.length === 0) {
        replace(tableContainer, emptyState({
          iconName: "inbox",
          title: "No Runs Found",
          message: allJobs.length === 0 ? "No jobs have been launched yet." : "No runs matched your search and filter criteria."
        }));
        return;
      }

      const table = dataTable({
        columns: [
          { key: "job_id", label: "Job ID", render: j => h("strong", { class: "mono text-teal" }, j.job_id) },
          { key: "dataset_name", label: "Dataset & Scope", render: j => j.dataset_name || j.dataset_type || "Standard" },
          { key: "training_mode", label: "Mode", render: j => j.dp_enabled ? h("span", { class: "badge", dataset: { kind: "running" } }, "DP-LoRA") : "LoRA" },
          { key: "status", label: "Status", render: j => badge(j.status) },
          { key: "epochs", label: "Epochs", render: j => fmtNumber(j.epochs || 1, 0) },
          { key: "subset_size", label: "Subset", render: j => fmtNumber(j.subset_size || j.num_records || 0, 0) },
          { key: "created_at", label: "Created At", render: j => fmtDate(j.created_at) }
        ],
        rows: filtered,
        onRowClick: (j) => selectJob(j)
      });

      replace(tableContainer, table);
    }

    async function selectJob(job) {
      selectedJob = job;
      sessionStorage.setItem("securelora_active_job", job.job_id);

      detailDrawer.style.display = "block";
      replace(detailDrawer, h("div", { class: "stack" }, skeletonLines(6)));

      try {
        const [jobRes, summaryRes, logsRes, artifactsRes] = await Promise.all([
          endpoints.job(job.job_id, { signal }).catch(() => ({ job })),
          endpoints.jobSummary(job.job_id, { signal }).catch(() => null),
          endpoints.jobLogs(job.job_id, { signal }).catch(() => ({ logs: "No logs found." })),
          endpoints.jobArtifacts(job.job_id, { signal }).catch(() => ({ artifacts: [] }))
        ]);
        if (signal.aborted) return;

        const liveJob = jobRes?.job || job;
        const stages = summaryRes?.stages || [];
        const logs = logsRes?.logs || "Waiting for execution logs...";
        const artifacts = artifactsRes?.artifacts || [];

        renderDrawerContent(liveJob, stages, logs, artifacts);
        detailDrawer.scrollIntoView({ behavior: "smooth", block: "nearest" });

      } catch (err) {
        if (isAbort(err)) return;
        replace(detailDrawer, errorState(err, () => selectJob(job)));
      }
    }

    function renderDrawerContent(job, stages, logs, artifacts) {
      const isFailed = ["FAILED", "SECURITY_POLICY_REJECTED", "SECURITY_SCREENING_FAILED"].includes(job.status);
      const isCancelled = ["CANCELLED", "CANCELED"].includes(job.status);
      const isIncomplete = ["CREATED", "PENDING", "STAGED"].includes(job.status);

      // Drawer Header
      const drawerHeader = h("div", { class: "row-between", style: "border-bottom: 1px solid var(--border); padding-bottom: 12px;" },
        h("div", null,
          h("div", { class: "row", style: "gap: 8px; align-items: center;" },
            h("h3", { style: "margin: 0; font-size: 1.15rem;" }, `Run Details: ${job.job_id}`),
            badge(job.status)
          ),
          h("div", { class: "small muted", style: "margin-top: 4px;" }, `Created ${fmtDate(job.created_at)} · Base Model: ${job.base_model_name || "JackFram/llama-68m"}`)
        ),
        h("div", { class: "row", style: "gap: 8px;" },
          button({
            label: "Close",
            iconName: "close",
            small: true,
            kind: "ghost",
            onClick: () => { detailDrawer.style.display = "none"; }
          })
        )
      );

      // Failure / Cancellation / Incomplete Rationale Panel
      let failurePanel = null;
      if (isFailed) {
        failurePanel = alertBox(
          "danger",
          `Execution Halted with Status: ${job.status}`,
          job.error
            ? ` Failure Rationale: ${job.error}`
            : " The pipeline was rejected or halted by a security policy gate. Review the screening or deployment verification stage for details.",
          h("div", { class: "small mono text-teal", style: "margin-top: 6px;" }, `Failure Stage: ${job.stage || "Pipeline Gate"}`)
        );
      } else if (isCancelled) {
        failurePanel = alertBox(
          "warning",
          "Pipeline Execution Cancelled",
          " This run was explicitly cancelled by the operator prior to final verification.",
          job.error ? h("div", { class: "small muted" }, job.error) : null
        );
      } else if (isIncomplete) {
        failurePanel = alertBox(
          "info",
          "Incomplete Pipeline Run",
          " This run is staged in CREATED status and has not yet started active fine-tuning or screening."
        );
      }

      // Drawer Tabs
      const tabPipelineBtn = h("button", {
        type: "button",
        class: `btn${activeInspectorTab === "pipeline" ? " btn-primary" : ""}`,
        style: "padding: 6px 12px; font-size: 0.85rem;"
      }, "1. Pipeline Lifecycle");

      const tabConfigBtn = h("button", {
        type: "button",
        class: `btn${activeInspectorTab === "config" ? " btn-primary" : ""}`,
        style: "padding: 6px 12px; font-size: 0.85rem;"
      }, "2. Configuration");

      const tabLogsBtn = h("button", {
        type: "button",
        class: `btn${activeInspectorTab === "logs" ? " btn-primary" : ""}`,
        style: "padding: 6px 12px; font-size: 0.85rem;"
      }, "3. Execution Logs");

      const tabArtifactsBtn = h("button", {
        type: "button",
        class: `btn${activeInspectorTab === "artifacts" ? " btn-primary" : ""}`,
        style: "padding: 6px 12px; font-size: 0.85rem;"
      }, `4. Artifacts (${artifacts.length})`);

      const tabBar = h("div", { class: "row", style: "gap: 8px; border-bottom: 1px solid var(--border); padding-bottom: 8px;" },
        tabPipelineBtn,
        tabConfigBtn,
        tabLogsBtn,
        tabArtifactsBtn
      );

      const tabContentHost = h("div", { class: "stack", style: "gap: 12px; margin-top: 12px;" });

      function renderTabContent() {
        tabPipelineBtn.className = `btn${activeInspectorTab === "pipeline" ? " btn-primary" : ""}`;
        tabConfigBtn.className = `btn${activeInspectorTab === "config" ? " btn-primary" : ""}`;
        tabLogsBtn.className = `btn${activeInspectorTab === "logs" ? " btn-primary" : ""}`;
        tabArtifactsBtn.className = `btn${activeInspectorTab === "artifacts" ? " btn-primary" : ""}`;

        if (activeInspectorTab === "pipeline") {
          if (stages.length === 0) {
            replace(tabContentHost, emptyState({
              iconName: "inbox",
              title: "Pipeline Lifecycle Pending",
              message: "No stage execution events recorded for this run yet."
            }));
            return;
          }
          const stageList = stages.map(st =>
            h("div", {
              class: "row-between",
              style: "padding: 10px 14px; background: var(--bg-hover); border-radius: var(--radius-sm); border-left: 3px solid var(--border);"
            },
              h("div", { style: "max-width: 80%;" },
                h("div", { class: "row", style: "gap: 8px; align-items: center;" },
                  h("strong", { style: "font-size: 0.9rem;" }, st.name),
                  h("span", { class: "mono small muted" }, `[${st.id}]`)
                ),
                h("div", { class: "small muted", style: "margin-top: 2px;" }, st.purpose),
                st.result ? h("div", { class: "small mono text-teal", style: "margin-top: 4px;" }, `Result: ${st.result}`) : null
              ),
              badge(st.status)
            )
          );
          replace(tabContentHost, h("div", { class: "stack", style: "gap: 8px;" }, ...stageList));

        } else if (activeInspectorTab === "config") {
          const configRows = [
            ["Job Identifier", job.job_id],
            ["Base Model Architecture", job.base_model_name || "JackFram/llama-68m"],
            ["Dataset Name / Type", job.dataset_name || job.dataset_type || "Standard Dataset"],
            ["Training Epochs", String(job.epochs || 1)],
            ["Sample Subset Size", fmtNumber(job.subset_size || job.num_records || 0, 0)],
            ["Differential Privacy", job.dp_enabled ? "Enabled (Opacus DP-LoRA)" : "Disabled (Standard LoRA)"],
            ["Target Epsilon (ε)", job.target_epsilon ? String(job.target_epsilon) : "Not Configured"],
            ["Target Delta (δ)", job.target_delta ? String(job.target_delta) : "1e-05"],
            ["Noise Multiplier", job.noise_multiplier ? String(job.noise_multiplier) : "Not Configured"],
            ["Max Clipping Norm", job.max_grad_norm ? String(job.max_grad_norm) : "1.0"],
            ["LoRA Rank (r)", String(job.lora_r || 16)],
            ["LoRA Alpha (α)", String(job.lora_alpha || 32)],
            ["LoRA Dropout", String(job.lora_dropout || 0.05)]
          ];

          const cfgGrid = h("div", { class: "stack", style: "gap: 8px; font-size: 0.85rem;" },
            ...configRows.map(([k, v]) =>
              h("div", { class: "row-between", style: "padding: 6px 0; border-bottom: 1px solid var(--border);" },
                h("span", { class: "muted" }, k),
                h("span", { class: "mono" }, v)
              )
            )
          );
          replace(tabContentHost, card({ title: "Hyperparameters & Cryptographic Parameters", body: cfgGrid }));

        } else if (activeInspectorTab === "logs") {
          const logBox = h("pre", {
            class: "mono",
            style: "background: var(--bg-hover); padding: 14px; border-radius: var(--radius-sm); max-height: 380px; overflow-y: auto; font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap; word-break: break-all;"
          }, logs);

          const copyBtn = button({
            label: "Copy Logs",
            iconName: "copy",
            small: true,
            onClick: () => {
              navigator.clipboard.writeText(logs);
              notifySuccess("Copied", "Execution logs copied to clipboard.");
            }
          });

          const refreshBtn = button({
            label: "Refresh Logs",
            iconName: "refresh",
            small: true,
            onClick: async () => {
              const res = await endpoints.jobLogs(job.job_id).catch(() => null);
              if (res?.logs) {
                logBox.textContent = res.logs;
                notifySuccess("Refreshed", "Logs updated.");
              }
            }
          });

          replace(tabContentHost, h("div", { class: "stack", style: "gap: 8px;" },
            h("div", { class: "row-between" },
              h("span", { class: "small muted" }, "Tailing last 200 lines from training.log"),
              h("div", { class: "row", style: "gap: 6px;" }, copyBtn, refreshBtn)
            ),
            logBox
          ));

        } else if (activeInspectorTab === "artifacts") {
          if (artifacts.length === 0) {
            replace(tabContentHost, emptyState({
              iconName: "packaging",
              title: "No Cryptographic Artifacts",
              message: "Package artifacts (adapter.enc, adapter.sig, manifest) have not been compiled yet."
            }));
            return;
          }

          const artRows = artifacts.map(a =>
            h("tr", null,
              h("td", null, h("strong", { class: "mono text-teal" }, a.name)),
              h("td", { class: "mono small" }, fmtBytes(a.size_bytes)),
              h("td", null,
                button({
                  label: "Download",
                  iconName: "download",
                  small: true,
                  kind: "secondary",
                  onClick: async () => {
                    try {
                      await downloadAuthenticated(a.download_url, a.name);
                      notifySuccess("Download Started", `Downloading ${a.name}`);
                    } catch (err) {
                      notifyError(err, "Download Failed");
                    }
                  }
                })
              )
            )
          );

          const artTable = h("table", { class: "table", style: "width: 100%; font-size: 0.85rem;" },
            h("thead", null,
              h("tr", null,
                h("th", null, "Artifact File"),
                h("th", null, "Size"),
                h("th", null, "Action")
              )
            ),
            h("tbody", null, ...artRows)
          );

          replace(tabContentHost, card({
            title: "Generated Cryptographic Artifacts Inventory",
            sub: "Allow-listed package files verified and downloadable from the secure job vault",
            body: artTable
          }));
        }
      }

      tabPipelineBtn.addEventListener("click", () => { activeInspectorTab = "pipeline"; renderTabContent(); });
      tabConfigBtn.addEventListener("click", () => { activeInspectorTab = "config"; renderTabContent(); });
      tabLogsBtn.addEventListener("click", () => { activeInspectorTab = "logs"; renderTabContent(); });
      tabArtifactsBtn.addEventListener("click", () => { activeInspectorTab = "artifacts"; renderTabContent(); });

      // Quick Nav Buttons
      const quickActions = h("div", { class: "row", style: "gap: 10px; margin-top: 16px; flex-wrap: wrap; border-top: 1px solid var(--border); padding-top: 14px;" },
        button({
          label: "Adapter Screening",
          iconName: "screening",
          kind: "primary",
          onClick: () => navigate("/screening", { job_id: job.job_id })
        }),
        button({
          label: "Secure Packaging",
          iconName: "packaging",
          kind: "secondary",
          onClick: () => navigate("/packaging", { job_id: job.job_id })
        }),
        button({
          label: "Deployment Verification",
          iconName: "deployment",
          kind: "secondary",
          onClick: () => navigate("/deployment", { job_id: job.job_id })
        }),
        button({
          label: "Evaluation & Reports",
          iconName: "evaluation",
          kind: "secondary",
          onClick: () => navigate("/evaluation", { job_id: job.job_id })
        })
      );

      const drawerContent = h("div", { class: "stack", style: "gap: 14px;" },
        drawerHeader,
        failurePanel,
        tabBar,
        tabContentHost,
        quickActions
      );

      renderTabContent();
      replace(detailDrawer, drawerContent);
    }

    searchInput.addEventListener("input", updateTable);
    statusFilter.addEventListener("change", updateTable);
    sortFilter.addEventListener("change", updateTable);

    async function loadJobs() {
      try {
        const res = await endpoints.jobs({ signal });
        if (signal.aborted) return;

        allJobs = res.jobs || [];
        updateTable();

        if (initialJobId) {
          const match = allJobs.find(j => j.job_id === initialJobId);
          if (match) selectJob(match);
        }
      } catch (err) {
        if (isAbort(err)) return;
        replace(tableContainer, errorState(err, loadJobs));
      }
    }

    loadJobs();
    return () => {};
  }
};
