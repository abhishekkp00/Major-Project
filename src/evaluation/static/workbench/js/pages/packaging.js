import { h, icon, clear, replace } from "../core/dom.js";
import { endpoints, downloadAuthenticated, userMessage, isAbort } from "../core/api.js";
import { fmtNumber, fmtBytes, fmtDate } from "../core/format.js";
import { pageHead, card, kpi, badge, button, emptyState, errorState, dataTable, skeletonKpis, skeletonCards, skeletonLines, toast } from "../core/ui.js";

function copyToClipboard(text, label = "Value") {
  if (!text) return;
  if (navigator?.clipboard?.writeText) {
    navigator.clipboard.writeText(text)
      .then(() => toast({ kind: "info", title: "Copied to Clipboard", message: `${label} copied.` }))
      .catch(() => fallbackCopy(text, label));
  } else {
    fallbackCopy(text, label);
  }
}

function fallbackCopy(text, label) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    toast({ kind: "info", title: "Copied to Clipboard", message: `${label} copied.` });
  } catch (_) {
    toast({ kind: "warning", title: "Copy Failed", message: "Please copy manually." });
  }
  document.body.removeChild(ta);
}

function hashSnippet(hash, label, fullText = null) {
  if (!hash) return h("span", { class: "muted" }, "None recorded");
  const display = hash.length > 20 ? `${hash.slice(0, 10)}…${hash.slice(-8)}` : hash;
  const target = fullText || hash;
  return h("div", { class: "row", style: "gap: 6px; align-items: center; display: inline-flex;" },
    h("code", { class: "mono small", title: target, style: "font-size: 0.82rem; background: var(--bg-hover); padding: 2px 6px; border-radius: 4px;" }, display),
    h("button", {
      class: "btn btn-sm",
      type: "button",
      title: `Copy full ${label}`,
      style: "padding: 2px 6px; font-size: 0.75rem;",
      onclick: () => copyToClipboard(target, label)
    }, icon("copy", { cls: "" }), "Copy")
  );
}

export default {
  async mount(root, { navigate, params, signal }) {
    let currentJobId = params?.job_id || sessionStorage.getItem("securelora_active_job") || null;

    const header = pageHead({
      title: "Cryptographic Packaging & Provenance Manifest",
      description: "Technical package-inspection workspace: SHA-256 digest sealing, RSA-2048-PSS signature verification, HKDF key derivation, and AES-256-GCM symmetric encryption.",
      actions: [
        button({ label: "Adapter Screening", iconName: "screening", onClick: () => navigate("/screening") }),
        button({ label: "Deployment Verification", iconName: "deployment", kind: "primary", onClick: () => navigate("/deployment") })
      ]
    });

    const selectorContainer = h("div", { class: "row-between", style: "background: var(--bg-surface); padding: 12px 16px; border: 1px solid var(--border); border-radius: var(--radius-md); flex-wrap: wrap; gap: 12px;" },
      h("div", { class: "row", style: "gap: 12px; align-items: center;" },
        h("label", { style: "font-weight: 600; font-size: 0.9rem;" }, "Audit Target Run:"),
        h("select", { id: "packaging-job-select", class: "select mono", style: "min-width: 280px;" },
          h("option", { value: "" }, "Loading available pipeline runs...")
        )
      ),
      h("div", { class: "row", style: "gap: 10px; align-items: center;" },
        h("div", { id: "packaging-target-meta", class: "small muted mono" }, ""),
        h("div", { id: "packaging-status-badge" }, badge("SELECT RUN"))
      )
    );

    const kpiContainer = h("div", null, skeletonKpis(4));
    const sequenceContainer = h("div", null, skeletonLines(3));
    const inventoryContainer = h("div", null, skeletonLines(4));
    const integrityContainer = h("div", null, skeletonLines(4));
    const cryptoDetailsContainer = h("div", null, skeletonLines(4));
    const disclaimersContainer = h("div", null, skeletonLines(2));

    const sequenceCard = card({
      title: "Packaging Lifecycle Sequence",
      sub: "Four-stage cryptographic pipeline: Adapter Source Inspection → HKDF & AES-GCM Encryption → SHA-256 Digest Sealing → RSA-PSS Signing",
      body: sequenceContainer
    });

    const inventoryCard = card({
      title: "Package File Inventory & Allow-Listed Artifacts",
      sub: "Cryptographic artifacts produced in the protected package directory. Authenticated downloads enforce default-deny security filtering.",
      body: inventoryContainer
    });

    const integrityCard = card({
      title: "Integrity Sealing & Provenance Signature Verification",
      sub: "Live cryptographic verification of ciphertext SHA-256 checksums and RSA-2048-PSS canonical manifest signatures",
      body: integrityContainer
    });

    const cryptoCard = card({
      title: "Key Derivation (HKDF-SHA256) & Symmetric Envelope (AES-256-GCM)",
      sub: "Cryptographic parameters, nonce metadata, and device binding context (RFC 5869 & NIST SP 800-38D)",
      body: cryptoDetailsContainer
    });

    const disclaimersCard = card({
      title: "Security Architecture & Attestation Disclaimers",
      sub: "Operational boundaries, software-derived attestation limitations, and transient plaintext memory considerations",
      body: disclaimersContainer
    });

    const layout = h("div", { class: "stack", style: "gap: 20px;" },
      header,
      selectorContainer,
      kpiContainer,
      sequenceCard,
      inventoryCard,
      integrityCard,
      cryptoCard,
      disclaimersCard
    );

    replace(root, layout);

    const jobSelect = selectorContainer.querySelector("#packaging-job-select");
    const statusBadge = selectorContainer.querySelector("#packaging-status-badge");
    const targetMeta = selectorContainer.querySelector("#packaging-target-meta");

    jobSelect.addEventListener("change", () => {
      currentJobId = jobSelect.value;
      if (currentJobId) {
        sessionStorage.setItem("securelora_active_job", currentJobId);
        loadPackaging(currentJobId);
      }
    });

    async function initPicker() {
      try {
        const res = await endpoints.jobs({ signal });
        if (signal.aborted) return;
        const jobs = res.jobs || [];

        jobSelect.innerHTML = "";
        if (jobs.length === 0) {
          jobSelect.innerHTML = `<option value="">No pipeline jobs found</option>`;
          replace(kpiContainer, emptyState({
            iconName: "packaging",
            title: "No Pipeline Runs Found",
            message: "Execute a pipeline run to produce cryptographic package artifacts.",
            action: button({ label: "Launch Training", iconName: "play", kind: "primary", onClick: () => navigate("/training") })
          }));
          return;
        }

        jobs.forEach(j => {
          const opt = h("option", {
            value: j.job_id,
            selected: currentJobId ? j.job_id === currentJobId : false
          }, `${j.job_id} (${j.dataset_name || "Standard"} — ${j.status})`);
          jobSelect.appendChild(opt);
        });

        if (!currentJobId || !jobs.some(j => j.job_id === currentJobId)) {
          currentJobId = jobs[0].job_id;
          jobSelect.value = currentJobId;
        }

        loadPackaging(currentJobId);
      } catch (err) {
        if (isAbort(err)) return;
        jobSelect.innerHTML = `<option value="">Failed to load jobs</option>`;
      }
    }

    async function loadPackaging(jobId) {
      if (!jobId) return;

      replace(kpiContainer, skeletonKpis(4));
      replace(sequenceContainer, skeletonLines(3));
      replace(inventoryContainer, skeletonLines(4));
      replace(integrityContainer, skeletonLines(4));
      replace(cryptoDetailsContainer, skeletonLines(4));
      replace(disclaimersContainer, skeletonLines(2));

      try {
        const [pkgRes, jobRes] = await Promise.all([
          endpoints.jobPackaging(jobId, { signal }).catch(err => ({ success: false, error: userMessage(err) })),
          endpoints.job(jobId, { signal }).catch(() => null)
        ]);

        if (signal.aborted) return;

        if (!pkgRes || !pkgRes.success) {
          replace(statusBadge, badge("FAILED"));
          replace(kpiContainer, errorState(pkgRes?.error || "Failed to load packaging details", () => loadPackaging(jobId)));
          return;
        }

        if (pkgRes.available === false) {
          replace(statusBadge, badge("NOT_PACKAGED"));
          targetMeta.textContent = `Job: ${jobId}`;
          replace(kpiContainer, emptyState({
            iconName: "packaging",
            title: "Packaging Artifacts Not Available",
            message: pkgRes.reason || "Cryptographic packaging has not been completed for this run."
          }));
          replace(sequenceContainer, h("p", { class: "muted" }, "No packaging sequence data recorded yet."));
          replace(inventoryContainer, h("p", { class: "muted" }, "No package files found in workspace."));
          replace(integrityContainer, h("p", { class: "muted" }, "Integrity digests not calculated."));
          replace(cryptoDetailsContainer, h("p", { class: "muted" }, "Cipher metadata unavailable."));
          replace(disclaimersContainer, h("p", { class: "muted" }, "Attestation boundaries will display once packaging commences."));
          return;
        }

        const data = pkgRes;
        const job = jobRes?.job || {};

        // Top badges & metadata
        targetMeta.textContent = `Package ID: ${data.package_id ? data.package_id.slice(0, 8) + '…' : 'Pending'} | Adapter: ${data.adapter_id}`;
        replace(statusBadge, badge(data.package_status || "PACKAGED"));

        // 1. Top KPI Row
        const sigVerified = data.signature?.verified === true;
        const digestMatch = data.integrity?.digest_matches_recorded === true;
        const isEncrypted = Boolean(data.encryption?.iv_hex && data.encryption?.tag_hex);

        replace(kpiContainer, h("div", { class: "kpis" },
          kpi({
            label: "Package State",
            value: data.package_status || "UNKNOWN",
            text: true,
            foot: `Version: ${data.package_version || "1.0.0"}`
          }),
          kpi({
            label: "Ciphertext Integrity",
            value: digestMatch ? "SHA-256 VERIFIED" : (data.integrity?.recorded_digest ? "MISMATCH" : "UNRECORDED"),
            text: true,
            foot: digestMatch ? "Computed digest matches adapter.hash" : "Cryptographic checksum mismatch"
          }),
          kpi({
            label: "Digital Signature",
            value: sigVerified ? "RSA-PSS (2048-BIT)" : (data.signature?.signature_present ? "VERIFICATION FAILED" : "UNSIGNED"),
            text: true,
            foot: sigVerified ? "Authenticity verified over manifest" : (data.signature?.verification_error || "Signature not verified")
          }),
          kpi({
            label: "Symmetric Cipher",
            value: isEncrypted ? "AES-256-GCM" : "UNENCRYPTED",
            text: true,
            foot: isEncrypted ? `IV (12B) + Tag (16B) sealed` : "Envelope uninitialized"
          })
        ));

        // 2. Lifecycle Sequence Stepper
        const stages = data.sequence_stages || {};
        const step1 = stages.step_1_inspect || { status: "UNKNOWN", name: "Adapter Source Inspection" };
        const step2 = stages.step_2_encrypt || { status: "PENDING", name: "HKDF Key Derivation & AES-GCM" };
        const step3 = stages.step_3_digest || { status: "PENDING", name: "SHA-256 Digest Sealing" };
        const step4 = stages.step_4_sign || { status: "PENDING", name: "RSA-PSS Manifest Signature" };

        const sequenceList = h("div", { class: "grid grid-4", style: "gap: 12px;" },
          renderSequenceStep("Step 1", "Inspect Adapter", step1.status, "Statistical norms & pre-screening", step1.screening_outcome ? `Outcome: ${step1.screening_outcome}` : null),
          renderSequenceStep("Step 2", "Encrypt Package", step2.status, "AES-256-GCM + HKDF-SHA256", step2.latency_ms ? `${fmtNumber(step2.latency_ms, 1)} ms latency` : null),
          renderSequenceStep("Step 3", "Generate Digest", step3.status, "SHA-256 over adapter.enc", data.integrity?.ciphertext_digest ? "Checksum generated" : null),
          renderSequenceStep("Step 4", "Sign Package", step4.status, "RSA-2048-PSS manifest seal", step4.latency_ms ? `${fmtNumber(step4.latency_ms, 1)} ms signing` : null)
        );
        replace(sequenceContainer, sequenceList);

        // 3. File Inventory Table
        const inventory = data.inventory || [];
        if (inventory.length === 0) {
          replace(inventoryContainer, emptyState({
            iconName: "inbox",
            title: "No Package Files Generated",
            message: "No allow-listed package artifacts exist in the protected workspace."
          }));
        } else {
          const invTable = dataTable({
            columns: [
              { key: "name", label: "File Name", render: a => h("strong", { class: "mono" }, a.name) },
              { key: "role", label: "Cryptographic Role", render: a => h("span", { class: "small" }, a.role) },
              { key: "size_bytes", label: "Size", num: true, render: a => fmtBytes(a.size_bytes) },
              {
                key: "sha256_digest",
                label: "SHA-256 Digest",
                render: a => hashSnippet(a.sha256_digest, `${a.name} SHA-256`)
              },
              {
                key: "actions",
                label: "Download",
                render: a => button({
                  label: "Download",
                  iconName: "download",
                  small: true,
                  kind: "secondary",
                  onClick: async () => {
                    try {
                      toast({ title: "Downloading", message: `Fetching ${a.name}...`, kind: "info" });
                      await downloadAuthenticated(a.download_url, a.name);
                      toast({ title: "Download Complete", message: `${a.name} downloaded securely.`, kind: "success" });
                    } catch (err) {
                      toast({ title: "Download Failed", message: userMessage(err), kind: "error" });
                    }
                  }
                })
              }
            ],
            rows: inventory
          });

          const inventoryStack = h("div", { class: "stack", style: "gap: 12px;" },
            invTable,
            h("div", { class: "notice notice-info", style: "padding: 8px 12px; font-size: 0.85rem;" },
              h("strong", null, "Security Invariant: "),
              "Private signing keys (e.g. dev_private.pem) and unmasked symmetric decryption keys are strictly excluded from package archives and download endpoints via path-traversal resistant allow-listing."
            )
          );
          replace(inventoryContainer, inventoryStack);
        }

        // 4. Cryptographic Integrity & Signature Verification
        const int = data.integrity || {};
        const sig = data.signature || {};

        const integrityGrid = h("div", { class: "grid grid-2", style: "gap: 16px;" },
          // Panel A: SHA-256 Checksums
          h("div", { class: "card", style: "background: var(--bg-hover); padding: 16px;" },
            h("div", { class: "row-between", style: "margin-bottom: 12px;" },
              h("h4", { style: "margin: 0; font-size: 0.92rem;" }, "SHA-256 Ciphertext Integrity"),
              badge(int.digest_matches_recorded ? "PASSED" : (int.recorded_digest ? "FAILED" : "PENDING"), int.digest_matches_recorded ? "CHECKSUM MATCH" : (int.recorded_digest ? "MISMATCH" : "PENDING"))
            ),
            h("div", { class: "stack", style: "gap: 8px; font-size: 0.85rem;" },
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Algorithm:"),
                h("span", { class: "mono" }, int.digest_algorithm || "SHA-256 (FIPS 180-4)")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Calculated Hash (adapter.enc):"),
                hashSnippet(int.ciphertext_digest, "Ciphertext SHA-256")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Recorded Hash (adapter.hash):"),
                hashSnippet(int.recorded_digest, "Recorded SHA-256")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Manifest Hash (encrypted_adapter_digest):"),
                hashSnippet(int.manifest_digest, "Manifest Digest")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Constant-Time Verification:"),
                h("span", { class: int.digest_matches_recorded ? "mono text-success" : "mono text-danger", style: "font-weight: 600;" },
                  int.digest_matches_recorded ? "PASS (hmac.compare_digest)" : "FAIL / INCOMPLETE"
                )
              )
            )
          ),

          // Panel B: RSA-PSS Signature
          h("div", { class: "card", style: "background: var(--bg-hover); padding: 16px;" },
            h("div", { class: "row-between", style: "margin-bottom: 12px;" },
              h("h4", { style: "margin: 0; font-size: 0.92rem;" }, "RSA-2048-PSS Signature Verification"),
              badge(sig.verified ? "PASSED" : (sig.signature_present ? "FAILED" : "PENDING"), sig.verified ? "SIGNATURE VALID" : (sig.signature_present ? "REJECTED" : "UNSIGNED"))
            ),
            h("div", { class: "stack", style: "gap: 8px; font-size: 0.85rem;" },
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Signature Algorithm:"),
                h("span", { class: "mono" }, sig.algorithm || "RSA-PSS with SHA-256 (MGF1)")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Canonical Manifest Digest:"),
                hashSnippet(sig.canonical_digest, "Canonical Manifest Digest")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Public Key Fingerprint (SHA-256):"),
                hashSnippet(sig.public_key_fingerprint, "Public Key Fingerprint")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Verification Status:"),
                h("span", { class: sig.verified ? "mono text-success" : "mono text-danger", style: "font-weight: 600;" },
                  sig.verified ? "CRYPTOGRAPHICALLY AUTHENTIC" : (sig.verification_error || "NOT VERIFIED")
                )
              )
            ),
            sig.public_key_pem ? h("details", { style: "margin-top: 12px; font-size: 0.8rem;" },
              h("summary", { style: "cursor: pointer; font-weight: 600; color: var(--primary);" }, "Inspect Public Key (X.509 PEM)"),
              h("div", { style: "position: relative; margin-top: 6px;" },
                h("pre", { class: "mono small", style: "background: var(--bg-surface); padding: 10px; border-radius: var(--radius-sm); max-height: 140px; overflow-y: auto; white-space: pre-wrap;" }, sig.public_key_pem),
                h("button", {
                  class: "btn btn-sm",
                  type: "button",
                  style: "position: absolute; top: 6px; right: 6px; padding: 2px 6px; font-size: 0.75rem;",
                  onclick: () => copyToClipboard(sig.public_key_pem, "Public Key PEM")
                }, icon("copy", { cls: "" }), "Copy PEM")
              )
            ) : null
          )
        );
        replace(integrityContainer, integrityGrid);

        // 5. Key Derivation & AES-256-GCM Envelope Details
        const kd = data.key_derivation || {};
        const enc = data.encryption || {};
        const man = data.manifest || {};

        const cryptoGrid = h("div", { class: "grid grid-2", style: "gap: 16px;" },
          // Panel A: HKDF Key Derivation
          h("div", { class: "card", style: "background: var(--bg-hover); padding: 16px;" },
            h("h4", { style: "margin: 0 0 12px 0; font-size: 0.92rem;" }, "HKDF-SHA256 Key Derivation (RFC 5869)"),
            h("div", { class: "stack", style: "gap: 8px; font-size: 0.85rem;" },
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "KDF Scheme:"),
                h("span", { class: "mono" }, kd.kdf_version || "hkdf-sha256-v2")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Per-Package Salt (Hex):"),
                hashSnippet(kd.salt_hex, "HKDF Salt")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Salt Length:"),
                h("span", { class: "mono" }, `${kd.salt_length_bytes || 32} bytes (CSPRNG random)`)
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "IKM Entropy Source:"),
                h("span", { class: "mono" }, kd.ikm_source || "Device Secret")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Context Info Binding:"),
                h("span", { class: "mono small", title: kd.info_context }, kd.info_context || "securelora-adapter-v2|<fp>")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Target Host Fingerprint Ref:"),
                hashSnippet(kd.device_fingerprint_ref, "Host Fingerprint Ref")
              ),
              h("p", { class: "small muted", style: "line-height: 1.4; margin: 8px 0 0 0;" },
                kd.zeroization_guarantee || "Derived symmetric keys reside exclusively in transient execution memory and are scrubbed immediately."
              )
            )
          ),

          // Panel B: AES-256-GCM Envelope
          h("div", { class: "card", style: "background: var(--bg-hover); padding: 16px;" },
            h("h4", { style: "margin: 0 0 12px 0; font-size: 0.92rem;" }, "AES-256-GCM Envelope Parameters"),
            h("div", { class: "stack", style: "gap: 8px; font-size: 0.85rem;" },
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Symmetric Cipher:"),
                h("span", { class: "mono" }, enc.cipher || "AES-256-GCM (NIST SP 800-38D)")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Initialization Vector / Nonce (12B):"),
                hashSnippet(enc.iv_hex, "IV Nonce")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "GMAC Authentication Tag (16B):"),
                hashSnippet(enc.tag_hex, "Authentication Tag")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Ciphertext Payload Size:"),
                h("span", { class: "mono" }, fmtBytes(enc.ciphertext_size_bytes || 0))
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Anti-Replay Sequence #:"),
                h("span", { class: "mono" }, man.sequence_number !== undefined ? `#${man.sequence_number}` : "N/A")
              ),
              h("div", { class: "row-between" },
                h("span", { class: "muted" }, "Package UUID:"),
                h("span", { class: "mono small" }, man.package_id || "Unassigned")
              )
            ),
            man.present ? h("details", { style: "margin-top: 12px; font-size: 0.8rem;" },
              h("summary", { style: "cursor: pointer; font-weight: 600; color: var(--primary);" }, "Inspect Package Manifest (package_manifest.json)"),
              h("div", { style: "position: relative; margin-top: 6px;" },
                h("pre", { class: "mono small", style: "background: var(--bg-surface); padding: 10px; border-radius: var(--radius-sm); max-height: 140px; overflow-y: auto; white-space: pre-wrap;" }, JSON.stringify(man, null, 2)),
                h("button", {
                  class: "btn btn-sm",
                  type: "button",
                  style: "position: absolute; top: 6px; right: 6px; padding: 2px 6px; font-size: 0.75rem;",
                  onclick: () => copyToClipboard(JSON.stringify(man, null, 2), "Package Manifest")
                }, icon("copy", { cls: "" }), "Copy Manifest")
              )
            ) : null
          )
        );
        replace(cryptoDetailsContainer, cryptoGrid);

        // 6. Security Architecture & Attestation Disclaimers
        const disc = data.security_disclaimers || {};
        const disclaimersGrid = h("div", { class: "grid grid-2", style: "gap: 16px;" },
          h("div", { class: "notice notice-warning", style: "padding: 12px 16px;" },
            h("div", { style: "font-weight: 600; margin-bottom: 4px;" }, "Attestation Model Boundary: Software vs Hardware"),
            h("p", { class: "small", style: "margin: 0; line-height: 1.45;" },
              disc.hardware_attestation || "Software-derived device binding relies on OS-reported telemetry and local secrets. It does not provide hardware-isolated key protection (e.g., TPM 2.0, ARM TrustZone, or HSM) and cannot defend against a compromised kernel."
            )
          ),
          h("div", { class: "notice notice-info", style: "padding: 12px 16px;" },
            h("div", { style: "font-weight: 600; margin-bottom: 4px;" }, "Transient Plaintext Storage Consideration"),
            h("p", { class: "small", style: "margin: 0; line-height: 1.45;" },
              disc.transient_plaintext_storage || "During adapter validation and PEFT initialization, decrypted model tensors are buffered in restricted-permission ephemeral storage before mapping into torch memory. In production, mount temporary workspace directories on RAM-backed tmpfs with strict POSIX permissions (0700)."
            )
          )
        );
        replace(disclaimersContainer, disclaimersGrid);

      } catch (err) {
        if (isAbort(err)) return;
        replace(kpiContainer, errorState(err, () => loadPackaging(jobId)));
      }
    }

    function renderSequenceStep(stepNum, stepName, status, description, extra) {
      const isPassed = status === "PASSED";
      const isFailed = status === "FAILED";
      const isBypass = status === "BYPASS";

      const badgeColor = isPassed ? "success" : (isFailed ? "danger" : (isBypass ? "warning" : "neutral"));

      return h("div", { class: "card", style: `background: var(--bg-hover); padding: 12px; border-left: 3px solid var(--${badgeColor === "success" ? "accent-teal" : (badgeColor === "danger" ? "danger" : (badgeColor === "warning" ? "accent-amber" : "border"))});` },
        h("div", { class: "row-between", style: "margin-bottom: 6px;" },
          h("span", { class: "mono small text-muted", style: "font-weight: 600;" }, stepNum),
          badge(status)
        ),
        h("div", { style: "font-weight: 600; font-size: 0.88rem; margin-bottom: 4px;" }, stepName),
        h("div", { class: "small muted", style: "line-height: 1.35;" }, description),
        extra ? h("div", { class: "small mono text-muted", style: "margin-top: 6px; font-size: 0.78rem;" }, extra) : null
      );
    }

    initPicker();
    return () => {};
  }
};
