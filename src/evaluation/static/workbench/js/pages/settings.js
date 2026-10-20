import { h, icon, clear, replace } from "../core/dom.js";
import { getTheme, setTheme, isSystemDark } from "../core/theme.js";
import { getToken, setToken, clearToken, maskToken, tokenStorageKind, endpoints, userMessage, isAbort } from "../core/api.js";
import { pageHead, card, badge, field, button, toast } from "../core/ui.js";

export default {
  async mount(root, { navigate, signal }) {
    const header = pageHead({
      title: "Settings & System Configuration",
      description: "Manage visual theme preferences, API authentication credentials, and inspect host runtime environment posture."
    });

    // --- Section 1: Visual Theme ---
    const currentTheme = getTheme();
    const darkRadio = h("input", { type: "radio", name: "workbench-theme", value: "dark", checked: currentTheme === "dark" });
    const lightRadio = h("input", { type: "radio", name: "workbench-theme", value: "light", checked: currentTheme === "light" });

    function onThemeChange(e) {
      const chosen = e.target.value;
      setTheme(chosen);
      toast({ title: "Theme Updated", message: `Workbench switched to ${chosen} mode.`, kind: "info" });
    }

    darkRadio.addEventListener("change", onThemeChange);
    lightRadio.addEventListener("change", onThemeChange);

    const themeCard = card({
      title: "Appearance & Theme",
      sub: "Select enterprise dark navy or high-contrast clean light mode. Persists across sessions.",
      body: h("div", { class: "grid grid-2", style: "gap: 16px;" },
        h("label", {
          class: "card",
          style: "cursor: pointer; padding: 16px; border: 2px solid " + (currentTheme === "dark" ? "var(--accent)" : "var(--border)") + "; display: flex; align-items: center; gap: 12px;"
        },
          darkRadio,
          h("div", null,
            h("strong", null, "Enterprise Dark Navy"),
            h("div", { class: "small muted" }, "Optimized for low-light security operations center viewing")
          )
        ),
        h("label", {
          class: "card",
          style: "cursor: pointer; padding: 16px; border: 2px solid " + (currentTheme === "light" ? "var(--accent)" : "var(--border)") + "; display: flex; align-items: center; gap: 12px;"
        },
          lightRadio,
          h("div", null,
            h("strong", null, "High-Contrast Light"),
            h("div", { class: "small muted" }, "Crisp neutral surfaces with high legibility for reports")
          )
        )
      )
    });

    // --- Section 2: API Authentication ---
    const activeToken = getToken();
    const storageKind = tokenStorageKind();

    const tokenInput = h("input", {
      type: "password",
      class: "input mono",
      placeholder: activeToken ? maskToken(activeToken) : "Paste Bearer token here...",
      style: "width: 100%;"
    });

    const persistCheckbox = h("input", { type: "checkbox", id: "token-persist-chk", checked: storageKind === "local" });

    const tokenStatusText = h("div", { class: "small muted", style: "margin-top: 6px;" },
      activeToken
        ? `Active token stored in ${storageKind === "local" ? "Local Storage (Persistent)" : "Session Storage (Tab Only)"}: ${maskToken(activeToken)}`
        : "No token set. Endpoints requiring authentication will return HTTP 401."
    );

    const saveTokenBtn = button({
      label: "Save API Token",
      iconName: "check",
      kind: "primary",
      onClick: () => {
        const val = tokenInput.value.trim();
        if (!val) {
          toast({ title: "Empty Token", message: "Enter a valid token string.", kind: "warn" });
          return;
        }
        setToken(val, { persist: persistCheckbox.checked });
        tokenInput.value = "";
        tokenInput.placeholder = maskToken(val);
        tokenStatusText.textContent = `Active token stored in ${persistCheckbox.checked ? "Local Storage" : "Session Storage"}: ${maskToken(val)}`;
        toast({ title: "Token Saved", message: "API authentication credentials updated.", kind: "success" });
      }
    });

    const clearTokenBtn = button({
      label: "Clear Token",
      iconName: "trash",
      kind: "ghost",
      onClick: () => {
        clearToken();
        tokenInput.value = "";
        tokenInput.placeholder = "Paste Bearer token here...";
        tokenStatusText.textContent = "No token set. Requests will be sent without Bearer headers.";
        toast({ title: "Token Cleared", message: "Removed authentication credentials.", kind: "info" });
      }
    });

    const testHealthBtn = button({
      label: "Probe Server Health",
      iconName: "refresh",
      kind: "secondary",
      onClick: async () => {
        testHealthBtn.disabled = true;
        testHealthBtn.innerHTML = `<span class="busy-spinner" style="width: 14px; height: 14px; margin-right: 6px;"></span> Probing...`;
        try {
          const res = await endpoints.health({ signal });
          toast({
            title: "Server Reachable",
            message: `Status: ${res.status || "OK"} · Environment: Active`,
            kind: "success"
          });
        } catch (err) {
          toast({
            title: "Probe Failed",
            message: userMessage(err),
            kind: "error"
          });
        } finally {
          testHealthBtn.disabled = false;
          testHealthBtn.innerHTML = `${icon("refresh").outerHTML} Probe Server Health`;
        }
      }
    });

    const authCard = card({
      title: "API Authentication (Bearer Token)",
      sub: "Bearer token sent in HTTP Authorization headers. Compatible with existing backend security gates.",
      body: h("div", { class: "stack", style: "gap: 16px;" },
        h("div", { class: "stack", style: "gap: 8px;" },
          tokenInput,
          tokenStatusText
        ),
        h("label", { class: "row", style: "gap: 8px; cursor: pointer;" },
          persistCheckbox,
          h("span", { class: "small" }, "Persist in Local Storage across browser restarts (Uncheck to keep in session only)")
        ),
        h("div", { class: "row-between", style: "margin-top: 8px;" },
          h("div", { class: "row", style: "gap: 8px;" },
            saveTokenBtn,
            clearTokenBtn
          ),
          testHealthBtn
        )
      )
    });

    // --- Section 3: Runtime Environment & Hardware Disclaimer ---
    const envDetailsBox = h("div", null);
    const envCard = card({
      title: "Host Runtime & Cryptographic Attestation",
      sub: "Platform capabilities, device identity, and software attestation model",
      body: envDetailsBox
    });

    async function loadEnv() {
      try {
        const p4Res = await endpoints.phase4Status({ signal }).catch(() => ({}));
        const rows = [
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Target Base Architecture"),
            h("span", { class: "mono" }, p4Res.base_model_name || "JackFram/llama-68m")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Device Salt Prefix"),
            h("span", { class: "mono text-teal" }, p4Res.salt_masked || "Configured")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0; border-bottom: 1px solid var(--border);" },
            h("span", { class: "muted" }, "Hardware Attestation Model"),
            h("span", { class: "mono" }, "Software-Derived HKDF Binding")
          ),
          h("div", { class: "row-between", style: "padding: 8px 0;" },
            h("span", { class: "muted" }, "Security Disclosure"),
            h("span", { class: "small muted", style: "max-width: 320px; text-align: right;" },
              "Device binding is rooted in software-derived platform fingerprints and node salts. No hardware TPM 2.0 or secure enclave is asserted."
            )
          )
        ];
        replace(envDetailsBox, h("div", { class: "stack", style: "gap: 4px;" }, ...rows));
      } catch (err) {
        if (!signal.aborted) {
          replace(envDetailsBox, h("div", { class: "small muted" }, "Could not retrieve runtime environment data."));
        }
      }
    }

    loadEnv();

    const layout = h("div", { class: "stack" },
      header,
      themeCard,
      authCard,
      envCard
    );

    replace(root, layout);
    return () => {};
  }
};
