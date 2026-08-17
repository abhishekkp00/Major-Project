# SecureLoRA: Technical & Academic Research Specification

## 1. Executive Summary

**SecureLoRA** is an end-to-end research framework designed for privacy-preserving, cryptographically protected, and security-screened Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models.

The system addresses five fundamental threat vectors in open-source fine-tuning and third-party adapter deployment:
1.  **PII/PHI Leakage in Model Output**: Un-sanitized training records causing LLMs to memorize and generate sensitive information.
2.  **Untrusted Third-Party LoRA Adapters**: Malicious adapters containing trojans, backdoors, or data exfiltration triggers.
3.  **Adaptive Evasion Attacks**: Adversarial adapters crafted to bypass standard anomaly detectors by constraining spectral norms.
4.  **Unauthorized Environment Execution**: Deployment of proprietary fine-tuned adapters on unauthorized hardware or virtual machines.
5.  **Package Tampering & Replay Attacks**: Man-in-the-middle modifications or replay of expired deployment archives.

---

## 2. Threat Model & Security Posture

```
              ┌─────────────────────────────────────────────────────────────┐
              │                    UNTRUSTED ENVIRONMENT                    │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             │                     SECURELORA PIPELINE                       │
             └───────────────────────────────┬───────────────────────────────┘
                                             │
     [ Gate 1: Hybrid PII Engine ] ──────────┼──> Eliminates Sensitive Entities in RAM
                                             │
     [ Gate 2: DP-LoRA Privacy Guard ] ──────┼──> Enforces (ε, δ)-Differential Privacy
                                             │
     [ Gate 3: Joint Adapter Screening ] ────┼──> Rejects Trojan / Adaptive Evading Adapters
                                             │
     [ Gate 4: Hardware Binding Key (HKDF) ] ┼──> Restricts Execution to Authorized CPU/GPU
                                             │
     [ Gate 5: RSA-PSS Signature & Nonce ] ──┼──> Prevents Archive Tampering & Replays
                                             │
              ┌──────────────────────────────▼──────────────────────────────┐
              │             VERIFIED & DEPLOYED INFERENCE MODEL             │
              └─────────────────────────────────────────────────────────────┘
```

### 2.1 Adversarial Capabilities
The attacker is assumed to have full access to:
*   The open-weight base language model architecture and parameters.
*   The screening algorithm's objective function and scoring metric (Adaptive Attacker Model).
*   The deployment network channel (enabling tampering and replay attempts).

The attacker **does not** have access to:
*   The private RSA signing key ($\text{SK}_{\text{RSA}}$).
*   The host machine's hardware secret salt ($S_{\text{device}}$).
*   The internal RAM state of the deployment runtime engine.

---

## 3. Core Architectural Modules

### 3.1 Hybrid PII Redaction Engine (`src/phase1/`)
*   **Methodology**: Integrates RFC/ISO pattern matching (Luhn checksums, IBAN, SSN, Credit Card) with ML-based Named Entity Recognition (SpaCy / Presidio transformers).
*   **Zero-Disk-Leakage Architecture**: Redaction occurs within volatile RAM prior to tokenization. Unredacted text is not flushed to persistent storage or swap space during normal execution.

### 3.2 Differentially Private LoRA (`src/phase2/`)
*   **Privacy Model**: $(\epsilon, \delta)$-Differential Privacy implemented via Rényi Differential Privacy (RDP) accountant.
*   **Mechanism**: Per-sample gradient clipping ($\|g_i\|_2 \le C$) and Gaussian noise injection ($z \sim \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$) applied exclusively to LoRA trainable matrices ($A$ and $B$).
*   **Empirical Target**: $\epsilon \le 2.5, \delta = 10^{-5}$.

### 3.3 Comparative Adapter Screening System (`src/security/adapter_screening/`)
*   **Structural-Only Screen**: Computes singular value decomposition (SVD) on adapter weight matrices $W = B \times A$. Detects spectral rank anomalies and Frobenius norm deviations.
*   **Behavioral-Only Screen**: Measures activation shift and output perplexity divergence on standard probe inputs.
*   **Combined (SecureLoRA) Screen**: Jointly evaluates structural spectral bounds and behavioral activation subspace projections. Achieved a **1.0000 F1 score ($\tau=0.35$)** on the evaluated multi-seed adaptive evasion suite.

### 3.4 Software-Derived Device Authorization & Encryption (`src/phase3/` & `src/security/`)
*   **Key Derivation**: Uses HKDF-SHA256 over software-derived device identity attributes (CPU model, system machine ID, disk UUID) concatenated with deployment salt $S_{\text{device}}$:
    $$\text{Key}_{\text{AES}} = \text{HKDF-SHA256}(\text{SoftwareDeviceIdentity} \parallel S_{\text{device}})$$
*   **Encryption**: AES-256-GCM authenticated encryption.
*   **Digital Signature**: RSA-PSS 2048-bit signature with SHA-256 digest over the manifest and encrypted archive.
*   **Anti-Replay Tracker**: Monotonically increasing sequence number and expiration nonce verification.

---

## 4. Key Empirical Findings

1.  **PII Redaction Performance**: The hybrid PII redaction engine achieved a **0.9620 micro-average F1 score** (0.9500 Precision, 0.9744 Recall) on the evaluated synthetic benchmark. (Generation-level memorization leakage rates were *Not experimentally verified* due to offline evaluation without live LLM weights loaded.)
2.  **Adaptive Evasion Robustness**: Single-modal structural screening degrades to **0.0% detection (100% FNR)** against Level-2 and Level-3 adaptive evasion attacks (averaging 75.0% across all levels), while SecureLoRA's joint Structural + Behavioral screen achieved a **1.0000 F1 score ($\tau=0.35$)** on the evaluated multi-seed evasion suite.
3.  **Sub-Linear Scalability**: Cryptographic encryption/decryption overhead scaled sub-linearly with model size (+9.02 ms from 68M to 350M parameters), while full security screening pass latency scaled by +68.77 ms (+77.79 ms total security latency increase across tiers).
4.  **Device Authorization Availability**: Adaptive device authorization achieved a **60.0% reduction in false rejections** (reducing legitimate FRR from 80.0% static down to 20.0% adaptive) while maintaining a **100.0% rejection rate** against foreign hardware clones on the evaluated test set.
