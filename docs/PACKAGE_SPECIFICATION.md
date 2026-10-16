# SecureLoRA Package Specification (Version 1.0.0+)

This document defines the formal package specification, required artifacts, canonical manifest schema, cryptographic bindings, and lifecycle validation rules for SecureLoRA distribution packages.

---

## 1. Package Directory & Archive Structure

A SecureLoRA package is a directory or `.tar.gz` archive containing the following required artifacts. Plaintext weights, private signing keys, and device secrets are **strictly forbidden** from the distribution package.

| Artifact File | Format | Purpose | Required |
| :--- | :--- | :--- | :--- |
| `adapter.enc` | Binary | AES-256-GCM encrypted adapter payload (tar archive of PEFT weights & config). | **Yes** |
| `adapter.hash` | Text (64 hex chars) | Lowercase SHA-256 hex digest of `adapter.enc` for fast integrity checks. | **Yes** |
| `adapter.sig` | Binary (256 bytes) | RSA-2048-PSS signature over the SHA-256 canonical manifest authentication digest. | **Yes** |
| `metadata.json` | JSON Object | Encryption envelope metadata (IV, authentication tag, KDF version, salt hex). | **Yes** |
| `public.pem` | PEM (PKCS#8) | RSA-2048 public verification key authenticating package provenance. | **Yes** |
| `package_manifest.json` | JSON Object | Extended provenance and anti-replay manifest binding 18+ security fields. | **Yes** |

### Forbidden Artifacts
The package directory and exported archives (`.tar.gz`) **must never contain**:
- Any RSA private signing key (`dev_private.pem`, `*private*.pem`, `*.key`, `*priv*.pem`).
- Any device secret or raw secret file (`device.secret`, `*.secret`, `*device_secret*`).
- Any plaintext adapter weight or configuration file (`adapter_model.safetensors`, `adapter_model.bin`, `adapter_config.json`, `*.safetensors`, `*.pt`, `*.bin`).

---

## 2. Package Creation Lifecycle & Ordering

To eliminate circular dependencies and ensure strict cryptographic consistency, package creation must execute in the following exact sequence:

```mermaid
flowchart TD
    A[Adapter Screening Gate] --> B[AES-256-GCM Encryption]
    B --> C[Compute Ciphertext SHA-256 Digest]
    C --> D[Save adapter.hash]
    D --> E[Stage public.pem]
    E --> F[Generate Manifest with 18+ Fields & Screening Report]
    F --> G[Compute Canonical Manifest Authentication Digest]
    G --> H[RSA-2048-PSS Signature Generation -> adapter.sig]
    H --> I[Update Non-Canonical artefact_hashes['adapter.sig']]
    I --> J[Save package_manifest.json]
    J --> K[Immediate In-Memory Self-Verification]
    K --> L[Scrub Secrets & Verify Completeness]
```

1. **Pre-packaging Security Screening Gate**: Vets real adapter weights for backdoors and structural anomalies.
2. **Adapter Encryption**: Encrypts adapter payload with AES-256-GCM and random 12-byte IV using the device-bound HKDF key.
3. **Ciphertext Integrity**: Computes SHA-256 of `adapter.enc` and atomically writes `adapter.hash`.
4. **Public Verification Key Staging**: Copies `public.pem` into the package directory.
5. **Manifest Generation**: Generates `package_manifest.json` with all canonical security fields, initial artifact hashes, and screening provenance.
6. **Signature Generation**:
   - Asserts `private_key_src` is present, readable, and a valid RSA private key (fails closed if missing or unusable when signing is required).
   - Computes canonical manifest digest binding all 18 security fields.
   - Signs canonical digest using RSA-PSS SHA-256.
   - Saves raw signature bytes to `adapter.sig`.
7. **Artifact Hash Update**: Updates `manifest["artefact_hashes"]["adapter.sig"]` with the SHA-256 hash of `adapter.sig` and saves manifest atomically.
8. **Immediate Self-Verification**: Executes in-memory signature verification against `public.pem` before returning success to guarantee zero manifest-signature drift.
9. **Secrets Scrubbing & Completeness Check**: Removes private keys/secrets and verifies all artifacts exist, are non-empty, and match recorded hashes.

---

## 3. Canonical Manifest Authentication Digest Specification

Current-format packages (`schema_version >= 1.0.0`) bind all security-critical metadata and the ciphertext hash into a deterministic canonical JSON string:

```python
canonical_dict = {
    "package_id": str(manifest["package_id"]),
    "adapter_id": str(manifest["adapter_id"]),
    "base_model_id": str(manifest["base_model_id"]),
    "model_revision": str(manifest["model_revision"]),
    "adapter_revision": str(manifest["adapter_revision"]),
    "package_version": str(manifest["package_version"]),
    "creation_timestamp": str(manifest["creation_timestamp"]),
    "expiration_timestamp": manifest["expiration_timestamp"],
    "sequence_number": int(manifest["sequence_number"]),
    "binding_policy_version": str(manifest["binding_policy_version"]),
    "kdf_version": str(manifest["kdf_version"]),
    "encryption_version": str(manifest["encryption_version"]),
    "signature_algorithm": str(manifest["signature_algorithm"]),
    "digest_algorithm": str(manifest["digest_algorithm"]),
    "deployment_policy": manifest["deployment_policy"],
    "device_fingerprint_hash_ref": str(manifest["device_fingerprint_hash_ref"]),
    "encrypted_adapter_digest": ciphertext_digest.strip().lower(),
}

canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
canonical_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
```

The RSA-PSS signature verifies `canonical_digest.encode("utf-8")`. Modifying any field in the manifest or ciphertext invalidates the signature.

---

## 4. Manifest Schema (`package_manifest.json`)

```json
{
  "schema_version": "1.0.0",
  "package_id": "urn:uuid:11112222-3333-4444-5555-666677778888",
  "adapter_id": "medical-lora-v1",
  "base_model_id": "distilbert-base-uncased",
  "model_reference": "distilbert-base-uncased",
  "model_revision": "main",
  "adapter_revision": "v1.0.0",
  "package_version": "1.0.0",
  "creation_timestamp": "2026-10-16T12:00:00+00:00",
  "created_at_utc": "2026-10-16T12:00:00+00:00",
  "expiration_timestamp": null,
  "binding_policy_version": "1.0.0",
  "kdf_version": "hkdf-sha256-v2",
  "hkdf_salt_hex": "64_char_hex_salt",
  "encryption_version": "aes-256-gcm-v1",
  "signature_algorithm": "rsa-pss-2048-sha256",
  "digest_algorithm": "sha256",
  "nonce_metadata": {
    "iv_bytes": 12,
    "tag_bytes": 16,
    "salt_reference": "hkdf_csprng_salt"
  },
  "deployment_policy": {
    "strictness": "high",
    "allowed_feature_changes": {
      "network_interface": true,
      "hostname": false,
      "machine_id": false,
      "disk_uuid": false
    }
  },
  "sequence_number": 42,
  "device_fingerprint_hash_ref": "64_char_hex_hash",
  "encrypted_adapter_digest": "64_char_hex_ciphertext_sha256",
  "verification_instructions": "Execute Phase 4 verification steps 1-9 in order before decryption or loading.",
  "artefact_hashes": {
    "adapter.enc": "sha256_hex",
    "adapter.hash": "sha256_hex",
    "adapter.sig": "sha256_hex",
    "metadata.json": "sha256_hex",
    "public.pem": "sha256_hex"
  },
  "screening_report": {
    "adapter_weight_path": "/path/to/adapter_model.safetensors",
    "screening_mode": "production",
    "real_weights_used": true,
    "real_callable_used": false,
    "risk_score": 0.05,
    "decision": "PASSED",
    "risk_level": "LOW"
  },
  "security_notes": {
    "plaintext_in_package": false,
    "private_key_in_package": false,
    "salt_in_package": false,
    "assurance": "Provides cryptographic authenticity and provenance under the assumed private-key security model."
  }
}
```

---

## 5. Legacy Versioning Policy

Legacy package compatibility is explicitly segregated:
- **Current versions (`1.0.0`, `2.0.0`, `3.0.0`)**: Require full canonical manifest schema validation and RSA-PSS verification over `canonical_digest`. Never fallback to raw ciphertext validation.
- **Legacy versions (`0.0.0`, `0.1.0`, `legacy`, `v0`)**: Processed solely via `_validate_legacy_package`, requiring `allow_legacy=True` flag. Authenticates raw ciphertext digest directly.
- **Unknown versions**: Rejected with typed `ManifestSchemaError`.
