import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.phase3.config import Phase3Config
from src.security import (
    encrypt_adapter,
    get_fingerprint_hash,
    compute_sha256,
    save_hash,
    derive_key_from_env,
    generate_dev_keypair,
    sign_digest,
    save_signature,
)
from src.phase3.verifier import verify_and_decrypt
from src.common.exceptions import VerificationError
from src.phase3.package_builder import build_package, export_package_archive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("secure_lora.phase3.main")


def cmd_protect(args: argparse.Namespace) -> int:
    """
    Step-by-step protection pipeline:
      1. Validate config
      2. Device fingerprint
      3. Key derivation
      4. Encrypt adapter (Step 4)
      5. Compute + save hash (Step 5)
      6. Generate dev RSA keys if needed; sign hash (Step 6)
      7. Build package + manifest (Step 7)
      8. Optionally export tar.gz
    """
    cfg = Phase3Config

    # Step 1 — validate
    try:
        cfg.validate()
    except (EnvironmentError, FileNotFoundError) as exc:
        logger.error("Config validation failed: %s", exc)
        return 1

    # Step 2 — fingerprint
    fp_hash = get_fingerprint_hash()

    # Step 3 — key derivation
    key = derive_key_from_env(fp_hash)

    # Step 4 — encrypt
    enc_meta = encrypt_adapter(
        adapter_input=cfg.ADAPTER_INPUT_DIR,
        output_enc_path=cfg.enc_path(),
        key=key,
        fingerprint_hash=fp_hash,
        metadata_path=cfg.metadata_path(),
    )

    # Step 5 — hash
    digest = compute_sha256(cfg.enc_path())
    save_hash(digest, cfg.hash_path())

    # Step 6 — RSA sign
    priv_path = cfg.RSA_PRIVATE_KEY_PATH
    pub_path  = cfg.RSA_PUBLIC_KEY_PATH
    if not priv_path.exists() or not pub_path.exists():
        logger.info("Dev RSA keypair not found — generating new pair…")
        generate_dev_keypair(priv_path, pub_path, key_size=cfg.RSA_KEY_BITS)

    signature = sign_digest(digest, priv_path)
    save_signature(signature, cfg.sig_path())

    # Step 7 — package
    manifest = build_package(
        package_dir=cfg.PROTECTED_OUTPUT_DIR,
        adapter_id=cfg.ADAPTER_ID,
        model_reference=cfg.MODEL_REFERENCE,
        fingerprint_hash=fp_hash,
        package_version=cfg.PACKAGE_VERSION,
        enc_metadata=enc_meta,
        public_key_src=pub_path,
    )

    # Optional tar export
    if args.archive:
        archive_path = export_package_archive(cfg.PROTECTED_OUTPUT_DIR)
        logger.info("Package archive: %s", archive_path)

    # Step 10 — inline report
    _print_report(
        fp_hash=fp_hash,
        enc_meta=enc_meta,
        manifest=manifest,
        sig_status="SIGNED",
        integrity_status="HASHED",
        auth_result="ENCRYPTED_AND_PACKAGED",
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """
    Step-by-step verification + controlled decryption pipeline (Steps 8 + 9).
    """
    cfg = Phase3Config
    package_dir = cfg.PROTECTED_OUTPUT_DIR
    output_path = Path(args.output) if args.output else Path("/tmp/restored_adapter_p3.tar.gz")

    logger.info("Starting Phase 3 verification on package: %s", package_dir)
    try:
        result = verify_and_decrypt(package_dir, output_path)
        logger.info("Decrypted adapter written to: %s", result)
        _print_report(
            fp_hash=get_fingerprint_hash(),
            enc_meta=json.loads((package_dir / "metadata.json").read_text()),
            manifest=json.loads((package_dir / "package_manifest.json").read_text()),
            sig_status="VERIFIED",
            integrity_status="VERIFIED",
            auth_result="AUTHORISED",
        )
        return 0
    except VerificationError as exc:
        logger.error("VERIFICATION FAILED: %s", exc)
        return 2


def cmd_report(args: argparse.Namespace) -> int:
    """Prints a security report from an existing package without decrypting."""
    cfg = Phase3Config
    pkg = cfg.PROTECTED_OUTPUT_DIR

    manifest_path = pkg / "package_manifest.json"
    meta_path     = pkg / "metadata.json"
    hash_path     = pkg / "adapter.hash"

    if not manifest_path.exists():
        logger.error("No package found at %s. Run 'protect' first.", pkg)
        return 1

    manifest  = json.loads(manifest_path.read_text())
    enc_meta  = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fp_hash   = manifest.get("device_fingerprint_hash_ref", "UNKNOWN")
    sig_ok    = (pkg / "adapter.sig").exists()
    hash_ok   = hash_path.exists()

    _print_report(
        fp_hash=fp_hash,
        enc_meta=enc_meta,
        manifest=manifest,
        sig_status="PRESENT" if sig_ok else "MISSING",
        integrity_status="PRESENT" if hash_ok else "MISSING",
        auth_result="NOT_VERIFIED_IN_REPORT_MODE",
    )
    return 0


def _print_report(
    fp_hash: str,
    enc_meta: dict,
    manifest: dict,
    sig_status: str,
    integrity_status: str,
    auth_result: str,
) -> None:
    report = {
        "phase": "3 — Adapter Protection and Device Binding",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint_hash_prefix": fp_hash[:16] + "…",
        "encryption_algorithm": enc_meta.get("algorithm", "N/A"),
        "adapter_format": enc_meta.get("adapter_format", "N/A"),
        "package_version": manifest.get("schema_version", "N/A"),
        "adapter_id": manifest.get("adapter_id", "N/A"),
        "model_reference": manifest.get("model_reference", "N/A"),
        "package_contents": list((manifest.get("artefact_hashes") or {}).keys()),
        "plaintext_in_package": manifest.get("security_notes", {}).get("plaintext_in_package", "N/A"),
        "signature_status": sig_status,
        "integrity_status": integrity_status,
        "authorization_result": auth_result,
    }
    print("\n" + "=" * 70)
    print("  PHASE 3 SECURITY REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print("=" * 70 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m phase3",
        description="Phase 3: Secure Adapter Protection and Device Binding",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("protect", help="Encrypt, hash, sign, and package the adapter.")
    p.add_argument(
        "--archive",
        action="store_true",
        help="Also export the package as a .tar.gz archive.",
    )

    v = sub.add_parser("verify", help="Verify package and decrypt on authorised device.")
    v.add_argument(
        "--output",
        default=None,
        help="Path for the decrypted adapter output (default: /tmp/restored_adapter_p3.tar.gz).",
    )

    sub.add_parser("report", help="Print security report from an existing package.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "protect": cmd_protect,
        "verify":  cmd_verify,
        "report":  cmd_report,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
