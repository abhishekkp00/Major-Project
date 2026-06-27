import argparse
import sys
import os
import json
import logging
from pathlib import Path

from secure_lora.security import generate_key, compute_sha256
from secure_lora.pipeline import SecureDatasetPipeline

# Load environment variables from .env file at startup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Custom fallback loader to read .env without external packages
    def load_dotenv(dotenv_path=".env"):
        path = Path(dotenv_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key:
                            os.environ[key] = val
    load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("secure_lora_cli")

def resolve_key(args_key_file: str = None) -> bytes:
    """
    Resolves the 32-byte encryption key from:
    1. The SECURE_LORA_KEY_HEX environment variable.
    2. The SECURE_LORA_KEY_PATH environment variable.
    3. The CLI parameter args_key_file.
    """
    # 1. Check HEX environment variable
    env_hex = os.environ.get("SECURE_LORA_KEY_HEX")
    if env_hex:
        try:
            cleaned_hex = env_hex.strip().strip("'\"")
            if cleaned_hex:
                key = bytes.fromhex(cleaned_hex)
                if len(key) != 32:
                    raise ValueError(f"Expected 32 bytes, got {len(key)}")
                logger.info("Using encryption key loaded from SECURE_LORA_KEY_HEX environment variable.")
                return key
        except Exception as e:
            logger.error(f"Error parsing SECURE_LORA_KEY_HEX env variable: {e}")
            sys.exit(1)
            
    # 2. Check path from environment variable or argument
    env_path = os.environ.get("SECURE_LORA_KEY_PATH")
    key_path_str = args_key_file or env_path
    if not key_path_str:
        logger.error(
            "Encryption key not specified! Please set SECURE_LORA_KEY_HEX or "
            "SECURE_LORA_KEY_PATH in .env, or pass the --key-file argument."
        )
        sys.exit(1)
        
    key_file = Path(key_path_str)
    if not key_file.exists():
        logger.error(f"Key file not found: {key_file}")
        sys.exit(1)
        
    if os.name == 'posix':
        mode = key_file.stat().st_mode
        if (mode & 0o077) != 0:
            logger.warning(f"Key file {key_file} has unsafe permissions. Recommended: chmod 600.")
            
    key = key_file.read_bytes()
    if len(key) != 32:
        logger.error(f"Invalid key length in {key_file}: expected 32 bytes, got {len(key)} bytes.")
        sys.exit(1)
        
    logger.info(f"Using encryption key loaded from file: {key_file}")
    return key

def write_key(key_file: Path, key: bytes) -> None:
    """Writes key to file and enforces strict 0600 permissions on POSIX systems."""
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(key)
    if os.name == 'posix':
        key_file.chmod(0o600)
    logger.info(f"Successfully generated and secured key file: {key_file} (permissions: 0600)")

def cmd_generate_key(args) -> None:
    key_file_str = args.key_file or os.environ.get("SECURE_LORA_KEY_PATH")
    if not key_file_str:
        logger.error("Key file destination path not specified. Set SECURE_LORA_KEY_PATH in .env or pass -k/--key-file.")
        sys.exit(1)
        
    key_file = Path(key_file_str)
    if key_file.exists() and not args.force:
        logger.error(f"Key file {key_file} already exists. Use --force to overwrite.")
        sys.exit(1)
        
    key = generate_key()
    write_key(key_file, key)

def cmd_encrypt(args) -> None:
    input_dir_str = args.input_dir or os.environ.get("SECURE_LORA_INPUT_DIR")
    output_dir_str = args.output_dir or os.environ.get("SECURE_LORA_OUTPUT_DIR")
    dataset_name = args.dataset_name or os.environ.get("SECURE_LORA_DATASET_NAME")
    version = args.version or os.environ.get("SECURE_LORA_DATASET_VERSION") or "1.0.0"
    
    if not input_dir_str:
        logger.error("Input directory not specified. Set SECURE_LORA_INPUT_DIR in .env or pass -i/--input-dir.")
        sys.exit(1)
    if not output_dir_str:
        logger.error("Output directory not specified. Set SECURE_LORA_OUTPUT_DIR in .env or pass -o/--output-dir.")
        sys.exit(1)
    if not dataset_name:
        logger.error("Dataset name not specified. Set SECURE_LORA_DATASET_NAME in .env or pass -n/--dataset-name.")
        sys.exit(1)
        
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    
    # Resolve the encryption key
    key = resolve_key(args.key_file)
    
    pipeline = SecureDatasetPipeline(key)
    try:
        metadata = pipeline.encrypt_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
            version=version,
            shred_raw=args.shred_raw
        )
        logger.info("Encryption pipeline finished. Metadata summary:")
        logger.info(f"  Dataset: {metadata['dataset_name']} (v{metadata['version']})")
        logger.info(f"  Ingested: {metadata['raw_records_ingested']} records")
        logger.info(f"  Processed: {metadata['num_records']} records")
        logger.info(f"  Size: {metadata['encrypted_file_size_bytes']} bytes")
        logger.info(f"  SHA-256 Checksum: {metadata['encrypted_file_sha256']}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

def cmd_decrypt(args) -> None:
    encrypted_file = Path(args.encrypted_file)
    output_file = Path(args.output_file)
    
    # Resolve key (args.key_file might be None if env variable loaded)
    key = resolve_key(args.key_file)
    
    pipeline = SecureDatasetPipeline(key)
    try:
        pipeline.decrypt_dataset_to_file(encrypted_file, output_file)
        logger.info(f"Plaintext dataset file restored to {output_file}")
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        sys.exit(1)

def cmd_verify(args) -> None:
    encrypted_file = Path(args.encrypted_file)
    metadata_file = Path(args.metadata_file)
    
    if not encrypted_file.exists():
        logger.error(f"Encrypted dataset file does not exist: {encrypted_file}")
        sys.exit(1)
    if not metadata_file.exists():
        logger.error(f"Metadata file does not exist: {metadata_file}")
        sys.exit(1)
        
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        expected_hash = metadata.get("encrypted_file_sha256")
        expected_size = metadata.get("encrypted_file_size_bytes")
        
        actual_hash = compute_sha256(encrypted_file)
        actual_size = encrypted_file.stat().st_size
        
        logger.info("Verifying integrity of encrypted file against metadata...")
        
        size_match = expected_size == actual_size
        hash_match = expected_hash == actual_hash
        
        if size_match and hash_match:
            logger.info("SUCCESS: The encrypted file matches metadata. Integrity is verified.")
        else:
            logger.error("INTEGRITY VIOLATION DETECTED!")
            if not size_match:
                logger.error(f"  Size mismatch: expected {expected_size} bytes, got {actual_size} bytes")
            if not hash_match:
                logger.error(f"  SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Secure Device-Bound LoRA Framework - Phase 1 Dataset Protection CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # subcommand: generate-key
    parser_key = subparsers.add_parser("generate-key", help="Generates a secure 256-bit encryption key")
    parser_key.add_argument("-k", "--key-file", help="Path to write the key file")
    parser_key.add_argument("-f", "--force", action="store_true", help="Force overwrite key file if it exists")
    
    # subcommand: encrypt
    parser_enc = subparsers.add_parser("encrypt", help="Ingest, preprocess, and encrypt a dataset folder")
    parser_enc.add_argument("-i", "--input-dir", help="Directory containing raw data files (.txt, .csv, .json, .md)")
    parser_enc.add_argument("-o", "--output-dir", help="Directory to save the encrypted dataset and metadata")
    parser_enc.add_argument("-k", "--key-file", help="Path to the encryption key file")
    parser_enc.add_argument("-n", "--dataset-name", help="Name of the dataset (e.g. corporate_finances)")
    parser_enc.add_argument("-v", "--version", help="Dataset version tag")
    parser_enc.add_argument("--shred-raw", action="store_true", help="Securely shred the raw input files after successful encryption")
    
    # subcommand: decrypt
    parser_dec = subparsers.add_parser("decrypt", help="Decrypts encrypted dataset file to a plaintext file (for audit/validation)")
    parser_dec.add_argument("-e", "--encrypted-file", required=True, help="Path to encrypted_dataset.enc")
    parser_dec.add_argument("-o", "--output-file", required=True, help="Path to output decrypted file (e.g. plaintext.jsonl)")
    parser_dec.add_argument("-k", "--key-file", help="Path to the encryption key file")
    
    # subcommand: verify
    parser_ver = subparsers.add_parser("verify", help="Verifies integrity of the encrypted dataset using metadata SHA-256")
    parser_ver.add_argument("-e", "--encrypted-file", required=True, help="Path to encrypted_dataset.enc")
    parser_ver.add_argument("-m", "--metadata-file", required=True, help="Path to dataset_metadata.json")
    
    args = parser.parse_args()
    
    if args.command == "generate-key":
        cmd_generate_key(args)
    elif args.command == "encrypt":
        cmd_encrypt(args)
    elif args.command == "decrypt":
        cmd_decrypt(args)
    elif args.command == "verify":
        cmd_verify(args)

if __name__ == "__main__":
    main()
