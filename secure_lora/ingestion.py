import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def ingest_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Ingests a single raw file and returns a list of raw records.
    Supports .txt, .csv, .json, and .md formats.
    """
    suffix = file_path.suffix.lower()
    records = []
    
    try:
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    cleaned = line.strip()
                    if cleaned:
                        records.append({
                            "text": cleaned,
                            "source_file": file_path.name,
                            "line_number": line_num
                        })
                        
        elif suffix == '.md':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                # Split markdown into paragraphs/sections by double newline
                blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
                for idx, block in enumerate(blocks, 1):
                    records.append({
                        "text": block,
                        "source_file": file_path.name,
                        "block_index": idx
                    })
                    
        elif suffix == '.csv':
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader, 1):
                    # Filter out None keys or empty headers
                    cleaned_row = {k: v for k, v in row.items() if k is not None}
                    cleaned_row["source_file"] = file_path.name
                    cleaned_row["row_index"] = row_idx
                    records.append(cleaned_row)
                    
        elif suffix == '.json':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data, 1):
                        if isinstance(item, dict):
                            item_copy = dict(item)
                            item_copy["source_file"] = file_path.name
                            item_copy["record_index"] = idx
                            records.append(item_copy)
                        elif isinstance(item, str):
                            records.append({
                                "text": item,
                                "source_file": file_path.name,
                                "record_index": idx
                            })
                elif isinstance(data, dict):
                    data_copy = dict(data)
                    data_copy["source_file"] = file_path.name
                    records.append(data_copy)
                else:
                    logger.warning(f"Unsupported JSON root type in {file_path}: {type(data)}")
                    
    except Exception as e:
        logger.error(f"Error ingesting file {file_path}: {e}")
        
    return records

def ingest_directory(directory_path: Path, recursive: bool = True) -> List[Dict[str, Any]]:
    """
    Scans the given directory for raw text, markdown, csv, and json files and ingests them.
    """
    all_records = []
    supported_extensions = {'.txt', '.csv', '.json', '.md'}
    
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return all_records
        
    glob_pattern = "**/*" if recursive else "*"
    for path in directory_path.glob(glob_pattern):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            logger.info(f"Ingesting file: {path.relative_to(directory_path)}")
            file_records = ingest_file(path)
            all_records.extend(file_records)
            
    logger.info(f"Ingested a total of {len(all_records)} raw records from {directory_path}")
    return all_records
