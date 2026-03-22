"""
normalizer.py
─────────────
Merge all raw fetched JSONs from data/raw/**/*.json into a single 
unified list of disease-variant records, deduplicate them, validate fields,
and write to data/processed/all_records.json.

Usage (from project root):
    python -m data_pipeline.normalizer
"""

import json
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DIR, PROC_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ensure output dir exists
PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = PROC_DIR / "all_records.json"


def normalize_all():
    log.info(f"Scanning {RAW_DIR} for JSON records...")
    
    all_files = list(RAW_DIR.rglob("*.json"))
    log.info(f"Found {len(all_files)} raw data files.")
    
    if not all_files:
        log.warning("No data files found! Have you run any fetchers yet?")
        return

    merged_records = []
    seen_ids = set()
    source_counts = {}

    for file_path in all_files:
        db_name = file_path.parent.name
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            log.error(f"Failed to read {file_path.name}: {e}")
            continue
            
        added = 0
        for rec in records:
            # 1. Deduplicate by record_id
            rec_id = rec.get("record_id")
            if not rec_id or rec_id in seen_ids:
                continue
                
            seen_ids.add(rec_id)
            merged_records.append(rec)
            
            # Keep count by source 
            source_db = rec.get("source_db", db_name)
            source_counts[source_db] = source_counts.get(source_db, 0) + 1
            added += 1
            
        log.debug(f"Loaded {added} unique records from {file_path.name}")

    if not merged_records:
        log.warning("No valid unique records loaded.")
        return
        
    log.info(f"Merged {len(merged_records)} total unique records.")
    
    # Write to disk
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, indent=2, ensure_ascii=False)
        
    # Print summary
    print(f"\n{'═'*60}")
    print("  Normalization Summary")
    print(f"{'═'*60}")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:20s}: {count:>6} records")
    print(f"{'═'*60}")
    print(f"  Total               : {len(merged_records):>6} records")
    print(f"  Saved to            : {OUT_FILE.relative_to(Path.cwd())}")


if __name__ == "__main__":
    normalize_all()
