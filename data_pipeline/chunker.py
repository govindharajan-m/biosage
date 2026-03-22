"""
chunker.py
───────────
Reads the unified `all_records.json`, extracts the `text_summary` + key metadata 
from each record, and breaks it into smaller logical chunks.
The output is `chunks.json`, which is the direct input for the Vector Store.

Usage:
    python -m data_pipeline.chunker
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

from .normalizer import OUT_FILE as IN_FILE
from config import PROC_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_FILE = PROC_DIR / "chunks.json"

# In a generic pipeline you might use langchain's RecursiveCharacterTextSplitter,
# but our records are already mostly structured and short. We'll simply ensure
# they aren't too long, and wrap them in a chunk structure.
MAX_WORDS = 500


def create_chunks() -> None:
    if not IN_FILE.exists():
        log.error(f"Input file not found: {IN_FILE}. Run normalizer first.")
        return

    with open(IN_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    log.info(f"Loaded {len(records)} records for chunking.")
    all_chunks = []

    for rec in records:
        chunks = _chunk_record(rec)
        all_chunks.extend(chunks)

    # Save to disk
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    log.info(f"Created {len(all_chunks)} chunks from {len(records)} original records.")
    print(f"\n✅ Saved {len(all_chunks)} chunks to data/processed/chunks.json")


def _chunk_record(record: dict) -> List[dict]:
    """Convert a single record into one or more chunks."""
    rec_id = record.get("record_id", "unknown")
    text = record.get("text_summary", "").strip()
    
    # Optional context to prepend to every chunk from this record 
    # to maintain semantic meaning if it gets split
    disease = record.get("disease_name", "Unknown Disease")
    gene = record.get("gene", "")
    species = record.get("species_common", "human")
    source = record.get("source_db", "Unknown")
    
    base_meta = {
        "record_id": rec_id,
        "disease_name": disease,
        "gene": gene,
        "species_common": species,
        "source_db": source
    }

    if not text:
        log.warning(f"Empty text for record {rec_id}, skipping.")
        return []

    words = text.split()
    
    # If the text is short enough, it's just one chunk
    if len(words) <= MAX_WORDS:
        return [{
            "chunk_id": f"{rec_id}_0",
            "text": text,
            "metadata": base_meta
        }]

    # If it's too long, split it up intelligently
    chunks = []
    chunk_idx = 0
    current_words = []
    
    # Small overlap to preserve context across boundaries
    OVERLAP = 20
    
    for i, word in enumerate(words):
        current_words.append(word)
        
        # When we hit max, or end of text
        if len(current_words) >= MAX_WORDS or i == len(words) - 1:
            chunk_text = " ".join(current_words)
            
            # If not the first chunk, prepend context so it stands alone semantically
            if chunk_idx > 0:
                context = f"[Continuation for {disease} / Gene {gene}]: "
                chunk_text = context + chunk_text
                
            chunks.append({
                "chunk_id": f"{rec_id}_{chunk_idx}",
                "text": chunk_text,
                "metadata": base_meta
            })
            
            chunk_idx += 1
            # Keep overlap words for the next chunk
            current_words = current_words[-OVERLAP:]
            
    return chunks


if __name__ == "__main__":
    create_chunks()
