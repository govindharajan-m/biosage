"""
fetchers/disgenet_fetcher.py
────────────────────────────
Fetches gene-disease associations from DisGeNET.
Provides association scores that measure how strongly a gene is linked to a disease.

Usage (from project root):
    python -m data_pipeline.fetchers.disgenet_fetcher
    python -m data_pipeline.fetchers.disgenet_fetcher --disease "asthma"
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import REQUEST_TIMEOUT, RAW_DIR, build_text_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR = RAW_DIR / "disgenet"
DISGENET_API = "https://www.disgenet.org/api"


# ── API Helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = {}) -> dict | list:
    for attempt in range(3):
        try:
            r = requests.get(
                f"{DISGENET_API}/{endpoint}",
                params=params, 
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            
            # Disgenet APIs often return JSON directly
            if r.content:
                return r.json()
            return []
            
        except requests.RequestException as e:
            log.warning(f"DisGeNET request failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
            
    raise RuntimeError(f"All retries failed for DisGeNET /{endpoint}")


# ── Fetch & Parse ─────────────────────────────────────────────────────────────

def fetch_disgenet_associations(disease_query: str, max_results: int = 100) -> list[dict]:
    """
    Search DisGeNET for a disease term to get its associations.
    Note: Free tier doesn't always support direct string search on the main endpoint, 
    so we search diseases first to get a UMLS concept ID, then fetch associations.
    """
    log.info(f"Searching DisGeNET for disease keyword: {disease_query!r}")
    
    try:
        # Search for disease to get UMLS CUI (Concept Unique Identifier)
        search_data = _get("gda/disease", {"disease": disease_query})
    except RuntimeError as e:
        log.error(f"DisGeNET API error: {e}")
        return []

    # Format varies by endpoint version; we'll parse standard gda returns
    if isinstance(search_data, dict) and "payload" in search_data:
        items = search_data["payload"]
    elif isinstance(search_data, list):
        items = search_data
    else:
        log.warning(f"Unexpected API response format from DisGeNET")
        items = []
        
    log.info(f"Found {len(items)} gene-disease associations matching keyword")
    
    # Sort by association score if available
    items.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    records = []
    for item in items[:max_results]:
        record = _parse_association(item, disease_query)
        if record:
            records.append(record)

    return records


def _parse_association(item: dict, target_disease: str) -> dict | None:
    """Parse one DisGeNET GDA (Gene-Disease Association) entry."""
    
    gene_symbol = item.get("gene_symbol")
    disease_name = item.get("disease_name", target_disease)
    
    if not gene_symbol:
        return None
        
    score = float(item.get("score", 0))
    ei = item.get("ei", "") # Evidence Index
    
    # For DisGeNET, we represent the "variant" at the gene level 
    # since it's an association database, not a specific variant database
    
    phenotype = f"DisGeNET association score {score:.3f} for gene {gene_symbol}."
    if ei:
        phenotype += f" Evidence Index: {ei}."

    record = {
        "record_id":              f"disgenet_{gene_symbol}_{disease_name.replace(' ', '_')}",
        "source_db":              "DisGeNET",
        "source_id":              item.get("diseaseid", ""), # usually UMLS CUI

        "disease_name":           disease_name,
        "disease_aliases":        [],
        "mim_number":             None,
        "mondo_id":               None,
        "mesh_id":                None,

        "species":                "Homo sapiens",
        "species_common":         "human",

        "gene":                   gene_symbol,
        "gene_id":                str(item.get("geneid", "")),
        "variant_type":           "association",
        "mutation":               "N/A",
        "chromosome":             "",
        "position":               None,
        "ref_allele":             None,
        "alt_allele":             None,
        "zygosity":               None,

        "clinical_significance":  "Associated" if score > 0.1 else "Weak association",
        "inheritance":            "",
        "phenotype":              phenotype,
        "onset":                  None,

        "pathway":                None,
        "pathway_id":             None,

        "pmids":                  [],
        "text_summary":           "",
    }

    record["text_summary"] = build_text_summary(record)
    return record


# ── Save ──────────────────────────────────────────────────────────────────────

def save_records(records: list[dict], slug: str):
    out_path = SAVE_DIR / f"{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(records)} records → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch DisGeNET gene-disease associations")
    parser.add_argument("--disease", default="asthma", help="Target disease keyword")
    parser.add_argument("--max", type=int, default=100, help="Max records to fetch (sorted by score)")
    args = parser.parse_args()

    records = fetch_disgenet_associations(args.disease, args.max)
    slug = f"disgenet_{args.disease.lower().replace(' ', '_')}"
    
    if records:
        save_records(records, slug)
        print(f"\n✅ Done — {len(records)} associations fetched for {args.disease}")
        print("\nSample record:")
        sample = records[0]
        for k in ["record_id", "disease_name", "gene", "clinical_significance", "phenotype"]:
            print(f"  {k}: {sample.get(k)}")
        print(f"  text_summary: {sample.get('text_summary', '')[:120]}...")
    else:
        print("⚠️  No records fetched — API may have changed or term not found")
