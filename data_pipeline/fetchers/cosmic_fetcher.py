"""
fetchers/cosmic_fetcher.py
───────────────────────────
Fetches cancer mutation records from COSMIC (Catalogue Of Somatic Mutations In Cancer).
Uses the public COSMIC REST API v3.3 to find mutations by gene.

Usage (from project root):
    python -m data_pipeline.fetchers.cosmic_fetcher
    python -m data_pipeline.fetchers.cosmic_fetcher --gene BRCA1 --disease "breast cancer" --max 100
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

SAVE_DIR = RAW_DIR / "cosmic"
COSMIC_API = "https://cancer.sanger.ac.uk/api/v3.3"


# ── API Helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = {}) -> dict:
    """Fetch from COSMIC API with retries."""
    for attempt in range(3):
        try:
            r = requests.get(
                f"{COSMIC_API}/{endpoint}",
                params=params, 
                timeout=REQUEST_TIMEOUT,
                headers={"Accept": "application/json"}
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning(f"COSMIC request failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All retries failed for COSMIC /{endpoint}")


# ── Fetch & Parse ─────────────────────────────────────────────────────────────

def fetch_cosmic_mutations(gene_symbol: str, target_disease: str, max_results: int = 100) -> list[dict]:
    """
    Search COSMIC for mutations in a specific gene.
    We'll tag the resulting records with the given target target_disease name.
    """
    log.info(f"Searching COSMIC for mutations in gene {gene_symbol} (max {max_results})")
    
    try:
        # COSMIC groups by gene symbol 
        data = _get(f"mutations/gene/{gene_symbol}", {"max_results": min(max_results, 500)})
    except RuntimeError as e:
        log.error(f"COSMIC API error: {e}")
        return []

    mutations = data.get("mutations", [])
    log.info(f"Found {len(mutations)} mutations for {gene_symbol}")

    # The API might be paginated, but for now we'll take top results
    records = []
    for mut in mutations[:max_results]:
        record = _parse_mutation(mut, gene_symbol, target_disease)
        if record:
            records.append(record)

    return records


def _parse_mutation(mut: dict, gene_symbol: str, target_disease: str) -> dict | None:
    """Parse one COSMIC mutation entry."""
    
    # E.g. "COSM12345"
    mut_id = mut.get("id")
    if not mut_id:
        return None
        
    # E.g. "c.123A>T", "p.Lys123Met"
    cds_mut = mut.get("cds", "")
    aa_mut = mut.get("aa", "")
    mutation_str = aa_mut if aa_mut else cds_mut
    
    # E.g. "Substitution - Missense"
    mut_type_raw = mut.get("type", "")
    
    # Map COSMIC somatic mutation types
    if "substitution" in mut_type_raw.lower():
        variant_type = "SNP"
    elif "deletion" in mut_type_raw.lower() or "insertion" in mut_type_raw.lower():
        variant_type = "indel"
    else:
        variant_type = "other"

    # In COSMIC, records are somatic mutations, usually pathogenic for the cancer
    clinsig = "Pathogenic (Somatic)"
    if "Pathogenic" in mut.get("fathmm_prediction", ""):
        clinsig = "Pathogenic"

    record = {
        "record_id":              f"cosmic_{mut_id}",
        "source_db":              "COSMIC",
        "source_id":              str(mut_id),

        "disease_name":           target_disease,
        "disease_aliases":        [],
        "mim_number":             None,
        "mondo_id":               None,
        "mesh_id":                None,

        "species":                "Homo sapiens",
        "species_common":         "human",

        "gene":                   gene_symbol,
        "gene_id":                None,
        "variant_type":           variant_type,
        "mutation":               mutation_str,
        "chromosome":             "",
        "position":               None,
        "ref_allele":             None,
        "alt_allele":             None,
        "zygosity":               "somatic",

        "clinical_significance":  clinsig,
        "inheritance":            "Somatic mutation",
        "phenotype":              f"Somatic mutation {mutation_str} in {gene_symbol} associated with cancer.",
        "onset":                  "adult",

        "pathway":                "Cancer / Cell Cycle",
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
    parser = argparse.ArgumentParser(description="Fetch COSMIC cancer mutations by gene")
    parser.add_argument("--gene", default="TP53", help="Gene symbol to search")
    parser.add_argument("--disease", default="Lung cancer", help="Target disease name to tag these mutations with")
    parser.add_argument("--max", type=int, default=100, help="Max records to fetch")
    args = parser.parse_args()

    records = fetch_cosmic_mutations(args.gene, args.disease, args.max)
    slug = f"cosmic_{args.gene.lower()}_{args.disease.lower().replace(' ', '_')}"
    
    if records:
        save_records(records, slug)
        print(f"\n✅ Done — {len(records)} somatic mutations fetched for {args.gene}")
        print("\nSample record:")
        sample = records[0]
        for k in ["record_id", "disease_name", "gene", "variant_type", "mutation", "clinical_significance"]:
            print(f"  {k}: {sample.get(k)}")
        print(f"  text_summary: {sample.get('text_summary', '')[:120]}...")
    else:
        print("⚠️  No records fetched — check gene symbol connectivity")
