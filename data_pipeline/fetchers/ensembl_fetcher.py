"""
fetchers/ensembl_fetcher.py
───────────────────────────
Fetches variant records using the Ensembl REST API for a given gene and species.
Great for enriching our dataset with cross-species variants (human, dog, mouse).

Usage (from project root):
    python -m data_pipeline.fetchers.ensembl_fetcher
    python -m data_pipeline.fetchers.ensembl_fetcher --gene BRCA1 --species human
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

SAVE_DIR = RAW_DIR / "ensembl"
ENSEMBL_REST = "https://rest.ensembl.org"

# Ensure the folder exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── API Helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = {}) -> dict | list:
    for attempt in range(3):
        try:
            r = requests.get(
                f"{ENSEMBL_REST}/{endpoint}",
                params=params, 
                timeout=REQUEST_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            r.raise_for_status()
            
            if r.content:
                return r.json()
            return []
            
        except requests.RequestException as e:
            log.warning(f"Ensembl request failed (attempt {attempt+1}/3): {e}")
            # Ensembl handles rate limits via 429 status code
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 2))
                log.info(f"Rate limited. Sleeping for {retry_after}s")
                time.sleep(retry_after)
            else:
                time.sleep(2 ** attempt)
            
    raise RuntimeError(f"All retries failed for Ensembl /{endpoint}")


# ── Fetch & Parse ─────────────────────────────────────────────────────────────

def fetch_ensembl_variants(gene_symbol: str, species: str = "human", max_results: int = 100) -> list[dict]:
    """
    Search Ensembl for variants in a given gene.
    First we look up the gene to get its region, then fetch variants overlapping that region.
    """
    log.info(f"Searching Ensembl for gene {gene_symbol} in {species}")
    
    try:
        # Step 1: Look up the gene
        lookup = _get(f"lookup/symbol/{species}/{gene_symbol}", {"expand": 1})
        if not lookup or "seq_region_name" not in lookup:
            log.error(f"Could not find gene {gene_symbol} in {species}")
            return []
            
        chrom = lookup["seq_region_name"]
        start = lookup["start"]
        end = lookup["end"]
        gene_id = lookup["id"]
        
        log.info(f"Gene {gene_symbol} found at {chrom}:{start}-{end} (ID: {gene_id})")
        
        # Step 2: Fetch variants overlapping this region
        # The region API limits to 5,000,000bp. We should be well within that for a single gene.
        region = f"{chrom}:{start}-{end}"
        log.info(f"Fetching variants in region {region}...")
        
        feature_data = _get(f"overlap/region/{species}/{region}", {
            "feature": "variation"
        })
        
    except RuntimeError as e:
        log.error(f"Ensembl API error: {e}")
        return []

    log.info(f"Found {len(feature_data)} variants in region")
    
    # We only want variants with a clinical significance or functional consequence
    # to avoid flooding our DB with non-coding/benign SNPs
    filtered = []
    for var in feature_data:
        conseq = var.get("clinical_significance", [])
        conseq_types = var.get("consequence_type", [])
        
        is_pathogenic = any("pathogenic" in c.lower() for c in conseq)
        is_missense = any("missense" in c.lower() or "stop_gained" in c.lower() for c in conseq_types)
        
        if is_pathogenic or is_missense:
            filtered.append(var)
    
    log.info(f"Filtered down to {len(filtered)} potentially impactful variants")

    records = []
    for var in filtered[:max_results]:
        record = _parse_variant(var, gene_symbol, gene_id, species)
        if record:
            records.append(record)

    return records


def _parse_variant(var: dict, gene_symbol: str, gene_id: str, species_common: str) -> dict | None:
    """Parse one Ensembl variation entry."""
    
    var_id = var.get("id")
    if not var_id:
        return None
        
    conseqs = var.get("clinical_significance", [])
    clinsig = conseqs[0] if conseqs else "Unknown significance"
    
    # Consequence types usually contains things like 'missense_variant'
    types = var.get("consequence_type", [])
    type_str = types[0] if types else ""
    
    if "missense" in type_str or "stop" in type_str or "splice" in type_str:
        variant_type = "SNP"
    elif "deletion" in type_str or "insertion" in type_str or "frameshift" in type_str:
        variant_type = "indel"
    else:
        variant_type = "other"

    alleles = var.get("alleles", [])
    ref = alleles[0] if alleles else ""
    alt = alleles[1] if len(alleles) > 1 else ""
    
    mutation_str = var_id # e.g. rs12345
    if ref and alt:
        mutation_str += f" ({ref}>{alt})"
        
    sci_names = {
        "human": "Homo sapiens",
        "dog": "Canis lupus familiaris",
        "mouse": "Mus musculus"
    }
    species_sci = sci_names.get(species_common.lower(), species_common)

    record = {
        "record_id":              f"ensembl_{var_id}",
        "source_db":              "Ensembl",
        "source_id":              var_id,

        "disease_name":           "Various / Unspecified", 
        "disease_aliases":        [],
        "mim_number":             None,
        "mondo_id":               None,
        "mesh_id":                None,

        "species":                species_sci,
        "species_common":         species_common.lower(),

        "gene":                   gene_symbol,
        "gene_id":                gene_id,
        "variant_type":           variant_type,
        "mutation":               mutation_str,
        "chromosome":             str(var.get("seq_region_name", "")),
        "position":               str(var.get("start", "")),
        "ref_allele":             ref,
        "alt_allele":             alt,
        "zygosity":               None,

        "clinical_significance":  clinsig.capitalize(),
        "inheritance":            "",
        "phenotype":              f"Ensembl variant {var_id} in {gene_symbol} ({type_str.replace('_', ' ')})",
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
    parser = argparse.ArgumentParser(description="Fetch Ensembl variants by gene")
    parser.add_argument("--gene", default="BRCA1", help="Gene symbol to search")
    parser.add_argument("--species", default="human", help="Species common name (human, dog, mouse)")
    parser.add_argument("--max", type=int, default=100, help="Max records to fetch")
    args = parser.parse_args()

    records = fetch_ensembl_variants(args.gene, args.species, args.max)
    slug = f"ensembl_{args.species.lower()}_{args.gene.lower()}"
    
    if records:
        save_records(records, slug)
        print(f"\n✅ Done — {len(records)} variants fetched for {args.gene} in {args.species}")
        print("\nSample record:")
        sample = records[0]
        for k in ["record_id", "gene", "variant_type", "mutation", "clinical_significance", "phenotype"]:
            print(f"  {k}: {sample.get(k)}")
    else:
        print("⚠️  No variants fetched — check gene symbol connectivity")
