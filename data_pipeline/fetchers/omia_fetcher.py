"""
fetchers/omia_fetcher.py
─────────────────────────
Fetches animal genetic disease records from OMIA (Online Mendelian
Inheritance in Animals) — covers 200+ animal species including dogs,
cats, cattle, horses, sheep, pigs, chickens, and more.

OMIA provides a public JSON API and bulk download.
API docs: https://omia.org/api/

Usage:
    python -m data_pipeline.fetchers.omia_fetcher
    python -m data_pipeline.fetchers.omia_fetcher --species dog
    python -m data_pipeline.fetchers.omia_fetcher --disease "hip dysplasia"
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

SAVE_DIR = RAW_DIR / "omia"
OMIA_API = "https://omia.org/api/v1"

# ── Species name mapping ──────────────────────────────────────────────────────
# OMIA species IDs → common + scientific names
SPECIES_MAP = {
    "9615":  ("dog",      "Canis lupus familiaris"),
    "9685":  ("cat",      "Felis catus"),
    "9913":  ("cattle",   "Bos taurus"),
    "9796":  ("horse",    "Equus caballus"),
    "9823":  ("pig",      "Sus scrofa"),
    "9940":  ("sheep",    "Ovis aries"),
    "9031":  ("chicken",  "Gallus gallus domesticus"),
    "9838":  ("camel",    "Camelus dromedarius"),
    "9986":  ("rabbit",   "Oryctolagus cuniculus"),
    "10090": ("mouse",    "Mus musculus"),
    "10116": ("rat",      "Rattus norvegicus"),
    "9544":  ("macaque",  "Macaca mulatta"),
    "9598":  ("chimpanzee", "Pan troglodytes"),
}

# Reverse map: common name → taxon ID
COMMON_TO_TAXON = {v[0]: k for k, v in SPECIES_MAP.items()}


# ── API Helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = {}) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(f"{OMIA_API}/{endpoint}",
                             params=params, timeout=REQUEST_TIMEOUT,
                             headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning(f"OMIA request failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All retries failed for OMIA /{endpoint}")


# ── Fetch by Species ──────────────────────────────────────────────────────────

def fetch_by_species(species_common: str, max_records: int = 500) -> list[dict]:
    """
    Fetch all genetic disease entries for a given species.
    e.g. species_common = "dog"
    """
    taxon_id = COMMON_TO_TAXON.get(species_common.lower())
    if not taxon_id:
        log.error(f"Unknown species: {species_common!r}")
        log.info(f"Available: {list(COMMON_TO_TAXON.keys())}")
        return []

    species_sci = SPECIES_MAP[taxon_id][1]
    log.info(f"Fetching OMIA records for {species_common} ({species_sci}, taxon {taxon_id})")

    try:
        data = _get("phene", {"species_id": taxon_id, "page_size": min(max_records, 500)})
    except RuntimeError as e:
        log.error(f"OMIA API unavailable: {e}")
        log.info("Falling back to OMIA bulk JSON download approach...")
        return _fetch_bulk_fallback(taxon_id, species_common, species_sci)

    results = data.get("results", [])
    log.info(f"Retrieved {len(results)} OMIA phene records")

    records = []
    for phene in results:
        record = _parse_phene(phene, species_common, species_sci)
        if record:
            records.append(record)

    return records


def fetch_by_disease_name(disease_name: str, max_records: int = 200) -> list[dict]:
    """
    Search OMIA for a disease by name across all species.
    """
    log.info(f"Searching OMIA for disease: {disease_name!r}")
    try:
        data = _get("phene", {"phene_name": disease_name, "page_size": max_records})
    except RuntimeError as e:
        log.error(f"OMIA API error: {e}")
        return []

    results = data.get("results", [])
    log.info(f"Found {len(results)} matching phene records")

    records = []
    for phene in results:
        taxon_id = str(phene.get("species_id", ""))
        species_info = SPECIES_MAP.get(taxon_id, ("unknown", "Unknown species"))
        record = _parse_phene(phene, species_info[0], species_info[1])
        if record:
            records.append(record)

    return records


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_phene(phene: dict, species_common: str, species_sci: str) -> dict | None:
    """Parse one OMIA 'phene' (phenotype/disease) entry."""

    omia_id      = str(phene.get("omia_id", phene.get("id", "")))
    disease_name = phene.get("phene_name", "")
    if not disease_name or not omia_id:
        return None

    gene_symbol  = phene.get("gene_symbol", "") or ""
    inheritance  = phene.get("inheritance", "") or ""
    variant_type = _map_variant_type(phene.get("variant_class", ""))
    mutation     = phene.get("variant_name", "") or phene.get("molecular_basis", "") or ""
    chromosome   = str(phene.get("chromosome", "")) if phene.get("chromosome") else ""
    summary      = phene.get("summary", "") or ""
    pmids        = [str(p) for p in phene.get("pubmed_ids", []) if p]

    # Map OMIA inheritance terminology
    inh_map = {
        "autosomal recessive":   "Autosomal recessive",
        "autosomal dominant":    "Autosomal dominant",
        "x-linked":              "X-linked",
        "x-linked recessive":    "X-linked recessive",
        "x-linked dominant":     "X-linked dominant",
        "y-linked":              "Y-linked",
        "mitochondrial":         "Mitochondrial",
        "polygenic":             "Polygenic",
        "multifactorial":        "Multifactorial",
    }
    inheritance = inh_map.get(inheritance.lower(), inheritance)

    record = {
        "record_id":              f"omia_{omia_id}",
        "source_db":              "OMIA",
        "source_id":              omia_id,

        "disease_name":           disease_name,
        "disease_aliases":        [],
        "mim_number":             None,
        "mondo_id":               None,
        "mesh_id":                None,

        "species":                species_sci,
        "species_common":         species_common,

        "gene":                   gene_symbol,
        "gene_id":                None,
        "variant_type":           variant_type,
        "mutation":               mutation,
        "chromosome":             chromosome,
        "position":               None,
        "ref_allele":             None,
        "alt_allele":             None,
        "zygosity":               None,

        "clinical_significance":  "Pathogenic" if gene_symbol else "Associated",
        "inheritance":            inheritance,
        "phenotype":              summary[:600] if summary else None,
        "onset":                  None,

        "pathway":                None,
        "pathway_id":             None,

        "pmids":                  pmids[:10],
        "text_summary":           "",
    }

    record["text_summary"] = build_text_summary(record)
    return record


def _map_variant_type(raw: str) -> str:
    raw = (raw or "").lower()
    if "snp" in raw or "substitut" in raw or "missense" in raw or "nonsense" in raw:
        return "SNP"
    if "deletion" in raw or "insertion" in raw or "indel" in raw:
        return "indel"
    if "copy number" in raw or "cnv" in raw or "duplication" in raw:
        return "CNV"
    if "splice" in raw:
        return "SNP"
    return "other" if raw else None


# ── Fallback: OMIA bulk JSON ──────────────────────────────────────────────────

def _fetch_bulk_fallback(taxon_id: str, species_common: str, species_sci: str) -> list[dict]:
    """
    If the OMIA REST API is unavailable, attempt to download the public
    bulk JSON export from GitHub mirror (updated monthly).
    """
    BULK_URL = "https://raw.githubusercontent.com/monarch-initiative/omia-ingest/main/data/omia.json"
    log.info(f"Attempting bulk OMIA download from: {BULK_URL}")
    try:
        r = requests.get(BULK_URL, timeout=60)
        r.raise_for_status()
        all_phenes = r.json()
        # Filter to this species
        filtered = [p for p in all_phenes if str(p.get("species_id")) == taxon_id]
        log.info(f"Bulk: found {len(filtered)} records for {species_common}")
        records = []
        for phene in filtered:
            record = _parse_phene(phene, species_common, species_sci)
            if record:
                records.append(record)
        return records
    except Exception as e:
        log.error(f"Bulk download also failed: {e}")
        return []


# ── Save ──────────────────────────────────────────────────────────────────────

def save_records(records: list[dict], slug: str):
    out_path = SAVE_DIR / f"{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(records)} records → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OMIA animal disease records")
    parser.add_argument("--species", default="dog",
                        help=f"Species common name. Options: {list(COMMON_TO_TAXON.keys())}")
    parser.add_argument("--disease", default=None,
                        help="Search by disease name instead of species")
    parser.add_argument("--max", type=int, default=300,
                        help="Max records to fetch")
    args = parser.parse_args()

    if args.disease:
        records = fetch_by_disease_name(args.disease, args.max)
        slug = args.disease.lower().replace(" ", "_")
    else:
        records = fetch_by_species(args.species, args.max)
        slug = args.species.lower()

    if records:
        save_records(records, f"omia_{slug}")
        print(f"\n✅ Done — {len(records)} animal disease records fetched")
        print("\nSample record:")
        sample = records[0]
        for k in ["record_id", "disease_name", "species_common", "gene",
                  "variant_type", "inheritance"]:
            print(f"  {k}: {sample.get(k)}")
        print(f"  text_summary: {sample.get('text_summary', '')[:120]}...")
    else:
        print("⚠️  No records fetched — check API connectivity and species name")
