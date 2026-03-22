"""
fetchers/omim_fetcher.py
─────────────────────────
Fetches human genetic disease entries from OMIM (Online Mendelian
Inheritance in Man) via their API.

OMIM provides rich disease summaries, gene links, inheritance patterns,
and allelic variant (mutation) listings — ideal for disease descriptions.

Usage:
    python -m data_pipeline.fetchers.omim_fetcher
    python -m data_pipeline.fetchers.omim_fetcher --disease "diabetes mellitus type 2" --max 50
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    OMIM_API_KEY, NCBI_BASE, NCBI_API_KEY, NCBI_EMAIL,
    NCBI_DELAY, REQUEST_TIMEOUT,
    RAW_DIR, build_text_summary
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR     = RAW_DIR / "omim"
OMIM_API_URL = "https://api.omim.org/api"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _omim_get(endpoint: str, params: dict) -> dict:
    """GET from OMIM API with retry."""
    params["apiKey"] = OMIM_API_KEY
    params["format"] = "json"
    for attempt in range(3):
        try:
            r = requests.get(f"{OMIM_API_URL}/{endpoint}",
                             params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning(f"OMIM request failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All retries failed for OMIM /{endpoint}")


# ── Search via NCBI Gene ───────────────────────────────────────────────────────
# OMIM search API is limited; we use NCBI to find MIM numbers first.

def _ncbi_params(extra: dict) -> dict:
    p = {"tool": "biosage", "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    p.update(extra)
    return p


def search_omim_mim_numbers(disease_name: str, max_results: int = 50) -> list[str]:
    """
    Use NCBI E-utilities to find OMIM MIM numbers for a disease.
    Returns a list of MIM number strings e.g. ['219700', '602421'].
    """
    log.info(f"Searching OMIM via NCBI for: {disease_name!r}")

    r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=_ncbi_params({
        "db": "omim",
        "term": f'"{disease_name}"[disease name]',
        "retmax": max_results,
        "retmode": "json",
    }), timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    time.sleep(NCBI_DELAY)

    ids = r.json().get("esearchresult", {}).get("idlist", [])
    log.info(f"Found {len(ids)} MIM numbers via NCBI")
    return ids


# ── Fetch OMIM Entries ────────────────────────────────────────────────────────

def fetch_omim_entries(mim_numbers: list[str]) -> list[dict]:
    """
    Fetch full OMIM entries for a list of MIM numbers.
    OMIM API allows up to 20 entries per request.
    """
    if not OMIM_API_KEY:
        log.error("OMIM_API_KEY not set in .env — cannot fetch OMIM data.")
        log.error("Register for a free research key at: https://www.omim.org/api")
        return []

    all_records = []
    batch_size = 20

    for i in range(0, len(mim_numbers), batch_size):
        batch = mim_numbers[i : i + batch_size]
        log.info(f"Fetching OMIM batch {i//batch_size + 1}: {len(batch)} entries")

        try:
            data = _omim_get("entry", {
                "mimNumber": ",".join(batch),
                "include": "text,allelicVariantList,geneMap,externalLinks",
            })
            time.sleep(0.5)   # OMIM is more sensitive to rate limits

            entries = (data.get("omim", {})
                           .get("entryList", []))

            for entry_wrap in entries:
                entry = entry_wrap.get("entry", {})
                record = _parse_omim_entry(entry)
                if record:
                    all_records.extend(record)

        except Exception as e:
            log.error(f"Error fetching OMIM batch: {e}")

    return all_records


def _parse_omim_entry(entry: dict) -> list[dict]:
    """
    Parse one OMIM entry into one or more normalized records.
    Each allelic variant becomes a separate record.
    If no allelic variants, we create one record from the entry itself.
    """
    mim_number   = str(entry.get("mimNumber", ""))
    title        = entry.get("titles", {}).get("preferredTitle", "")
    # OMIM titles look like "CYSTIC FIBROSIS; CF" — clean them up
    disease_name = title.split(";")[0].strip().title()
    aliases      = [a.strip().title() for a in title.split(";")[1:]] if ";" in title else []

    # Inheritance from prefix symbol
    prefix = entry.get("prefix", "")
    inheritance_map = {
        "*": "Gene only",
        "#": "Autosomal dominant",
        "%": "Unknown mechanism",
        "+": "Gene with phenotype",
        "^": "Moved/removed",
    }
    inheritance_hint = inheritance_map.get(prefix, "")

    # Gene map
    gene_map   = entry.get("geneMap", {})
    gene_symbol = gene_map.get("geneSymbols", "").split(",")[0].strip()
    gene_id    = str(gene_map.get("geneId", ""))
    chromosome = str(gene_map.get("chromosome", ""))

    # Text summary from OMIM text sections
    text_sections = entry.get("textSectionList", [])
    phenotype_text = ""
    for section in text_sections:
        s = section.get("textSection", {})
        if s.get("textSectionName") in ("description", "clinicalFeatures"):
            raw = s.get("textSectionContent", "")
            # Strip OMIM markup like {1234} and \n
            phenotype_text = (raw.replace("\n", " ")
                                  .replace("\\n", " ")
                                  [:600])
            break

    # PMIDs from references
    pmids = []
    for ref in entry.get("referenceList", []):
        pmid = ref.get("reference", {}).get("pubmedID")
        if pmid:
            pmids.append(str(pmid))

    # ── Allelic Variants (individual mutations) ────────────────────────────────
    allelic_variants = entry.get("allelicVariantList", [])
    records = []

    if allelic_variants:
        for av_wrap in allelic_variants:
            av = av_wrap.get("allelicVariant", {})
            mutation_name = av.get("name", "")
            dbsnp_id      = av.get("dbsnpId", "")
            mutation_text = av.get("text", "")[:400]

            variant_type = "SNP" if (dbsnp_id.startswith("rs") or
                                      "substitut" in mutation_text.lower()) else "indel"

            record = {
                "record_id":              f"omim_{mim_number}_{av.get('number', '')}",
                "source_db":              "OMIM",
                "source_id":              mim_number,

                "disease_name":           disease_name,
                "disease_aliases":        aliases,
                "mim_number":             mim_number,
                "mondo_id":               None,
                "mesh_id":                None,

                "species":                "Homo sapiens",
                "species_common":         "human",

                "gene":                   gene_symbol,
                "gene_id":                gene_id,
                "variant_type":           variant_type,
                "mutation":               mutation_name,
                "chromosome":             chromosome,
                "position":               None,
                "ref_allele":             None,
                "alt_allele":             None,
                "zygosity":               None,

                "clinical_significance":  "Pathogenic",
                "inheritance":            inheritance_hint,
                "phenotype":              mutation_text or phenotype_text,
                "onset":                  None,

                "pathway":                None,
                "pathway_id":             None,

                "pmids":                  pmids[:10],
                "text_summary":           "",
            }
            record["text_summary"] = build_text_summary(record)
            records.append(record)

    else:
        # No allelic variants — create one disease-level record
        record = {
            "record_id":              f"omim_{mim_number}",
            "source_db":              "OMIM",
            "source_id":              mim_number,

            "disease_name":           disease_name,
            "disease_aliases":        aliases,
            "mim_number":             mim_number,
            "mondo_id":               None,
            "mesh_id":                None,

            "species":                "Homo sapiens",
            "species_common":         "human",

            "gene":                   gene_symbol,
            "gene_id":                gene_id,
            "variant_type":           None,
            "mutation":               None,
            "chromosome":             chromosome,
            "position":               None,
            "ref_allele":             None,
            "alt_allele":             None,
            "zygosity":               None,

            "clinical_significance":  None,
            "inheritance":            inheritance_hint,
            "phenotype":              phenotype_text,
            "onset":                  None,

            "pathway":                None,
            "pathway_id":             None,

            "pmids":                  pmids[:10],
            "text_summary":           "",
        }
        record["text_summary"] = build_text_summary(record)
        records.append(record)

    return records


# ── Save ──────────────────────────────────────────────────────────────────────

def save_records(records: list[dict], slug: str):
    out_path = SAVE_DIR / f"{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(records)} records → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_disease(disease_name: str, max_results: int = 50) -> list[dict]:
    slug = disease_name.lower().replace(" ", "_").replace("/", "_")
    mims = search_omim_mim_numbers(disease_name, max_results)
    if not mims:
        log.warning("No MIM numbers found.")
        return []
    records = fetch_omim_entries(mims)
    if records:
        save_records(records, slug)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OMIM records for a disease")
    parser.add_argument("--disease", default="cystic fibrosis", help="Disease name")
    parser.add_argument("--max",     type=int, default=20, help="Max MIM entries to fetch")
    args = parser.parse_args()

    records = fetch_disease(args.disease, args.max)
    print(f"\n✅ Done — {len(records)} records fetched for '{args.disease}'")
    if records:
        print("\nSample record:")
        sample = records[0]
        for k in ["record_id", "disease_name", "gene", "mutation", "inheritance", "phenotype"]:
            v = str(sample.get(k, ""))
            print(f"  {k}: {v[:100]}")
