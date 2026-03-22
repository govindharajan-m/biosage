"""
fetchers/clinvar_fetcher.py
────────────────────────────
Fetches human disease variant records from NCBI ClinVar via the
E-utilities API, normalizes them to the unified schema, and saves
results as JSON to data/raw/clinvar/.

Usage (from project root):
    python -m data_pipeline.fetchers.clinvar_fetcher
    python -m data_pipeline.fetchers.clinvar_fetcher --disease "breast cancer" --max 200
"""

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# Go up two levels (fetchers → data_pipeline → project root) so config is found
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    NCBI_BASE, NCBI_API_KEY, NCBI_EMAIL,
    NCBI_DELAY, REQUEST_TIMEOUT,
    RAW_DIR, build_text_summary
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR = RAW_DIR / "clinvar"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ncbi_params(extra: dict) -> dict:
    """Inject shared NCBI auth params."""
    p = {"tool": "biosage", "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    p.update(extra)
    return p


def _get(url: str, params: dict) -> requests.Response:
    """GET with retry on transient errors."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning(f"Request failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All retries failed for {url}")


# ── Search ────────────────────────────────────────────────────────────────────

def search_clinvar(disease_name: str, max_results: int = 500) -> list[str]:
    """
    Search ClinVar for a disease and return a list of ClinVar Variation IDs.
    We filter for 'clinsig pathogenic' to focus on disease-causing variants.
    """
    query = f'"{disease_name}"[disease/phenotype] AND ("pathogenic"[clinsig] OR "likely pathogenic"[clinsig])'
    log.info(f"Searching ClinVar: {query!r} (max {max_results})")

    r = _get(f"{NCBI_BASE}/esearch.fcgi", _ncbi_params({
        "db": "clinvar",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }))
    time.sleep(NCBI_DELAY)

    data = r.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    total = data.get("esearchresult", {}).get("count", 0)
    log.info(f"Found {total} total records; retrieving {len(ids)}")
    return ids


# ── Fetch & Parse ─────────────────────────────────────────────────────────────

def fetch_clinvar_records(ids: list[str], batch_size: int = 50) -> list[dict]:
    """Fetch ClinVar XML in batches and parse to normalized records."""
    all_records = []

    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        log.info(f"Fetching batch {i//batch_size + 1}: IDs {batch[0]}…{batch[-1]}")

        r = _get(f"{NCBI_BASE}/efetch.fcgi", _ncbi_params({
            "db": "clinvar",
            "id": ",".join(batch),
            "rettype": "vcv",
            "retmode": "xml",
            "is_variationid": "true",
        }))
        time.sleep(NCBI_DELAY)

        records = _parse_clinvar_xml(r.content)
        all_records.extend(records)
        log.info(f"  → Parsed {len(records)} records (total so far: {len(all_records)})")

    return all_records


def _parse_clinvar_xml(xml_bytes: bytes) -> list[dict]:
    """Parse a ClinVar VCV XML blob into a list of normalized records."""
    records = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log.error(f"XML parse error: {e}")
        return records

    # Each <VariationArchive> is one variant record
    for va in root.iter("VariationArchive"):
        try:
            record = _extract_variation(va)
            if record:
                records.append(record)
        except Exception as e:
            var_id = va.get("VariationID", "?")
            log.warning(f"Failed to parse VariationID={var_id}: {e}")

    return records


def _extract_variation(va: ET.Element) -> dict | None:
    """Extract fields from a <VariationArchive> element."""

    var_id    = va.get("VariationID", "")
    var_name  = va.get("VariationName", "")
    var_type  = va.get("VariationType", "")

    # Map ClinVar variation types to our schema vocabulary
    type_map = {
        "single nucleotide variant": "SNP",
        "snv": "SNP",
        "deletion": "indel",
        "insertion": "indel",
        "indel": "indel",
        "duplication": "CNV",
        "copy number gain": "CNV",
        "copy number loss": "CNV",
        "inversion": "CNV",
        "microsatellite": "other",
        "fusion": "fusion",
    }
    variant_type = type_map.get(var_type.lower(), "other")

    # ── Gene ──────────────────────────────────────────────────────────────────
    gene_el = va.find(".//Gene")
    gene_symbol = gene_el.get("Symbol", "") if gene_el is not None else ""
    gene_id     = gene_el.get("GeneID", "")  if gene_el is not None else ""

    # ── Chromosome / Position ─────────────────────────────────────────────────
    loc = va.find(".//SequenceLocation[@Assembly='GRCh38']")
    chromosome = loc.get("Chr", "")     if loc is not None else ""
    position   = loc.get("start", "")  if loc is not None else ""

    # ── Clinical Significance ─────────────────────────────────────────────────
    # Use the aggregate (most severe) classification
    agg = va.find(".//AggregateClassification")
    if agg is None:
        agg = va.find(".//GermlineClassification")
    clinsig = ""
    if agg is not None:
        desc = agg.find("Description")
        clinsig = desc.text.strip() if desc is not None else ""

    # ── Disease / Phenotype ───────────────────────────────────────────────────
    disease_name = ""
    aliases: list[str] = []
    mim_number = ""

    trait_set = va.find(".//TraitSet[@Type='Disease']")
    if trait_set is not None:
        trait = trait_set.find("Trait")
        if trait is not None:
            # Primary name
            for name_el in trait.findall("Name/ElementValue"):
                if name_el.get("Type") == "Preferred":
                    disease_name = name_el.text or ""
                elif name_el.get("Type") == "Alternate":
                    if name_el.text:
                        aliases.append(name_el.text)
            # OMIM cross-reference
            for xref in trait.findall("XRef"):
                if xref.get("DB") == "OMIM":
                    mim_number = xref.get("ID", "")

    if not disease_name:
        return None   # Skip records with no disease association

    # ── Inheritance ───────────────────────────────────────────────────────────
    inheritance = ""
    for attr in va.findall(".//AttributeSet/Attribute[@Type='ModeOfInheritance']"):
        inheritance = attr.text or ""
        break

    # ── PMIDs ─────────────────────────────────────────────────────────────────
    pmids = [
        citation.find("ID").text
        for citation in va.findall(".//Citation")
        if citation.find("ID") is not None and citation.find("ID").get("Source") == "PubMed"
    ]

    # ── Assemble ──────────────────────────────────────────────────────────────
    record = {
        "record_id":              f"clinvar_{var_id}",
        "source_db":              "ClinVar",
        "source_id":              var_id,

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
        "mutation":               var_name,
        "chromosome":             chromosome,
        "position":               position,
        "ref_allele":             None,
        "alt_allele":             None,
        "zygosity":               None,

        "clinical_significance":  clinsig,
        "inheritance":            inheritance,
        "phenotype":              None,
        "onset":                  None,

        "pathway":                None,
        "pathway_id":             None,

        "pmids":                  list(set(pmids)),
        "text_summary":           "",
    }

    record["text_summary"] = build_text_summary(record)
    return record


# ── Save ──────────────────────────────────────────────────────────────────────

def save_records(records: list[dict], disease_slug: str):
    """Save normalized records to JSON."""
    out_path = SAVE_DIR / f"{disease_slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(records)} records → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_disease(disease_name: str, max_results: int = 500) -> list[dict]:
    """Full pipeline: search → fetch → parse → save."""
    slug = disease_name.lower().replace(" ", "_").replace("/", "_")
    ids  = search_clinvar(disease_name, max_results)
    if not ids:
        log.warning("No results found.")
        return []
    records = fetch_clinvar_records(ids)
    save_records(records, slug)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ClinVar records for a disease")
    parser.add_argument("--disease", default="cystic fibrosis", help="Disease name to search")
    parser.add_argument("--max",     type=int, default=200, help="Max records to fetch")
    args = parser.parse_args()

    records = fetch_disease(args.disease, args.max)
    print(f"\n✅ Done — {len(records)} records fetched for '{args.disease}'")
    if records:
        print("\nSample record:")
        sample = records[0]
        for k in ["record_id", "disease_name", "gene", "variant_type", "mutation", "clinical_significance"]:
            print(f"  {k}: {sample.get(k)}")
        print(f"  text_summary: {sample.get('text_summary', '')[:120]}...")
