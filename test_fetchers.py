"""
test_fetchers.py
─────────────────
Quick smoke-test for ClinVar, OMIM, and OMIA fetchers.
Run this first to confirm your API keys work and data is flowing.

    python test_fetchers.py

Each test fetches a small number of records and prints a summary.
Full data collection uses larger --max values in the individual fetchers.
"""

import json
import sys
from pathlib import Path

# Fix Unicode/emoji output on Windows (cp1252 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Make sure project root is on the path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import NCBI_API_KEY, OMIM_API_KEY, ANTHROPIC_API_KEY, RAW_DIR


def sep(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)


def check_keys():
    sep("0 · API Key Check")
    keys = {
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "NCBI_API_KEY":      NCBI_API_KEY,
        "OMIM_API_KEY":      OMIM_API_KEY,
    }
    all_ok = True
    for name, val in keys.items():
        if val and not val.startswith("your_"):
            print(f"  ✅ {name} — SET")
        else:
            print(f"  ⚠️  {name} — NOT SET (add to .env)")
            if name in ("NCBI_API_KEY",):
                print(f"     → ClinVar/OMIA will still work, just slower (3 req/sec)")
            if name == "OMIM_API_KEY":
                print(f"     → OMIM test will be skipped")
                all_ok = False
    return all_ok


def test_clinvar():
    sep("1 · ClinVar Fetcher — Cystic Fibrosis (20 records)")
    from data_pipeline.fetchers.clinvar_fetcher import fetch_disease
    try:
        records = fetch_disease("cystic fibrosis", max_results=20)
        if records:
            print(f"\n  ✅ Fetched {len(records)} ClinVar records")
            r = records[0]
            print(f"  disease_name : {r['disease_name']}")
            print(f"  gene         : {r['gene']}")
            print(f"  variant_type : {r['variant_type']}")
            print(f"  mutation     : {r['mutation'][:60]}")
            print(f"  clinsig      : {r['clinical_significance']}")
            print(f"  text_summary : {r['text_summary'][:100]}...")
        else:
            print("  ⚠️  No records returned — check NCBI connectivity")
    except Exception as e:
        print(f"  ❌ ClinVar test failed: {e}")


def test_omim():
    sep("2 · OMIM Fetcher — Cystic Fibrosis (5 entries)")
    if not OMIM_API_KEY or OMIM_API_KEY.startswith("your_"):
        print("  ⏭️  Skipped — OMIM_API_KEY not set")
        print("  → Register at https://www.omim.org/api to enable this")
        return
    from data_pipeline.fetchers.omim_fetcher import fetch_disease
    try:
        records = fetch_disease("cystic fibrosis", max_results=5)
        if records:
            print(f"\n  ✅ Fetched {len(records)} OMIM records")
            r = records[0]
            print(f"  disease_name : {r['disease_name']}")
            print(f"  mim_number   : {r['mim_number']}")
            print(f"  gene         : {r['gene']}")
            print(f"  inheritance  : {r['inheritance']}")
            print(f"  phenotype    : {str(r.get('phenotype',''))[:100]}...")
        else:
            print("  ⚠️  No records returned")
    except Exception as e:
        print(f"  ❌ OMIM test failed: {e}")


def test_omia():
    sep("3 · OMIA Fetcher — Dog diseases (20 records)")
    from data_pipeline.fetchers.omia_fetcher import fetch_by_species
    try:
        records = fetch_by_species("dog", max_records=20)
        if records:
            print(f"\n  ✅ Fetched {len(records)} OMIA records")
            r = records[0]
            print(f"  disease_name   : {r['disease_name']}")
            print(f"  species_common : {r['species_common']}")
            print(f"  species        : {r['species']}")
            print(f"  gene           : {r['gene']}")
            print(f"  inheritance    : {r['inheritance']}")
            print(f"  text_summary   : {r['text_summary'][:100]}...")
        else:
            print("  ⚠️  No records — OMIA API may be temporarily down, check connectivity")
    except Exception as e:
        print(f"  ❌ OMIA test failed: {e}")


def test_cosmic():
    sep("4 · COSMIC Fetcher — BRCA1 Mutation (5 records)")
    from data_pipeline.fetchers.cosmic_fetcher import fetch_cosmic_mutations
    try:
        records = fetch_cosmic_mutations("BRCA1", "Breast Cancer", max_results=5)
        if records:
            print(f"\n  ✅ Fetched {len(records)} COSMIC records")
            r = records[0]
            print(f"  disease_name : {r['disease_name']}")
            print(f"  gene         : {r['gene']}")
            print(f"  mutation     : {r['mutation']}")
            print(f"  clinsig      : {r['clinical_significance']}")
            print(f"  text_summary : {r['text_summary'][:100]}...")
        else:
            print("  ⚠️  No records returned")
    except Exception as e:
        print(f"  ❌ COSMIC test failed: {e}")


def test_disgenet():
    sep("5 · DisGeNET Fetcher — Asthma (5 associations)")
    from data_pipeline.fetchers.disgenet_fetcher import fetch_disgenet_associations
    try:
        records = fetch_disgenet_associations("asthma", max_results=5)
        if records:
            print(f"\n  ✅ Fetched {len(records)} DisGeNET records")
            r = records[0]
            print(f"  disease_name : {r['disease_name']}")
            print(f"  gene         : {r['gene']}")
            print(f"  clinsig      : {r['clinical_significance']}")
            print(f"  phenotype    : {r['phenotype']}")
            print(f"  text_summary : {r['text_summary'][:100]}...")
        else:
            print("  ⚠️  No records returned")
    except Exception as e:
        print(f"  ❌ DisGeNET test failed: {e}")


def test_ensembl():
    sep("6 · Ensembl Fetcher — Human BRCA1 (5 variants)")
    from data_pipeline.fetchers.ensembl_fetcher import fetch_ensembl_variants
    try:
        records = fetch_ensembl_variants("BRCA1", "human", max_results=5)
        if records:
            print(f"\n  ✅ Fetched {len(records)} Ensembl records")
            r = records[0]
            print(f"  species      : {r['species_common']}")
            print(f"  gene         : {r['gene']}")
            print(f"  mutation     : {r['mutation']}")
            print(f"  position     : {r['chromosome']}:{r['position']}")
            print(f"  clinsig      : {r['clinical_significance']}")
            print(f"  text_summary : {r['text_summary'][:100]}...")
        else:
            print("  ⚠️  No records returned")
    except Exception as e:
        print(f"  ❌ Ensembl test failed: {e}")


def test_saved_files():
    sep("7 · Saved File Check")
    import os
    for db in ["clinvar", "omim", "omia", "cosmic", "disgenet", "ensembl"]:
        folder = RAW_DIR / db
        files = list(folder.glob("*.json")) if folder.exists() else []
        if files:
            total = 0
            for f in files:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                    total += len(data)
            print(f"  ✅ {db:10s}: {len(files)} file(s), {total} records")
        else:
            print(f"  ⚠️  {db:10s}: no files yet")


if __name__ == "__main__":
    print("\n🧬 BioSage — Fetcher Tests")
    check_keys()
    test_clinvar()
    test_omim()
    test_omia()
    test_cosmic()
    test_disgenet()
    test_ensembl()
    test_saved_files()
    print(f"\n{'═'*60}")
    print("  Done! Check data/raw/ for saved JSON files.")
    print('═'*60 + "\n")
