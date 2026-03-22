"""
bulk_fetch.py
─────────────
Fetches ClinVar records for a large list of genetic diseases,
then rebuilds the full pipeline: normalizer → chunker → vector store.

Usage (from project root):
    python bulk_fetch.py
    python bulk_fetch.py --max 200   # records per disease (default 500)
    python bulk_fetch.py --skip-embed  # just fetch + normalize, skip embedding
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Disease list ───────────────────────────────────────────────────────────────
# ClinVar has strong coverage for all of these.
DISEASES = [
    # Cancer / tumor suppressors
    "BRCA1 breast cancer",
    "BRCA2 breast cancer",
    "Lynch syndrome",
    "Li-Fraumeni syndrome",
    "familial adenomatous polyposis",
    "hereditary diffuse gastric cancer",
    "Von Hippel-Lindau disease",
    "retinoblastoma",
    "Cowden syndrome",
    "Gorlin syndrome",

    # Neurological / neurodegenerative
    "Huntington disease",
    "Parkinson disease",
    "Alzheimer disease",
    "amyotrophic lateral sclerosis",
    "spinal muscular atrophy",
    "Charcot-Marie-Tooth disease",
    "Friedreich ataxia",
    "tuberous sclerosis",
    "neurofibromatosis type 1",
    "neurofibromatosis type 2",
    "Rett syndrome",
    "Angelman syndrome",
    "Prader-Willi syndrome",

    # Cardiac / vascular
    "hypertrophic cardiomyopathy",
    "dilated cardiomyopathy",
    "long QT syndrome",
    "Brugada syndrome",
    "Marfan syndrome",
    "Loeys-Dietz syndrome",
    "familial hypercholesterolemia",
    "arrhythmogenic right ventricular cardiomyopathy",

    # Respiratory / pulmonary
    "cystic fibrosis",
    "alpha-1 antitrypsin deficiency",
    "primary ciliary dyskinesia",
    "pulmonary arterial hypertension",

    # Metabolic / lysosomal
    "phenylketonuria",
    "Gaucher disease",
    "Fabry disease",
    "Pompe disease",
    "Niemann-Pick disease",
    "maple syrup urine disease",
    "galactosemia",
    "glycogen storage disease",
    "homocystinuria",
    "tyrosinemia",
    "mucopolysaccharidosis",

    # Hematological
    "sickle cell disease",
    "beta thalassemia",
    "hemophilia A",
    "hemophilia B",
    "hereditary spherocytosis",
    "glucose-6-phosphate dehydrogenase deficiency",
    "von Willebrand disease",
    "Diamond-Blackfan anemia",
    "Fanconi anemia",

    # Muscular / connective tissue
    "Duchenne muscular dystrophy",
    "Becker muscular dystrophy",
    "myotonic dystrophy",
    "Ehlers-Danlos syndrome",
    "osteogenesis imperfecta",
    "achondroplasia",

    # Renal / urological
    "autosomal dominant polycystic kidney disease",
    "autosomal recessive polycystic kidney disease",
    "Alport syndrome",
    "Bartter syndrome",

    # Immune / inflammatory
    "severe combined immunodeficiency",
    "chronic granulomatous disease",
    "Wiskott-Aldrich syndrome",
    "familial Mediterranean fever",
    "autoinflammatory syndrome",

    # Ocular
    "Leber congenital amaurosis",
    "retinitis pigmentosa",
    "Stargardt disease",
    "choroideremia",

    # Endocrine
    "congenital adrenal hyperplasia",
    "multiple endocrine neoplasia type 1",
    "multiple endocrine neoplasia type 2",
    "maturity onset diabetes of the young",

    # Other rare / syndromic
    "Fragile X syndrome",
    "Down syndrome",
    "Noonan syndrome",
    "CHARGE syndrome",
    "DiGeorge syndrome",
    "Smith-Lemli-Opitz syndrome",
    "Wilson disease",
    "Menkes disease",
    "incontinentia pigmenti",
    "Bardet-Biedl syndrome",
    "Usher syndrome",
    "Waardenburg syndrome",
    "Pendred syndrome",
    "congenital hypothyroidism",
    "Kallmann syndrome",
]


def run_bulk_fetch(max_per_disease: int = 500):
    from data_pipeline.fetchers.clinvar_fetcher import fetch_disease

    log.info(f"Starting bulk fetch for {len(DISEASES)} diseases, up to {max_per_disease} records each.")
    total_fetched = 0
    failed = []

    for i, disease in enumerate(DISEASES, 1):
        log.info(f"[{i}/{len(DISEASES)}] Fetching: {disease}")
        try:
            records = fetch_disease(disease, max_results=max_per_disease)
            total_fetched += len(records)
            log.info(f"  → {len(records)} records")
        except Exception as e:
            log.error(f"  → FAILED: {e}")
            failed.append(disease)
        time.sleep(0.5)  # be polite to NCBI

    log.info(f"\nBulk fetch complete. Total records fetched: {total_fetched}")
    if failed:
        log.warning(f"Failed diseases ({len(failed)}): {failed}")
    return total_fetched


def run_pipeline():
    log.info("Step 1/3: Running normalizer...")
    from data_pipeline.normalizer import normalize_all
    normalize_all()

    log.info("Step 2/3: Running chunker...")
    from data_pipeline.chunker import create_chunks
    create_chunks()

    log.info("Step 3/3: Embedding into ChromaDB...")
    from data_pipeline.vector_store import ChromaStore
    store = ChromaStore()
    store.populate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk fetch disease data and rebuild vector DB")
    parser.add_argument("--max", type=int, default=500, help="Max ClinVar records per disease (default: 500)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching, just rebuild the pipeline")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    args = parser.parse_args()

    if not args.skip_fetch:
        total = run_bulk_fetch(args.max)
        print(f"\n{'='*60}")
        print(f"  Fetch complete: {total} total records across {len(DISEASES)} diseases")
        print(f"{'='*60}\n")

    if not args.skip_embed:
        run_pipeline()
    else:
        log.info("Step 1/2: Running normalizer...")
        from data_pipeline.normalizer import normalize_all
        normalize_all()
        log.info("Step 2/2: Running chunker...")
        from data_pipeline.chunker import create_chunks
        create_chunks()
        log.info("Skipped embedding (--skip-embed). Run vector_store.py manually to embed.")
