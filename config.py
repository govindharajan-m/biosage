"""
config.py — Central configuration and unified disease record schema.
All fetchers import from here so settings stay consistent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))

for d in [RAW_DIR / "clinvar", RAW_DIR / "omim", RAW_DIR / "omia",
          RAW_DIR / "cosmic", RAW_DIR / "disgenet", PROC_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
NCBI_API_KEY      = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL        = os.getenv("NCBI_EMAIL", "")
OMIM_API_KEY      = os.getenv("OMIM_API_KEY", "")
DISGENET_API_KEY  = os.getenv("DISGENET_API_KEY", "")   # optional — get free key at disgenet.org

# ── NCBI E-utilities base URL ─────────────────────────────────────────────────
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── Request settings ──────────────────────────────────────────────────────────
# NCBI allows 10 req/sec with key, 3 without
NCBI_DELAY = 0.11 if NCBI_API_KEY else 0.34   # seconds between requests
REQUEST_TIMEOUT = 30                            # seconds

# ── Model Settings ────────────────────────────────────────────────────────────
# HuggingFace model ID for sentence embeddings
# BioBERT used ~400MB and caused OOM errors, switching to the fast, lightweight
# multi-purpose 22MB MiniLM model which runs smoothly on constrained systems.
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── Groq (free cloud LLM) ─────────────────────────────────────────────────────
# Sign up free at https://console.groq.com → API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Unified Disease Record Schema ─────────────────────────────────────────────
# Every fetcher must return a list of dicts matching this schema.
# Missing optional fields should be None (not omitted).

RECORD_SCHEMA = {
    # --- Identifiers ---
    "record_id":            str,   # Unique: "<source>_<source_id>"  e.g. "clinvar_12345"
    "source_db":            str,   # "ClinVar" | "OMIM" | "OMIA" | "COSMIC" | "DisGeNET" | ...
    "source_id":            str,   # Original ID in the source DB

    # --- Disease ---
    "disease_name":         str,   # Primary name e.g. "Cystic Fibrosis"
    "disease_aliases":      list,  # ["CF", "Mucoviscidosis"]
    "mim_number":           str,   # OMIM MIM number if available (optional)
    "mondo_id":             str,   # MONDO ontology ID (optional)
    "mesh_id":              str,   # MeSH ID (optional)

    # --- Species ---
    "species":              str,   # "Homo sapiens" | "Canis lupus familiaris" | etc.
    "species_common":       str,   # "human" | "dog" | "cattle" | etc.

    # --- Gene / Variant ---
    "gene":                 str,   # Gene symbol e.g. "CFTR"
    "gene_id":              str,   # NCBI Gene ID (optional)
    "variant_type":         str,   # "SNP" | "CNV" | "indel" | "deletion" | "fusion" | "other"
    "mutation":             str,   # e.g. "F508del", "rs113993960", "c.1521_1523delCTT"
    "chromosome":           str,   # "7"
    "position":             str,   # GRCh38 position (optional)
    "ref_allele":           str,   # Reference allele (optional)
    "alt_allele":           str,   # Alternate allele (optional)
    "zygosity":             str,   # "heterozygous" | "homozygous" (optional)

    # --- Clinical ---
    "clinical_significance": str,  # "Pathogenic" | "Likely pathogenic" | "VUS" | "Benign" | ...
    "inheritance":           str,  # "Autosomal dominant" | "Autosomal recessive" | "X-linked" | ...
    "phenotype":             str,  # Free-text clinical description (optional)
    "onset":                 str,  # "childhood" | "adult" | "congenital" (optional)

    # --- Pathway ---
    "pathway":              str,   # e.g. "Ion transport", "DNA repair"
    "pathway_id":           str,   # KEGG / Reactome ID (optional)

    # --- Literature ---
    "pmids":                list,  # List of PubMed IDs ["8298640", "9012406"]

    # --- Text summary ---
    # This is what gets embedded — built from all fields above
    "text_summary":         str,
}


def build_text_summary(record: dict) -> str:
    """
    Convert a normalized record dict into a natural-language paragraph
    suitable for embedding. All fetchers call this before saving.
    """
    parts = []

    disease  = record.get("disease_name", "Unknown disease")
    species  = record.get("species_common") or record.get("species", "")
    gene     = record.get("gene", "")
    mut      = record.get("mutation", "")
    vtype    = record.get("variant_type", "")
    sig      = record.get("clinical_significance", "")
    inh      = record.get("inheritance", "")
    phenotype = record.get("phenotype", "")
    source   = record.get("source_db", "")
    src_id   = record.get("source_id", "")
    pathway  = record.get("pathway", "")

    parts.append(f"Disease: {disease}.")
    if species:
        parts.append(f"Species: {species}.")
    if gene:
        parts.append(f"Gene: {gene}.")
    if vtype and mut:
        parts.append(f"Variant: {vtype} — {mut}.")
    elif mut:
        parts.append(f"Mutation: {mut}.")
    if sig:
        parts.append(f"Clinical significance: {sig}.")
    if inh:
        parts.append(f"Inheritance: {inh}.")
    if pathway:
        parts.append(f"Pathway: {pathway}.")
    if phenotype:
        # Truncate very long phenotype strings
        parts.append(f"Phenotype: {phenotype[:500]}.")
    if source and src_id:
        parts.append(f"Source: {source} ({src_id}).")

    return " ".join(parts)
