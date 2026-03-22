"""
Query normalization and fast-path intent classifier for BioSage.

Two responsibilities:
  normalize_query(query) — expands common disease abbreviations / misspellings
                           to canonical database-friendly names.
  fast_classify(query)   — keyword/regex-based intent classifier that avoids an
                           LLM round-trip for unambiguous queries.
                           Returns one of: disease | variant | gene | drug | general
                           Returns None when the query is ambiguous (use LLM).
"""

import difflib
import re

# ── Synonym / alias map ───────────────────────────────────────────────────────
# canonical name → list of known aliases (lower-case)
_SYNONYMS: dict[str, list[str]] = {
    "cystic fibrosis":                  ["cf", "mucoviscidosis"],
    "alzheimer disease":                ["alzheimers", "alzheimer's", "ad dementia",
                                         "alzheimer's disease"],
    "parkinson disease":                ["parkinsons", "parkinson's", "pd",
                                         "parkinson's disease"],
    "huntington disease":               ["huntingtons", "huntington's",
                                         "huntington's disease", "hd"],
    "sickle cell disease":              ["sickle cell anemia", "scd", "hbs disease",
                                         "sickle-cell disease"],
    "type 2 diabetes":                  ["t2d", "t2dm", "diabetes type 2",
                                         "type 2 diabetes mellitus",
                                         "type ii diabetes"],
    "type 1 diabetes":                  ["t1d", "t1dm", "diabetes type 1",
                                         "type 1 diabetes mellitus",
                                         "juvenile diabetes"],
    "amyotrophic lateral sclerosis":    ["als", "lou gehrig disease",
                                         "lou gehrig's disease",
                                         "motor neurone disease"],
    "phenylketonuria":                  ["pku"],
    "spinal muscular atrophy":          ["sma"],
    "duchenne muscular dystrophy":      ["dmd"],
    "tuberous sclerosis complex":       ["tsc", "tuberous sclerosis"],
    "marfan syndrome":                  ["marfan's syndrome", "marfans"],
    "down syndrome":                    ["trisomy 21", "t21", "down's syndrome"],
    "fragile x syndrome":               ["fra x", "fraxa", "fragile x"],
    "multiple sclerosis":               ["ms"],
    "rheumatoid arthritis":             ["ra"],
    "systemic lupus erythematosus":     ["sle", "lupus"],
    "hemophilia a":                     ["factor viii deficiency", "haemophilia a"],
    "hemophilia b":                     ["factor ix deficiency", "christmas disease",
                                         "haemophilia b"],
    "colorectal cancer":                ["crc", "colon cancer", "bowel cancer"],
    "non-hodgkin lymphoma":             ["nhl"],
    "chronic myeloid leukemia":         ["cml"],
    "acute myeloid leukemia":           ["aml"],
    "acute lymphoblastic leukemia":     ["all"],
    "chronic lymphocytic leukemia":     ["cll"],
    "attention deficit hyperactivity disorder": ["adhd"],
    "autism spectrum disorder":         ["asd", "autism"],
    "obsessive compulsive disorder":    ["ocd"],
    "post-traumatic stress disorder":   ["ptsd"],
    "polycystic kidney disease":        ["pkd"],
    "polycystic ovary syndrome":        ["pcos"],
    "inflammatory bowel disease":       ["ibd"],
    "irritable bowel syndrome":         ["ibs"],
    "gastroesophageal reflux disease":  ["gerd", "gord", "acid reflux"],
    "chronic obstructive pulmonary disease": ["copd"],
    "transient ischemic attack":        ["tia"],
    "atrial fibrillation":              ["afib", "af"],
    "coronary artery disease":          ["cad"],
    "pulmonary arterial hypertension":  ["pah"],
    "wilson disease":                   ["wilson's disease"],
    "gaucher disease":                  ["gaucher's disease"],
    "niemann-pick disease":             ["niemann pick disease"],
    "tay-sachs disease":                ["tay sachs disease", "tay sachs"],
    "neurofibromatosis type 1":         ["nf1"],
    "neurofibromatosis type 2":         ["nf2"],
    "von hippel-lindau disease":        ["vhl disease", "vhl syndrome"],
}

# Build reverse map: alias → canonical (all lower-case)
_ALIAS_MAP: dict[str, str] = {}
for _canonical, _aliases in _SYNONYMS.items():
    for _alias in _aliases:
        _ALIAS_MAP[_alias.lower().strip()] = _canonical

# Strip common question preambles before matching the disease name
_PREAMBLE_RE = re.compile(
    r"^(what is|tell me about|explain|describe|overview of|what are the|"
    r"symptoms of|causes of|genetics of|information on|what do you know about|"
    r"how does|what causes|diagnosis of|treatment for|what is the|"
    r"give me info on|can you explain|i want to know about|help me understand)\s+",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """
    Normalize a user query by expanding known disease abbreviations and
    correcting common misspellings.  The rest of the query text is preserved.
    """
    q = query.strip()
    # Strip question preambles so "what is cystic fibrosis" → "cystic fibrosis"
    q = _PREAMBLE_RE.sub("", q).strip().rstrip("?.!")
    q_lower = q.lower()

    # Exact full-query match (e.g. user typed just "cf")
    if q_lower in _ALIAS_MAP:
        return _ALIAS_MAP[q_lower]

    # Word / phrase boundary match within a longer query
    for alias, canonical in sorted(_ALIAS_MAP.items(), key=lambda x: -len(x[0])):
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, q_lower):
            q = re.sub(pattern, canonical, q, flags=re.IGNORECASE)
            q_lower = q.lower()  # keep in sync for subsequent matches

    # Remove consecutive duplicate words that expansion may introduce
    # e.g. "huntingtons disease" → "huntington disease disease" → "huntington disease"
    words = q.split()
    deduped = [words[0]] if words else []
    for w in words[1:]:
        if w.lower() != deduped[-1].lower():
            deduped.append(w)
    return " ".join(deduped)


# ── Fuzzy typo correction ─────────────────────────────────────────────────────

# Flat list of all canonical names for fuzzy matching
_ALL_CANONICAL = list(_SYNONYMS.keys())


def fuzzy_correct(query: str) -> tuple[str | None, float]:
    """
    Attempt to correct a misspelled disease name using fuzzy string matching.

    Returns (corrected_canonical, confidence) where confidence is 0.0–1.0,
    or (None, 0.0) if no close match is found.
    Only fires when the cleaned query does NOT already match any known alias.
    """
    q_clean = _PREAMBLE_RE.sub("", query.strip()).lower().rstrip("?.!")

    # Already an exact alias or canonical — no correction needed
    if q_clean in _ALIAS_MAP or q_clean in _SYNONYMS:
        return None, 0.0

    matches = difflib.get_close_matches(q_clean, _ALL_CANONICAL, n=1, cutoff=0.75)
    if not matches:
        return None, 0.0

    best = matches[0]
    score = difflib.SequenceMatcher(None, q_clean, best).ratio()
    # Require high confidence to avoid false positives like "banana" → "marfan"
    if score >= 0.78 and best != q_clean:
        return best, round(score, 3)
    return None, 0.0


# ── Fast keyword-based intent classifier ─────────────────────────────────────

_DISEASE_KEYWORDS = (
    "disease", "syndrome", "disorder", "cancer", "carcinoma", "tumour", "tumor",
    "leukemia", "leukaemia", "lymphoma", "fibrosis", "dystrophy", "deficiency",
    "anemia", "anaemia", "sclerosis", "atrophy", "palsy", "myopathy",
    "neuropathy", "encephalopathy", "epilepsy", "diabetes", "arthritis",
    "hypertension", "asthma", "obesity", "infection", "fever", "influenza",
    "malaria", "tuberculosis", "hiv", "aids", "hepatitis", "psoriasis",
    "eczema", "dermatitis", "osteoporosis", "osteoarthritis",
    "what is", "tell me about", "explain", "overview of", "symptoms of",
    "causes of", "genetics of", "pathophysiology of", "mechanism of",
    "how does", "clinical features", "clinical presentation",
)

_GENE_KEYWORDS = (
    "gene function", "gene expression", "gene mutation", "gene variant",
    "what does", "what is the role of", "encodes", "protein product",
    "knockout", "promoter", "transcription factor", "signaling pathway",
)

_DRUG_KEYWORDS = (
    "drug", "medication", "medicine", "therapy", "therapeutic", "treatment",
    "inhibitor", "agonist", "antagonist", "blocker", "monoclonal antibody",
    "clinical trial", "dose", "dosing", "pharmacology", "pharmacokinetics",
    "adverse effect", "side effect", "contraindication",
)

_LIT_KEYWORDS = (
    "paper", "study", "publication", "article", "research", "review",
    "literature", "published", "journal", "authors",
)

# Uppercase gene symbol (2–8 uppercase letters/digits, optionally with digits)
_GENE_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
# rsID
_RSID_RE = re.compile(r"^rs\d+$", re.IGNORECASE)


def fast_classify(query: str) -> str | None:
    """
    Classify a biomedical query without calling the LLM.

    Returns one of:  disease | variant | gene | drug | general
    Returns None when the query is ambiguous and the LLM should decide.
    """
    q = query.strip()
    ql = q.lower()

    # ── rsID ─────────────────────────────────────────────────────────────────
    if _RSID_RE.match(q):
        return "variant"

    # ── explicit variant language ─────────────────────────────────────────────
    if "rs" in ql and re.search(r"\brs\d+\b", ql):
        return "variant"
    if any(kw in ql for kw in ("variant", "mutation", "snp", "polymorphism",
                                "pathogenic", "benign", "vus", "acmg")):
        # Could be gene OR disease context — let LLM decide unless combined
        # with a gene symbol pattern
        if _GENE_SYMBOL_RE.search(q) and "disease" not in ql and "syndrome" not in ql:
            return "variant"

    # ── gene-specific ─────────────────────────────────────────────────────────
    has_gene_symbol = bool(_GENE_SYMBOL_RE.search(q))
    if has_gene_symbol and any(kw in ql for kw in _GENE_KEYWORDS):
        return "gene"
    # Pure gene symbol queries: "BRCA1", "TP53 function", "what does CFTR do"
    if has_gene_symbol and re.match(r"^[A-Z][A-Z0-9]{1,7}(\s|$)", q):
        return "gene"

    # ── disease ───────────────────────────────────────────────────────────────
    if any(kw in ql for kw in _DISEASE_KEYWORDS):
        return "disease"

    # ── drug ─────────────────────────────────────────────────────────────────
    if any(kw in ql for kw in _DRUG_KEYWORDS):
        return "drug"

    # ── literature ────────────────────────────────────────────────────────────
    if any(kw in ql for kw in _LIT_KEYWORDS):
        return "general"

    # Ambiguous — let the LLM classifier decide
    return None
