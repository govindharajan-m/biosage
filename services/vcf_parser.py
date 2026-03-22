"""
VCF (Variant Call Format) parser — pure Python, no bioinformatics deps.

Parses VCF 4.x files, extracts variants, and produces summary statistics
for the Variant Risk Report.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VCFVariant:
    chrom: str
    pos: int
    rsid: Optional[str]
    ref: str
    alt: str
    qual: Optional[float]
    filter_val: str
    info: dict = field(default_factory=dict)
    genotype: Optional[str] = None

    # Populated after database lookup
    clinical_significance: Optional[str] = None
    gene: Optional[str] = None
    associated_disease: Optional[str] = None
    evidence_sources: List[str] = field(default_factory=list)

    @property
    def variant_type(self) -> str:
        if len(self.ref) == 1 and len(self.alt) == 1:
            return "SNV"
        if len(self.ref) > len(self.alt):
            return "Deletion"
        if len(self.ref) < len(self.alt):
            return "Insertion"
        return "MNV"

    @property
    def is_known(self) -> bool:
        return self.rsid is not None

    def to_dict(self) -> dict:
        return {
            "chrom":                self.chrom,
            "pos":                  self.pos,
            "rsid":                 self.rsid,
            "ref":                  self.ref,
            "alt":                  self.alt,
            "qual":                 self.qual,
            "filter":               self.filter_val,
            "variant_type":         self.variant_type,
            "genotype":             self.genotype,
            "clinical_significance": self.clinical_significance,
            "gene":                 self.gene,
            "associated_disease":   self.associated_disease,
            "evidence_sources":     self.evidence_sources,
            "info_summary": {
                k: str(v) for k, v in list(self.info.items())[:8]
            },
        }


# ─── Parser ────────────────────────────────────────────────────────────────────

def parse_vcf(content: str) -> List[VCFVariant]:
    """
    Parse VCF content string.  Handles VCF 4.0 – 4.3.
    Returns a list of VCFVariant objects.
    """
    variants: List[VCFVariant] = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        cols = line.split("\t")
        if len(cols) < 5:
            continue

        try:
            chrom   = re.sub(r"^[Cc][Hh][Rr]", "", cols[0])
            pos     = int(cols[1])
            id_col  = cols[2]
            ref     = cols[3].upper()
            # take first ALT allele only
            alt     = cols[4].split(",")[0].upper()

            rsid = id_col if re.match(r"^rs\d+$", id_col, re.I) else None

            qual = None
            if len(cols) > 5 and cols[5] not in (".", ""):
                try:
                    qual = float(cols[5])
                except ValueError:
                    pass

            filter_val = cols[6] if len(cols) > 6 else "."

            info: dict = {}
            if len(cols) > 7 and cols[7] not in (".", ""):
                for item in cols[7].split(";"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        info[k] = v
                    else:
                        info[item] = True

            genotype = None
            if len(cols) > 9:
                gt = cols[9].split(":")[0]
                genotype = gt

            variants.append(VCFVariant(
                chrom=chrom, pos=pos, rsid=rsid,
                ref=ref, alt=alt, qual=qual,
                filter_val=filter_val, info=info, genotype=genotype,
            ))

        except (ValueError, IndexError) as exc:
            logger.debug(f"Skipping malformed VCF line: {exc}")

    return variants


# ─── Summary ───────────────────────────────────────────────────────────────────

def build_risk_report(variants: List[VCFVariant]) -> dict:
    """
    Summarise a list of VCFVariants (optionally annotated from live databases)
    into a Variant Risk Report.
    """
    total   = len(variants)
    known   = [v for v in variants if v.is_known]
    unknown = [v for v in variants if not v.is_known]

    by_sig: dict[str, List[VCFVariant]] = {
        "Pathogenic":               [],
        "Likely Pathogenic":        [],
        "Variant of Uncertain Significance": [],
        "Likely Benign":            [],
        "Benign":                   [],
        "Unknown":                  [],
    }

    for v in variants:
        sig = v.clinical_significance or "Unknown"
        key = _normalise_sig(sig)
        by_sig.setdefault(key, []).append(v)

    high_risk = by_sig["Pathogenic"] + by_sig["Likely Pathogenic"]

    by_type: dict[str, int] = {}
    for v in variants:
        by_type[v.variant_type] = by_type.get(v.variant_type, 0) + 1

    return {
        "total_variants":           total,
        "known_variants":           len(known),
        "novel_variants":           len(unknown),
        "pathogenic_count":         len(by_sig["Pathogenic"]),
        "likely_pathogenic_count":  len(by_sig["Likely Pathogenic"]),
        "vus_count":                len(by_sig["Variant of Uncertain Significance"]),
        "likely_benign_count":      len(by_sig["Likely Benign"]),
        "benign_count":             len(by_sig["Benign"]),
        "unknown_count":            len(by_sig["Unknown"]),
        "high_risk_variants":       [v.to_dict() for v in high_risk[:20]],
        "all_variants":             [v.to_dict() for v in variants[:200]],
        "variant_type_breakdown":   by_type,
        "risk_level":               _overall_risk(len(by_sig["Pathogenic"]), len(by_sig["Likely Pathogenic"])),
    }


def _normalise_sig(sig: str) -> str:
    s = sig.lower()
    if "pathogenic" in s and "likely" in s:
        return "Likely Pathogenic"
    if "pathogenic" in s:
        return "Pathogenic"
    if "likely benign" in s:
        return "Likely Benign"
    if "benign" in s:
        return "Benign"
    if "uncertain" in s or "vus" in s:
        return "Variant of Uncertain Significance"
    return "Unknown"


def _overall_risk(pathogenic: int, likely_pathogenic: int) -> str:
    if pathogenic >= 1:
        return "HIGH"
    if likely_pathogenic >= 2:
        return "ELEVATED"
    if likely_pathogenic >= 1:
        return "MODERATE"
    return "LOW"
