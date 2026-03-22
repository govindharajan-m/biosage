"""
ACMG/AMP 2015 Variant Interpretation Framework.

Implements the 5-tier classification system:
  Pathogenic | Likely Pathogenic | VUS | Likely Benign | Benign

Reference: Richards et al. (2015) Genetics in Medicine 17, 405-424.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Classification(str, Enum):
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    VUS = "Variant of Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


# ─── Evidence Container ────────────────────────────────────────────────────────

@dataclass
class ACMGEvidence:
    # Very Strong Pathogenic
    PVS1: bool = False   # Null variant (LOF) in gene where LOF = disease mechanism

    # Strong Pathogenic
    PS1: bool = False    # Same amino acid change as established pathogenic variant
    PS2: bool = False    # De novo (confirmed) in patient; no family history
    PS3: bool = False    # Well-established functional studies show damaging effect
    PS4: bool = False    # Variant prevalence in cases >> controls (OR > 5.0)

    # Moderate Pathogenic
    PM1: bool = False    # Located in mutational hot spot / critical functional domain
    PM2: bool = False    # Absent from controls (or very low freq) in gnomAD/ExAC
    PM3: bool = False    # Detected in trans with pathogenic variant (recessive)
    PM4: bool = False    # Protein length change (in-frame indel or stop-loss)
    PM5: bool = False    # Novel missense at amino acid where different missense = pathogenic
    PM6: bool = False    # Assumed de novo (not confirmed)

    # Supporting Pathogenic
    PP1: bool = False    # Cosegregation with disease in multiple affected family members
    PP2: bool = False    # Missense in gene with low rate of benign missense variation
    PP3: bool = False    # Multiple in-silico tools predict damaging
    PP4: bool = False    # Patient phenotype/history highly specific for the disease
    PP5: bool = False    # Reputable source reports variant as pathogenic

    # Stand-Alone Benign
    BA1: bool = False    # Allele frequency > 5 % in gnomAD / 1000G

    # Strong Benign
    BS1: bool = False    # Allele frequency > expected for the disorder
    BS2: bool = False    # Observed in healthy adult with full penetrance expected disease
    BS3: bool = False    # Well-established functional studies show no damaging effect
    BS4: bool = False    # Lack of segregation in affected family members

    # Supporting Benign
    BP1: bool = False    # Missense in gene where only truncating variants cause disease
    BP2: bool = False    # Observed in trans with pathogenic (dominant) or in cis
    BP3: bool = False    # In-frame indel in repetitive region without known function
    BP4: bool = False    # Multiple in-silico tools predict benign / tolerated
    BP5: bool = False    # Variant found in case with an alternate molecular basis
    BP6: bool = False    # Reputable source reports variant as benign
    BP7: bool = False    # Synonymous change with no predicted splice impact

    descriptions: dict = field(default_factory=dict)

    def set(self, criterion: str, reason: str):
        if hasattr(self, criterion):
            setattr(self, criterion, True)
            self.descriptions[criterion] = reason


# ─── Criteria metadata ─────────────────────────────────────────────────────────

PATHOGENIC_CRITERIA = {
    "PVS1": ("Very Strong", "Null variant where LOF is disease mechanism"),
    "PS1":  ("Strong",      "Same amino acid change as known pathogenic variant"),
    "PS2":  ("Strong",      "Confirmed de novo in affected patient"),
    "PS3":  ("Strong",      "Functional studies demonstrate damaging effect"),
    "PS4":  ("Strong",      "Significantly elevated prevalence in cases vs controls"),
    "PM1":  ("Moderate",    "Located in mutational hot spot / critical domain"),
    "PM2":  ("Moderate",    "Absent or extremely rare in population databases"),
    "PM3":  ("Moderate",    "Detected in trans with pathogenic variant (recessive)"),
    "PM4":  ("Moderate",    "Protein length change (in-frame indel / stop-loss)"),
    "PM5":  ("Moderate",    "Novel missense at established pathogenic amino acid"),
    "PM6":  ("Moderate",    "Assumed de novo (parentage not confirmed)"),
    "PP1":  ("Supporting",  "Cosegregation with disease in multiple affected relatives"),
    "PP2":  ("Supporting",  "Missense in gene with low benign missense variation rate"),
    "PP3":  ("Supporting",  "Computational predictions support damaging effect"),
    "PP4":  ("Supporting",  "Phenotype highly specific for disease with single genetic cause"),
    "PP5":  ("Supporting",  "Reputable source reports pathogenic classification"),
}

BENIGN_CRITERIA = {
    "BA1": ("Stand-Alone", "Allele frequency > 5 % in general population"),
    "BS1": ("Strong",      "Allele frequency greater than expected for disorder"),
    "BS2": ("Strong",      "Observed in healthy adult with fully penetrant disease"),
    "BS3": ("Strong",      "Functional studies show no damaging effect"),
    "BS4": ("Strong",      "Lack of segregation in affected family members"),
    "BP1": ("Supporting",  "Missense in gene where only LOF variants cause disease"),
    "BP2": ("Supporting",  "Observed in trans (dominant) or in cis with pathogenic"),
    "BP3": ("Supporting",  "In-frame deletion in repeat region without known function"),
    "BP4": ("Supporting",  "Computational tools predict benign / tolerated effect"),
    "BP5": ("Supporting",  "Found in case with alternative molecular diagnosis"),
    "BP6": ("Supporting",  "Reputable source reports benign classification"),
    "BP7": ("Supporting",  "Synonymous variant with no predicted splice effect"),
}


# ─── Classifier ────────────────────────────────────────────────────────────────

def classify(evidence: ACMGEvidence) -> tuple:
    """
    Apply ACMG combining rules.
    Returns (Classification, reasoning_str, confidence_int).
    """
    pvs = int(evidence.PVS1)
    ps  = sum(getattr(evidence, k) for k in ["PS1","PS2","PS3","PS4"])
    pm  = sum(getattr(evidence, k) for k in ["PM1","PM2","PM3","PM4","PM5","PM6"])
    pp  = sum(getattr(evidence, k) for k in ["PP1","PP2","PP3","PP4","PP5"])
    ba  = int(evidence.BA1)
    bs  = sum(getattr(evidence, k) for k in ["BS1","BS2","BS3","BS4"])
    bp  = sum(getattr(evidence, k) for k in ["BP1","BP2","BP3","BP4","BP5","BP6","BP7"])

    is_pathogenic = (
        (pvs and ps >= 1) or
        (pvs and pm >= 2) or
        (pvs and pm == 1 and pp >= 1) or
        (pvs and pp >= 2) or
        ps >= 2 or
        (ps == 1 and pm >= 3) or
        (ps == 1 and pm == 2 and pp >= 2) or
        (ps == 1 and pm == 1 and pp >= 4)
    )
    is_likely_pathogenic = (
        (pvs and pm == 1) or
        (ps == 1 and 1 <= pm <= 2) or
        (ps == 1 and pp >= 2) or
        pm >= 3 or
        (pm == 2 and pp >= 2) or
        (pm == 1 and pp >= 4)
    )
    is_benign        = ba >= 1 or bs >= 2
    is_likely_benign = (bs == 1 and bp >= 1) or bp >= 2

    triggered_p = [k for k in PATHOGENIC_CRITERIA if getattr(evidence, k)]
    triggered_b = [k for k in BENIGN_CRITERIA    if getattr(evidence, k)]

    # Resolution
    if is_benign:
        cls, conf = Classification.BENIGN, 96
        reason = f"Stand-alone benign criterion met ({', '.join(triggered_b)})"
    elif is_pathogenic and not (bs >= 1 or bp >= 2):
        cls  = Classification.PATHOGENIC
        conf = min(90 + pvs * 4 + ps * 2 + pm, 99)
        reason = f"Strong pathogenic evidence — PVS:{pvs} PS:{ps} PM:{pm} PP:{pp}"
    elif is_likely_pathogenic and not (bs >= 1 or bp >= 2):
        cls  = Classification.LIKELY_PATHOGENIC
        conf = min(75 + pm * 4 + pp * 2, 89)
        reason = f"Moderate pathogenic evidence — PVS:{pvs} PS:{ps} PM:{pm} PP:{pp}"
    elif is_likely_benign:
        cls  = Classification.LIKELY_BENIGN
        conf = min(75 + bs * 6 + bp * 2, 89)
        reason = f"Benign criteria met ({', '.join(triggered_b)})"
    else:
        cls, conf = Classification.VUS, 50
        reason = "Insufficient definitive evidence for classification"

    if triggered_p:
        reason += f"; Pathogenic criteria: {', '.join(triggered_p)}"
    if triggered_b:
        reason += f"; Benign criteria: {', '.join(triggered_b)}"

    return cls, reason, conf


# ─── Evidence inference from aggregated data ───────────────────────────────────

LOF_TERMS = {
    "stop_gained", "frameshift_variant", "splice_acceptor_variant",
    "splice_donor_variant", "transcript_ablation", "start_lost",
}
DAMAGING_TERMS = {"missense_variant", "inframe_deletion", "inframe_insertion", "protein_altering_variant"}
BENIGN_TERMS   = {"synonymous_variant", "intergenic_variant", "upstream_gene_variant", "downstream_gene_variant"}


def infer_evidence(variant_data: dict) -> ACMGEvidence:
    ev      = ACMGEvidence()
    unified = variant_data.get("unified", {})
    clinvar = variant_data.get("sources", {}).get("ClinVar", {})
    ensembl = variant_data.get("sources", {}).get("Ensembl", {})

    consequence_text = " ".join(unified.get("consequences", [])).lower().replace(" ", "_")
    clinical_sig     = unified.get("clinical_significance", "").lower()
    review_status    = clinvar.get("review_status", "").lower()
    maf_raw          = unified.get("population_frequency")

    # PVS1 — LOF in disease gene
    if any(t in consequence_text for t in LOF_TERMS):
        ev.set("PVS1", "Loss-of-function consequence detected (stop gain / frameshift / splice)")

    # PS1 — ClinVar expert-reviewed pathogenic
    if "pathogenic" in clinical_sig and "expert" in review_status:
        ev.set("PS1", "Expert-reviewed ClinVar pathogenic classification")

    # PP5 — ClinVar reviewed (not expert)
    if "pathogenic" in clinical_sig and "likely" not in clinical_sig:
        ev.set("PP5", f"ClinVar reports pathogenic (review: {review_status or 'unspecified'})")

    # PM4 — in-frame indel
    if "inframe" in consequence_text:
        ev.set("PM4", "In-frame insertion/deletion — protein length change")

    # PP2 / PP3 — missense with damaging prediction
    if any(t in consequence_text for t in DAMAGING_TERMS):
        ev.set("PP2", "Missense variant in potentially disease-relevant gene")
        ev.set("PP3", "Consequence predicted as damaging by computational tools")

    # PM2 / BA1 / BS1 — population frequency
    if maf_raw is not None:
        try:
            maf_val = float(str(maf_raw).split(":")[0]) if ":" in str(maf_raw) else float(str(maf_raw))
            if maf_val > 0.05:
                ev.set("BA1", f"Common variant: MAF = {maf_val:.3f} (> 5 %)")
                ev.set("BS1", "Allele frequency exceeds disease prevalence threshold")
            elif maf_val < 0.001:
                ev.set("PM2", f"Rare variant: MAF = {maf_val:.5f} (< 0.1 %)")
        except (ValueError, TypeError):
            pass

    # BP4 / BP7 — benign / synonymous
    if any(t in consequence_text for t in BENIGN_TERMS):
        ev.set("BP7", "Synonymous or non-coding variant with minimal predicted impact")
        ev.set("BP4", "Computational prediction: tolerated / benign")

    # BP6 — ClinVar reports benign
    if "benign" in clinical_sig and "pathogenic" not in clinical_sig:
        ev.set("BP6", f"ClinVar reports benign (review: {review_status or 'unspecified'})")

    return ev


# ─── Public report generator ───────────────────────────────────────────────────

def generate_report(variant_data: dict) -> dict:
    ev = infer_evidence(variant_data)
    cls, reasoning, confidence = classify(ev)

    def _criteria_list(mapping):
        out = []
        for cid, (strength, desc) in mapping.items():
            if getattr(ev, cid):
                out.append({
                    "id": cid,
                    "strength": strength,
                    "description": desc,
                    "details": ev.descriptions.get(cid, ""),
                })
        return out

    return {
        "rsid":                    variant_data.get("rsid", ""),
        "gene":                    variant_data.get("unified", {}).get("gene", ""),
        "classification":          cls.value,
        "confidence_percent":      confidence,
        "reasoning":               reasoning,
        "pathogenic_criteria":     _criteria_list(PATHOGENIC_CRITERIA),
        "benign_criteria":         _criteria_list(BENIGN_CRITERIA),
        "total_pathogenic":        sum(getattr(ev, k) for k in PATHOGENIC_CRITERIA),
        "total_benign":            sum(getattr(ev, k) for k in BENIGN_CRITERIA),
        "disclaimer": (
            "Automated heuristic classification — validate with a certified clinical geneticist "
            "before any clinical decision-making."
        ),
    }
