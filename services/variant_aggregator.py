"""
VariantAggregator — queries ClinVar, Ensembl, dbSNP, and UniProt
for a given rsID and returns a unified, conflict-annotated result.

aggregate()       — sequential (used by legacy endpoints)
async_aggregate() — parallel asyncio gather (3-4× faster, used by streaming)
"""

import asyncio
import logging
import time
from typing import Optional

import requests

from config import NCBI_API_KEY, NCBI_EMAIL, NCBI_BASE, REQUEST_TIMEOUT, OMIM_API_KEY

logger = logging.getLogger(__name__)

_HEADERS = {"Accept": "application/json", "User-Agent": "BioSage/1.0"}
if NCBI_EMAIL:
    _HEADERS["email"] = NCBI_EMAIL


def _get(url: str, params: dict = None, timeout: int = REQUEST_TIMEOUT) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"GET {url} failed: {e}")
        return None


class VariantAggregator:
    # ──────────────────────────────────────────────────────────────────────
    # ClinVar
    # ──────────────────────────────────────────────────────────────────────
    def query_clinvar(self, rsid: str) -> Optional[dict]:
        params = {
            "db": "clinvar",
            "term": f"{rsid}[rs]",
            "retmax": 5,
            "retmode": "json",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        search = _get(f"{NCBI_BASE}/esearch.fcgi", params)
        if not search:
            return None
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None

        fetch_params = {"db": "clinvar", "id": ",".join(ids[:3]), "retmode": "json"}
        if NCBI_API_KEY:
            fetch_params["api_key"] = NCBI_API_KEY
        data = _get(f"{NCBI_BASE}/esummary.fcgi", fetch_params)
        if not data:
            return None

        result_set = data.get("result", {})
        uids = result_set.get("uids", [])
        records = []
        for uid in uids:
            rec = result_set.get(uid, {})
            if not rec:
                continue
            germline = rec.get("germline_classification", {})
            genes = rec.get("genes", [])
            conditions = rec.get("trait_set", [])
            records.append({
                "clinvar_id": uid,
                "variant_name": rec.get("title", ""),
                "gene": genes[0].get("symbol", "") if genes else "",
                "clinical_significance": germline.get("description", "Unknown"),
                "review_status": germline.get("review_status", ""),
                "associated_diseases": [
                    c.get("trait_name", "") for c in conditions if c.get("trait_name")
                ][:4],
            })

        if not records:
            return None

        primary = records[0]
        all_sigs = list(dict.fromkeys(r["clinical_significance"] for r in records))
        all_diseases = list(dict.fromkeys(
            d for r in records for d in r["associated_diseases"]
        ))[:5]

        return {
            "source": "ClinVar",
            "rsid": rsid,
            "gene": primary["gene"],
            "variant_name": primary["variant_name"],
            "clinical_significance": primary["clinical_significance"],
            "all_significances": all_sigs,
            "review_status": primary["review_status"],
            "associated_diseases": all_diseases,
            "clinvar_ids": [r["clinvar_id"] for r in records],
            "url": f"https://www.ncbi.nlm.nih.gov/clinvar/?term={rsid}[rs]",
        }

    # ──────────────────────────────────────────────────────────────────────
    # Ensembl
    # ──────────────────────────────────────────────────────────────────────
    def query_ensembl(self, rsid: str) -> Optional[dict]:
        data = _get(
            f"https://rest.ensembl.org/variation/human/{rsid}",
            {"content-type": "application/json"},
        )
        if not data:
            return None

        mappings = data.get("mappings", [])
        location = ""
        chromosome = ""
        if mappings:
            m = mappings[0]
            chromosome = str(m.get("seq_region_name", ""))
            start = m.get("start", "")
            location = f"Chr{chromosome}:{start}"

        maf = data.get("MAF")
        clinical_sig = data.get("clinical_significance", [])
        if isinstance(clinical_sig, list):
            clinical_sig = ", ".join(clinical_sig) if clinical_sig else "Unknown"

        # Build consequence list
        consequences = []
        for m in mappings:
            for tc in m.get("transcript_consequences", [])[:3]:
                gene = tc.get("gene_symbol", "")
                terms = tc.get("consequence_terms", [])
                if gene and terms:
                    consequences.append(f"{gene}: {', '.join(terms)}")

        return {
            "source": "Ensembl",
            "rsid": rsid,
            "clinical_significance": clinical_sig,
            "location": location,
            "chromosome": chromosome,
            "maf": maf,
            "var_class": data.get("var_class", ""),
            "ancestral_allele": data.get("ancestral_allele", ""),
            "consequences": consequences[:5],
            "synonyms": data.get("synonyms", [])[:5],
            "url": f"https://www.ensembl.org/Homo_sapiens/Variation/Explore?v={rsid}",
        }

    # ──────────────────────────────────────────────────────────────────────
    # dbSNP
    # ──────────────────────────────────────────────────────────────────────
    def query_dbsnp(self, rsid: str) -> Optional[dict]:
        rsnum = rsid.lower().replace("rs", "")
        params = {"db": "snp", "id": rsnum, "retmode": "json"}
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        data = _get(f"{NCBI_BASE}/esummary.fcgi", params)
        if not data:
            return None

        result = data.get("result", {})
        uids = result.get("uids", [])
        if not uids:
            return None
        snp = result.get(uids[0], {})
        if not snp:
            return None

        fxn_set = snp.get("fxn_set", [])
        consequences = list(dict.fromkeys(
            f.get("fxn_class", "").replace("_", " ")
            for f in fxn_set[:6]
            if f.get("fxn_class")
        ))

        return {
            "source": "dbSNP",
            "rsid": rsid,
            "allele_origin": snp.get("allele_origin", ""),
            "global_maf": snp.get("global_maf", ""),
            "snp_class": snp.get("snp_class", ""),
            "chromosomal_position": snp.get("chrpos", ""),
            "consequences": consequences[:5],
            "url": f"https://www.ncbi.nlm.nih.gov/snp/{rsid}",
        }

    # ──────────────────────────────────────────────────────────────────────
    # UniProt
    # ──────────────────────────────────────────────────────────────────────
    def query_uniprot(self, gene: str) -> Optional[dict]:
        if not gene:
            return None
        data = _get(
            "https://rest.uniprot.org/uniprotkb/search",
            {
                "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "fields": "accession,protein_name,gene_names,cc_function,sequence",
                "size": 1,
            },
        )
        if not data or not data.get("results"):
            return None

        entry = data["results"][0]
        accession = entry.get("primaryAccession", "")
        pn = entry.get("proteinDescription", {})
        rec_name = pn.get("recommendedName", {})
        protein_name = rec_name.get("fullName", {}).get("value", "") if rec_name else ""

        function_text = ""
        for c in entry.get("comments", []):
            if c.get("commentType") == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")[:600]
                    break

        seq = entry.get("sequence", {})
        return {
            "source": "UniProt",
            "gene": gene,
            "accession": accession,
            "protein_name": protein_name,
            "function": function_text,
            "sequence_length": seq.get("length", ""),
            "mass": seq.get("molWeight", ""),
            "url": f"https://www.uniprot.org/uniprotkb/{accession}",
        }

    # ──────────────────────────────────────────────────────────────────────
    # Parallel async aggregation (primary path for streaming endpoint)
    # ──────────────────────────────────────────────────────────────────────
    async def async_aggregate(self, rsid: str) -> dict:
        """
        Run ClinVar, Ensembl, and dbSNP queries concurrently, then resolve
        the gene symbol and fetch UniProt in a second async step.

        Typical latency improvement: 3-4× over the sequential aggregate().
        """
        loop = asyncio.get_running_loop()

        async def run(fn, *args):
            try:
                return await loop.run_in_executor(None, fn, *args)
            except Exception as exc:
                logger.warning("async_aggregate %s(%s): %s", fn.__name__, args, exc)
                return None

        # Phase 1: three independent DB lookups in parallel
        clinvar, ensembl, dbsnp = await asyncio.gather(
            run(self.query_clinvar, rsid),
            run(self.query_ensembl, rsid),
            run(self.query_dbsnp, rsid),
        )

        sources: dict = {}
        if clinvar:
            sources["ClinVar"] = clinvar
        if ensembl:
            sources["Ensembl"] = ensembl
        if dbsnp:
            sources["dbSNP"] = dbsnp

        # Phase 2: UniProt (needs gene symbol from phase 1)
        gene = (
            (clinvar or {}).get("gene")
            or _gene_from_consequences((ensembl or {}).get("consequences", []))
            or ""
        )
        uniprot = await run(self.query_uniprot, gene) if gene else None
        if uniprot:
            sources["UniProt"] = uniprot

        unified   = self._unify(sources, rsid, gene)
        conflicts = self._conflicts(sources)

        return {
            "rsid": rsid,
            "sources": sources,
            "unified": unified,
            "conflicts": conflicts,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Sequential aggregation entry point (legacy / non-streaming callers)
    # ──────────────────────────────────────────────────────────────────────
    def aggregate(self, rsid: str) -> dict:
        sources = {}

        clinvar = self.query_clinvar(rsid)
        if clinvar:
            sources["ClinVar"] = clinvar
        time.sleep(0.15)

        ensembl = self.query_ensembl(rsid)
        if ensembl:
            sources["Ensembl"] = ensembl

        dbsnp = self.query_dbsnp(rsid)
        if dbsnp:
            sources["dbSNP"] = dbsnp
        time.sleep(0.15)

        # Resolve gene symbol across sources
        gene = (
            (clinvar or {}).get("gene")
            or _gene_from_consequences((ensembl or {}).get("consequences", []))
            or ""
        )

        uniprot = self.query_uniprot(gene) if gene else None
        if uniprot:
            sources["UniProt"] = uniprot

        unified = self._unify(sources, rsid, gene)
        conflicts = self._conflicts(sources)

        return {
            "rsid": rsid,
            "sources": sources,
            "unified": unified,
            "conflicts": conflicts,
        }

    def _unify(self, sources: dict, rsid: str, gene: str) -> dict:
        clinvar = sources.get("ClinVar", {})
        ensembl = sources.get("Ensembl", {})
        dbsnp = sources.get("dbSNP", {})
        uniprot = sources.get("UniProt", {})

        clinical_sig = (
            clinvar.get("clinical_significance")
            or ensembl.get("clinical_significance")
            or "Unknown"
        )
        maf_raw = ensembl.get("maf") or dbsnp.get("global_maf") or None
        maf_str = str(maf_raw) if maf_raw is not None else None
        location = ensembl.get("location") or dbsnp.get("chromosomal_position") or ""
        chromosome = ensembl.get("chromosome") or ""

        return {
            "rsid": rsid,
            "gene": gene,
            "clinical_significance": clinical_sig,
            "location": location,
            "chromosome": chromosome,
            "population_frequency": maf_str,
            "associated_diseases": clinvar.get("associated_diseases", []),
            "var_class": ensembl.get("var_class") or dbsnp.get("snp_class") or "",
            "consequences": (ensembl.get("consequences") or dbsnp.get("consequences") or []),
            "protein_name": (uniprot or {}).get("protein_name", ""),
            "protein_function": (uniprot or {}).get("function", ""),
            "sources_available": list(sources.keys()),
        }

    def _conflicts(self, sources: dict) -> list:
        sigs = {
            name: data["clinical_significance"]
            for name, data in sources.items()
            if data.get("clinical_significance") not in (None, "", "Unknown")
        }
        if len(set(sigs.values())) > 1:
            return [{"field": "clinical_significance", "values": sigs}]
        return []


def _gene_from_consequences(consequences: list) -> str:
    for c in consequences:
        if ":" in c:
            return c.split(":")[0].strip()
    return ""
