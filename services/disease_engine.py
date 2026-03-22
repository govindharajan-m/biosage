"""
DiseaseEngine — multi-database disease information aggregation.

Queries NCBI MedGen, OMIM, Open Targets, DisGeNET, ClinVar,
ChEMBL, Reactome, and HPO. Returns a structured evidence bundle
where every data point carries citation metadata.
"""

import asyncio
import logging
from typing import Optional

import requests

from config import NCBI_API_KEY, NCBI_BASE, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "BioSage/3.0 (biomedical-research-platform)",
}


def _get(url: str, params: dict = None, timeout: int = REQUEST_TIMEOUT) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"GET {url} failed: {e}")
        return None


def _post(url: str, json_data: dict, timeout: int = REQUEST_TIMEOUT) -> Optional[dict]:
    try:
        r = requests.post(url, json=json_data, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"POST {url} failed: {e}")
        return None


def _ncbi_params(extra: dict = None) -> dict:
    p = {"retmode": "json"}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if extra:
        p.update(extra)
    return p


class DiseaseEngine:
    """
    Orchestrates multi-database disease information retrieval.
    All methods return structured dicts with citation metadata attached.
    The aggregate() method is the primary entry point.
    """

    # ── NCBI MedGen ────────────────────────────────────────────────────────

    def search_medgen(self, disease_name: str) -> Optional[dict]:
        """Fetch disease concept and definition from NCBI MedGen."""
        params = _ncbi_params({
            "db": "medgen",
            "term": f'"{disease_name}"[TITL]',
            "retmax": 3,
        })
        search = _get(f"{NCBI_BASE}/esearch.fcgi", params)
        if not search:
            return None
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            # Broader fallback
            params["term"] = disease_name
            search = _get(f"{NCBI_BASE}/esearch.fcgi", params)
            if not search:
                return None
            ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None

        fetch = _get(f"{NCBI_BASE}/esummary.fcgi", _ncbi_params({"db": "medgen", "id": ids[0]}))
        if not fetch:
            return None
        result = fetch.get("result", {})
        uids = result.get("uids", [])
        if not uids:
            return None
        rec = result.get(uids[0], {})

        defn = rec.get("definition", "") or ""
        if isinstance(defn, dict):
            defn = defn.get("value") or defn.get("text") or str(defn)
        return {
            "source": "NCBI MedGen",
            "source_url": f"https://www.ncbi.nlm.nih.gov/medgen/{uids[0]}",
            "concept_id": str(rec.get("conceptid", "") or ""),
            "name": str(rec.get("title", disease_name) or disease_name),
            "definition": str(defn)[:600],
            "synonyms": [str(s.get("name", "")) for s in rec.get("synonyms", []) if s.get("name")][:6],
            "semantic_type": str(rec.get("semantictype", "") or ""),
        }

    # ── OMIM ───────────────────────────────────────────────────────────────

    def search_omim(self, disease_name: str) -> list:
        """Fetch OMIM entries for the disease via NCBI eutils."""
        params = _ncbi_params({
            "db": "omim",
            "term": f'"{disease_name}"[TITL]',
            "retmax": 5,
        })
        search = _get(f"{NCBI_BASE}/esearch.fcgi", params)
        if not search:
            return []
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch = _get(f"{NCBI_BASE}/esummary.fcgi", _ncbi_params({"db": "omim", "id": ",".join(ids[:4])}))
        if not fetch:
            return []

        results = []
        result = fetch.get("result", {})
        for uid in result.get("uids", []):
            rec = result.get(uid, {})
            if not rec:
                continue
            results.append({
                "source": "OMIM",
                "source_url": f"https://www.omim.org/entry/{uid}",
                "mim_number": uid,
                "title": rec.get("title", ""),
                "genes": [g.get("genesymbol", "") for g in rec.get("geneMap", []) if g.get("genesymbol")][:5],
            })
        return results

    # ── Open Targets ────────────────────────────────────────────────────────

    def get_associations_opentargets(self, disease_name: str) -> list:
        """Query Open Targets Platform for top gene-disease associations."""
        search_q = """
        query($q: String!) {
          search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits { id name }
          }
        }"""
        res = _post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            {"query": search_q, "variables": {"q": disease_name}},
        )
        if not res:
            return []
        hits = res.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return []
        disease_id = hits[0]["id"]
        disease_label = hits[0]["name"]

        assoc_q = """
        query($id: String!) {
          disease(efoId: $id) {
            associatedTargets(page: {index: 0, size: 20}) {
              rows {
                score
                target { id approvedSymbol approvedName biotype }
              }
            }
          }
        }"""
        assoc = _post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            {"query": assoc_q, "variables": {"id": disease_id}},
        )
        if not assoc:
            return []
        rows = assoc.get("data", {}).get("disease", {}).get("associatedTargets", {}).get("rows", [])
        results = []
        for row in rows:
            t = row.get("target", {})
            gene_id = t.get("id", "")
            results.append({
                "source": "Open Targets",
                "source_url": f"https://platform.opentargets.org/target/{gene_id}/associations",
                "disease_name": disease_label,
                "disease_efo_id": disease_id,
                "gene": t.get("approvedSymbol", ""),
                "gene_name": t.get("approvedName", ""),
                "gene_type": t.get("biotype", ""),
                "ensembl_id": gene_id,
                "association_score": round(float(row.get("score", 0)), 3),
            })
        return results

    # ── DisGeNET ────────────────────────────────────────────────────────────

    def get_associations_disgenet(self, disease_name: str) -> list:
        """Query DisGeNET for gene-disease associations (public API)."""
        data = _get(
            "https://www.disgenet.org/api/gda/disease/search",
            {"disease_name": disease_name, "source": "ALL", "format": "json", "limit": 20},
        )
        if not data or not isinstance(data, list):
            return []
        results = []
        for rec in data[:15]:
            gene = rec.get("geneName") or rec.get("gene_symbol", "")
            if not gene:
                continue
            score = rec.get("score")
            results.append({
                "source": "DisGeNET",
                "source_url": f"https://www.disgenet.org/gene/{rec.get('geneNcbiID', '')}",
                "gene": gene,
                "gene_ncbi_id": str(rec.get("geneNcbiID", "")),
                "association_score": round(float(score), 3) if score else None,
                "n_pmids": rec.get("pmidCount", 0),
                "n_snps": rec.get("snpCount", 0),
            })
        results.sort(key=lambda x: x.get("association_score") or 0, reverse=True)
        return results

    # ── ClinVar variants ────────────────────────────────────────────────────

    def get_clinvar_variants(self, disease_name: str) -> list:
        """Fetch pathogenic/likely-pathogenic variants from ClinVar for a disease."""
        params = _ncbi_params({
            "db": "clinvar",
            "term": (
                f'"{disease_name}"[Disease/Phenotype] AND '
                '(pathogenic[Clinical_significance] OR "likely pathogenic"[Clinical_significance])'
            ),
            "retmax": 20,
            "sort": "relevance",
        })
        search = _get(f"{NCBI_BASE}/esearch.fcgi", params)
        if not search:
            return []
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch = _get(f"{NCBI_BASE}/esummary.fcgi", _ncbi_params({"db": "clinvar", "id": ",".join(ids[:12])}))
        if not fetch:
            return []

        results = []
        rs = fetch.get("result", {})
        for uid in rs.get("uids", []):
            rec = rs.get(uid, {})
            if not rec:
                continue
            genes = rec.get("genes", [])
            germline = rec.get("germline_classification", {})
            results.append({
                "source": "ClinVar",
                "source_url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/",
                "clinvar_id": uid,
                "variant_name": rec.get("title", ""),
                "gene": genes[0].get("symbol", "") if genes else "",
                "clinical_significance": germline.get("description", ""),
                "review_status": germline.get("review_status", ""),
            })
        return results

    # ── ChEMBL drugs ────────────────────────────────────────────────────────

    def get_drugs(self, disease_name: str) -> list:
        """Query ChEMBL drug indications for the disease."""
        data = _get(
            "https://www.ebi.ac.uk/chembl/api/data/drug_indication",
            {"efo_term__icontains": disease_name, "format": "json", "limit": 20},
        )
        if not data:
            return []
        results = []
        seen = set()
        for rec in data.get("drug_indications", []):
            cid = rec.get("molecule_chembl_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            results.append({
                "source": "ChEMBL",
                "source_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/",
                "drug_chembl_id": cid,
                "drug_name": rec.get("molecule_name") or cid,
                "indication": rec.get("efo_term", ""),
                "max_phase": rec.get("max_phase_for_ind", 0),
            })
        results.sort(key=lambda x: x.get("max_phase") or 0, reverse=True)
        return results

    # ── Reactome pathways ───────────────────────────────────────────────────

    def get_pathways(self, gene_symbol: str) -> list:
        """Fetch Reactome pathways for a gene via the Reactome content service."""
        # Primary: search by gene symbol
        search = _get(
            "https://reactome.org/ContentService/search/query",
            params={"query": gene_symbol, "species": "Homo sapiens", "types": "Pathway", "cluster": "true"},
        )
        if search and search.get("results"):
            entries = (search["results"] or [{}])[0].get("entries", [])
            data = entries[:12]
        else:
            data = []
        if not data:
            # Fallback: gene-to-pathway mapping via Ensembl ID lookup
            lookup = _get(
                f"https://reactome.org/ContentService/data/mapping/UniProt/{gene_symbol}/pathways",
                params={"species": "9606"},
            )
            data = lookup if lookup and isinstance(lookup, list) else []

        results = []
        for p in data[:12]:
            stid = p.get("stId") or p.get("dbId", "")
            name = p.get("name") or p.get("displayName", "")
            if not name:
                continue
            results.append({
                "source": "Reactome",
                "source_url": f"https://reactome.org/PathwayBrowser/#/{stid}" if stid else "https://reactome.org",
                "pathway_id": str(stid),
                "pathway_name": name,
                "gene": gene_symbol,
            })
        return results

    # ── HPO phenotypes ──────────────────────────────────────────────────────

    def get_phenotypes(self, disease_name: str) -> list:
        """
        Retrieve HPO phenotype terms via EBI OLS4 (Ontology Lookup Service).
        Falls back to Open Targets phenotypes if OLS4 returns nothing.
        """
        # EBI OLS4 — searches Human Phenotype Ontology terms matching the disease
        data = _get(
            "https://www.ebi.ac.uk/ols4/api/search",
            params={
                "q": disease_name,
                "ontology": "hp",
                "rows": 20,
                "fieldList": "id,label,description,obo_id",
            },
        )
        results = []
        if data:
            docs = data.get("response", {}).get("docs", [])
            for doc in docs[:15]:
                label = doc.get("label", "")
                hpo_id = doc.get("obo_id") or doc.get("id", "")
                if not label:
                    continue
                # Skip terms that are the disease itself — we want phenotype terms
                if label.lower() == disease_name.lower():
                    continue
                results.append({
                    "source": "HPO",
                    "source_url": f"https://hpo.jax.org/app/browse/term/{hpo_id}",
                    "hpo_id": hpo_id,
                    "phenotype": label,
                    "frequency": "",
                })

        # Open Targets phenotype fallback if OLS4 gave nothing
        if not results:
            results = self._get_phenotypes_opentargets(disease_name)

        return results[:20]

    def _get_phenotypes_opentargets(self, disease_name: str) -> list:
        """Fetch HPO phenotypes from Open Targets using the correct GQL schema."""
        search_q = """
        query($q: String!) {
          search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits { id name }
          }
        }"""
        res = _post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            {"query": search_q, "variables": {"q": disease_name}},
        )
        if not res:
            return []
        hits = res.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return []
        disease_id = hits[0]["id"]

        # Open Targets exposes HPO phenotypes via the hpo field on disease
        pheno_q = """
        query($id: String!) {
          disease(efoId: $id) {
            phenotypes(page: {index: 0, size: 20}) {
              rows {
                phenotypeHPO { id name }
                evidence { diseaseFromSource }
              }
            }
          }
        }"""
        pheno = _post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            {"query": pheno_q, "variables": {"id": disease_id}},
        )
        if not pheno or pheno.get("errors"):
            return []
        rows = (
            pheno.get("data", {}).get("disease", {})
            .get("phenotypes", {}).get("rows", [])
        )
        results = []
        for row in rows[:20]:
            hpo = row.get("phenotypeHPO") or {}
            label = hpo.get("name", "")
            hpo_id = hpo.get("id", "")
            if not label:
                continue
            results.append({
                "source": "HPO",
                "source_url": f"https://hpo.jax.org/app/browse/term/{hpo_id}",
                "hpo_id": hpo_id,
                "phenotype": label,
                "frequency": "",
            })
        return results

    # ── Progressive streaming aggregator ────────────────────────────────────
    # Yields partial evidence events as each DB call completes, then yields
    # the final complete evidence bundle.  Server uses this for streaming
    # endpoints so the LLM can start as soon as the first results arrive.

    async def stream_aggregate(self, disease_name: str):
        """
        Async generator.  Yields dicts:
          {"type": "partial", "key": str, "data": any}   — as each DB responds
          {"type": "complete", "evidence": dict}          — full bundle at end

        Adds a 6-second global timeout so slow DBs never block synthesis.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        async def _run(key: str, fn, *args):
            try:
                result = await loop.run_in_executor(None, fn, *args)
            except Exception as e:
                logger.warning("stream_aggregate %s error: %s", key, e)
                result = None
            await queue.put((key, result))

        tasks_def = [
            ("medgen",      self.search_medgen,                  (disease_name,)),
            ("omim",        self.search_omim,                    (disease_name,)),
            ("opentargets", self.get_associations_opentargets,   (disease_name,)),
            ("disgenet",    self.get_associations_disgenet,      (disease_name,)),
            ("clinvar",     self.get_clinvar_variants,           (disease_name,)),
            ("drugs",       self.get_drugs,                      (disease_name,)),
            ("phenotypes",  self.get_phenotypes,                 (disease_name,)),
        ]

        tasks = [asyncio.create_task(_run(k, fn, *args)) for k, fn, args in tasks_def]
        n_tasks = len(tasks)
        received: dict = {}

        try:
            deadline = loop.time() + 6.0  # 6-second total budget
            for _ in range(n_tasks):
                remaining = max(0.1, deadline - loop.time())
                try:
                    key, data = await asyncio.wait_for(queue.get(), timeout=remaining)
                    received[key] = data
                    yield {"type": "partial", "key": key, "data": data}
                except asyncio.TimeoutError:
                    logger.warning("stream_aggregate: timeout hit after %d/%d sources", len(received), n_tasks)
                    break
        finally:
            for t in tasks:
                t.cancel()

        # Phase 2: pathways (needs top gene)
        ot_genes  = received.get("opentargets") or []
        dgn_genes = received.get("disgenet") or []
        genes     = ot_genes if ot_genes else dgn_genes
        top_gene  = genes[0].get("gene", "") if genes else ""
        pathways: list = []
        if top_gene:
            try:
                pathways = await asyncio.wait_for(
                    loop.run_in_executor(None, self.get_pathways, top_gene),
                    timeout=4.0,
                )
            except asyncio.TimeoutError:
                pass
            yield {"type": "partial", "key": "pathways", "data": pathways or []}

        # Assemble complete evidence bundle
        evidence = await self._build_evidence(disease_name, received, pathways or [])
        yield {"type": "complete", "evidence": evidence}

    async def _build_evidence(self, disease_name: str, received: dict, pathways: list) -> dict:
        """Build the standard evidence bundle from received partial results."""
        medgen       = received.get("medgen")
        omim_entries = received.get("omim") or []
        ot_genes     = received.get("opentargets") or []
        dgn_genes    = received.get("disgenet") or []
        variants     = received.get("clinvar") or []
        drugs        = received.get("drugs") or []
        phenotypes   = received.get("phenotypes") or []

        genes       = ot_genes if ot_genes else dgn_genes
        gene_source = "Open Targets" if ot_genes else ("DisGeNET" if dgn_genes else "")

        evidence: dict = {
            "query": disease_name, "sources_used": [],
            "overview": None, "omim_entries": [],
            "genes": [], "variants": [], "drugs": [],
            "pathways": [], "phenotypes": [], "all_citations": [],
        }
        n = [0]

        def cite(db, url, label):
            n[0] += 1
            evidence["all_citations"].append({"n": n[0], "db": db, "url": url, "label": label})
            return n[0]

        if medgen:
            medgen["cn"] = cite(medgen["source"], medgen["source_url"], medgen["name"])
            evidence["overview"] = medgen
            evidence["sources_used"].append("NCBI MedGen")

        for entry in omim_entries:
            entry["cn"] = cite(entry["source"], entry["source_url"], entry["title"])
            evidence["omim_entries"].append(entry)
        if omim_entries:
            evidence["sources_used"].append("OMIM")

        for g in genes:
            lbl = (f"{g.get('gene','')} — {g.get('gene_name','')}"
                   if gene_source == "Open Targets" else g.get("gene", ""))
            g["cn"] = cite(g["source"], g["source_url"], lbl)
        evidence["genes"] = genes
        if genes and gene_source:
            evidence["sources_used"].append(gene_source)

        for v in variants:
            v["cn"] = cite(v["source"], v["source_url"], v.get("variant_name") or v.get("clinvar_id", ""))
            evidence["variants"].append(v)
        if variants:
            evidence["sources_used"].append("ClinVar")

        for d in drugs:
            d["cn"] = cite(d["source"], d["source_url"], d["drug_name"])
            evidence["drugs"].append(d)
        if drugs:
            evidence["sources_used"].append("ChEMBL")

        for p in pathways:
            p["cn"] = cite(p["source"], p["source_url"], p["pathway_name"])
            evidence["pathways"].append(p)
        if pathways:
            evidence["sources_used"].append("Reactome")

        for ph in phenotypes:
            ph["cn"] = cite(ph["source"], ph["source_url"], ph["phenotype"])
            evidence["phenotypes"].append(ph)
        if phenotypes:
            evidence["sources_used"].append("HPO")

        seen: set = set()
        evidence["sources_used"] = [
            s for s in evidence["sources_used"] if not (s in seen or seen.add(s))
        ]
        return evidence

    # ── Parallel async aggregator ────────────────────────────────────────────

    async def async_aggregate(self, disease_name: str) -> dict:
        """
        Parallel version of aggregate() using asyncio.gather + thread pool.

        Phase 1: All independent sources run concurrently (MedGen, OMIM,
                 Open Targets, DisGeNET, ClinVar, ChEMBL, HPO).
        Phase 2: Pathways (needs top gene from phase 1).

        Typical speedup: 3-5x over sequential aggregate().
        """
        loop = asyncio.get_running_loop()

        async def run(fn, *args):
            try:
                return await loop.run_in_executor(None, fn, *args)
            except Exception as e:
                logger.warning("async_aggregate %s error: %s", fn.__name__, e)
                return None

        logger.info("DiseaseEngine.async_aggregate (parallel): '%s'", disease_name)

        # Phase 1: all independent fetches in parallel
        (medgen, omim_entries, ot_genes, dgn_genes, variants, drugs, phenotypes) = \
            await asyncio.gather(
                run(self.search_medgen, disease_name),
                run(self.search_omim, disease_name),
                run(self.get_associations_opentargets, disease_name),
                run(self.get_associations_disgenet, disease_name),
                run(self.get_clinvar_variants, disease_name),
                run(self.get_drugs, disease_name),
                run(self.get_phenotypes, disease_name),
            )

        omim_entries = omim_entries or []
        ot_genes     = ot_genes or []
        dgn_genes    = dgn_genes or []
        variants     = variants or []
        drugs        = drugs or []
        phenotypes   = phenotypes or []

        genes = ot_genes if ot_genes else dgn_genes
        gene_source = "Open Targets" if ot_genes else ("DisGeNET" if dgn_genes else "")
        top_gene = genes[0].get("gene", "") if genes else ""

        # Phase 2: pathways (depends on top gene)
        pathways: list = []
        if top_gene:
            pathways = await run(self.get_pathways, top_gene) or []

        # Assemble evidence bundle
        evidence: dict = {
            "query": disease_name,
            "sources_used": [],
            "overview": None,
            "omim_entries": [],
            "genes": [],
            "variants": [],
            "drugs": [],
            "pathways": [],
            "phenotypes": [],
            "all_citations": [],
        }
        n = [0]

        def cite(db: str, url: str, label: str) -> int:
            n[0] += 1
            evidence["all_citations"].append({"n": n[0], "db": db, "url": url, "label": label})
            return n[0]

        if medgen:
            medgen["cn"] = cite(medgen["source"], medgen["source_url"], medgen["name"])
            evidence["overview"] = medgen
            evidence["sources_used"].append("NCBI MedGen")

        for entry in omim_entries:
            entry["cn"] = cite(entry["source"], entry["source_url"], entry["title"])
            evidence["omim_entries"].append(entry)
        if omim_entries:
            evidence["sources_used"].append("OMIM")

        for g in genes:
            lbl = f"{g.get('gene','')} — {g.get('gene_name','')}" if gene_source == "Open Targets" else g.get("gene","")
            g["cn"] = cite(g["source"], g["source_url"], lbl)
        evidence["genes"] = genes
        if genes and gene_source:
            evidence["sources_used"].append(gene_source)

        for v in variants:
            v["cn"] = cite(v["source"], v["source_url"], v.get("variant_name") or v.get("clinvar_id",""))
            evidence["variants"].append(v)
        if variants:
            evidence["sources_used"].append("ClinVar")

        for d in drugs:
            d["cn"] = cite(d["source"], d["source_url"], d["drug_name"])
            evidence["drugs"].append(d)
        if drugs:
            evidence["sources_used"].append("ChEMBL")

        for p in pathways:
            p["cn"] = cite(p["source"], p["source_url"], p["pathway_name"])
            evidence["pathways"].append(p)
        if pathways:
            evidence["sources_used"].append("Reactome")

        for ph in phenotypes:
            ph["cn"] = cite(ph["source"], ph["source_url"], ph["phenotype"])
            evidence["phenotypes"].append(ph)
        if phenotypes:
            evidence["sources_used"].append("HPO")

        seen: set = set()
        evidence["sources_used"] = [
            s for s in evidence["sources_used"] if not (s in seen or seen.add(s))
        ]
        logger.info(
            "DiseaseEngine.async_aggregate: %d citations, %d sources",
            len(evidence["all_citations"]), len(evidence["sources_used"]),
        )
        return evidence

    # ── Master aggregator ───────────────────────────────────────────────────

    def aggregate(self, disease_name: str) -> dict:
        """
        Aggregate evidence about a disease from all integrated sources.

        Returns a structured dict ready for LLM synthesis:
        {
            "query": str,
            "sources_used": [str],
            "overview": dict | None,
            "omim_entries": [dict],
            "genes": [dict],
            "variants": [dict],
            "drugs": [dict],
            "pathways": [dict],
            "phenotypes": [dict],
            "all_citations": [{"n": int, "db": str, "url": str, "label": str}]
        }
        """
        logger.info(f"DiseaseEngine.aggregate: '{disease_name}'")

        evidence = {
            "query": disease_name,
            "sources_used": [],
            "overview": None,
            "omim_entries": [],
            "genes": [],
            "variants": [],
            "drugs": [],
            "pathways": [],
            "phenotypes": [],
            "all_citations": [],
        }

        n = [0]  # citation counter

        def cite(db: str, url: str, label: str) -> int:
            n[0] += 1
            evidence["all_citations"].append({"n": n[0], "db": db, "url": url, "label": label})
            return n[0]

        # 1. MedGen overview
        medgen = self.search_medgen(disease_name)
        if medgen:
            medgen["cn"] = cite(medgen["source"], medgen["source_url"], medgen["name"])
            evidence["overview"] = medgen
            evidence["sources_used"].append("NCBI MedGen")

        # 2. OMIM
        for entry in self.search_omim(disease_name):
            entry["cn"] = cite(entry["source"], entry["source_url"], entry["title"])
            evidence["omim_entries"].append(entry)
        if evidence["omim_entries"]:
            evidence["sources_used"].append("OMIM")

        # 3. Gene associations — Open Targets (preferred), fallback DisGeNET
        ot_genes = self.get_associations_opentargets(disease_name)
        if ot_genes:
            for g in ot_genes:
                g["cn"] = cite(g["source"], g["source_url"], f"{g['gene']} — {g['gene_name']}")
            evidence["genes"] = ot_genes
            evidence["sources_used"].append("Open Targets")
        else:
            dgn = self.get_associations_disgenet(disease_name)
            for g in dgn:
                g["cn"] = cite(g["source"], g["source_url"], g["gene"])
            evidence["genes"] = dgn
            if dgn:
                evidence["sources_used"].append("DisGeNET")

        # 4. Pathogenic variants from ClinVar
        for v in self.get_clinvar_variants(disease_name):
            v["cn"] = cite(v["source"], v["source_url"], v["variant_name"] or v["clinvar_id"])
            evidence["variants"].append(v)
        if evidence["variants"]:
            evidence["sources_used"].append("ClinVar")

        # 5. Drugs from ChEMBL
        for d in self.get_drugs(disease_name):
            d["cn"] = cite(d["source"], d["source_url"], d["drug_name"])
            evidence["drugs"].append(d)
        if evidence["drugs"]:
            evidence["sources_used"].append("ChEMBL")

        # 6. Pathways — use top gene from associations
        top_gene = ""
        if evidence["genes"]:
            top_gene = evidence["genes"][0].get("gene", "")
        if top_gene:
            for p in self.get_pathways(top_gene):
                p["cn"] = cite(p["source"], p["source_url"], p["pathway_name"])
                evidence["pathways"].append(p)
            if evidence["pathways"]:
                evidence["sources_used"].append("Reactome")

        # 7. Phenotypes from HPO
        for ph in self.get_phenotypes(disease_name):
            ph["cn"] = cite(ph["source"], ph["source_url"], ph["phenotype"])
            evidence["phenotypes"].append(ph)
        if evidence["phenotypes"]:
            evidence["sources_used"].append("HPO")

        # Deduplicate sources list
        seen = set()
        evidence["sources_used"] = [s for s in evidence["sources_used"] if not (s in seen or seen.add(s))]

        logger.info(
            f"DiseaseEngine: {len(evidence['all_citations'])} citations from "
            f"{len(evidence['sources_used'])} sources"
        )
        return evidence
