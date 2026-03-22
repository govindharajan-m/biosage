"""
PubMed literature mining service.
Uses NCBI E-utilities (free, no auth required for basic use).
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests

from config import NCBI_API_KEY, NCBI_BASE, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def _get(url: str, params: dict) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r
    except Exception as e:
        logger.warning(f"PubMed request failed {url}: {e}")
        return None


class PubMedService:

    def search(self, query: str, max_results: int = 10) -> List[dict]:
        """Full-text search PubMed; returns list of article dicts."""

        # ── Step 1: esearch — get PMIDs ────────────────────────────────
        search_params = {
            "db":      "pubmed",
            "term":    query,
            "retmax":  max_results,
            "retmode": "json",
            "sort":    "relevance",
        }
        if NCBI_API_KEY:
            search_params["api_key"] = NCBI_API_KEY

        r = _get(f"{NCBI_BASE}/esearch.fcgi", search_params)
        if not r:
            return []
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # ── Step 2: esummary — titles, journals, dates ─────────────────
        summary_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        if NCBI_API_KEY:
            summary_params["api_key"] = NCBI_API_KEY

        r2 = _get(f"{NCBI_BASE}/esummary.fcgi", summary_params)
        if not r2:
            return []
        result_set = r2.json().get("result", {})

        # ── Step 3: efetch — abstracts (top 6 only) ────────────────────
        abstract_params = {
            "db":      "pubmed",
            "id":      ",".join(ids[:6]),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if NCBI_API_KEY:
            abstract_params["api_key"] = NCBI_API_KEY

        abstracts: dict = {}
        r3 = _get(f"{NCBI_BASE}/efetch.fcgi", abstract_params)
        if r3:
            abstracts = _parse_abstracts(r3.text)

        # ── Assemble papers ────────────────────────────────────────────
        papers = []
        for uid in result_set.get("uids", []):
            art = result_set.get(uid, {})
            if not art or isinstance(art, str):
                continue

            authors = art.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            pub_date = art.get("pubdate", "")
            year = pub_date[:4] if pub_date else ""

            papers.append({
                "pmid":       uid,
                "title":      art.get("title", "Untitled"),
                "authors":    author_str,
                "journal":    art.get("source", ""),
                "year":       year,
                "abstract":   abstracts.get(uid, "")[:600],
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "doi":        art.get("elocationid", ""),
            })

        return papers

    def search_for_variant(self, rsid: str = "", gene: str = "", disease: str = "") -> List[dict]:
        """Build a targeted query for a genomic variant."""
        terms = []
        if rsid:
            terms.append(f'"{rsid}"')
        if gene:
            terms.append(f'"{gene}"[Gene]')
        if disease:
            terms.append(f'"{disease}"[MeSH Terms]')
        if not terms:
            return []
        return self.search(" AND ".join(terms), max_results=8)

    def search_for_gene(self, gene: str) -> List[dict]:
        """General gene-focused search."""
        return self.search(
            f'"{gene}"[Gene] AND (variant OR mutation OR pathogenic)',
            max_results=8,
        )


def _parse_abstracts(xml_text: str) -> dict:
    abstracts = {}
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            if pmid_el is None:
                continue
            pmid = pmid_el.text
            texts = article.findall(".//AbstractText")
            if texts:
                abstracts[pmid] = " ".join(
                    (el.text or "") for el in texts if el.text
                )
    except ET.ParseError as e:
        logger.warning(f"Abstract XML parse error: {e}")
    return abstracts
