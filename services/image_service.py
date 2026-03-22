"""
image_service.py — Fetches disease-relevant images from Wikipedia + PMC open-access articles.
"""
import re
import logging
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "BioSage/1.0 (biomedical research platform; https://github.com/biosage)"}
_TIMEOUT = 12
_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_NCBI_PARAMS = "tool=biosage&email=biosage@research.ai"


async def fetch_disease_images(disease: str, pmids: list[str] | None = None) -> list[dict]:
    """
    Returns up to 5 image dicts: {url, caption, source, source_url}
    Sources: Wikipedia thumbnail + PMC open-access article figures.
    """
    results: list[dict] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
        # 1. Wikipedia — always first, most reliable
        wiki = await _wiki_image(client, disease)
        if wiki:
            results.append(wiki)

        # 2. PMC figures from cited PubMed articles (if provided)
        if pmids:
            for pmid in pmids[:4]:
                if len(results) >= 5:
                    break
                figs = await _pmc_figures_from_pmid(client, pmid)
                results.extend(figs[:2])

        # 3. If still < 2 images, do a PMC open-access search for the disease
        if len(results) < 2:
            figs = await _pmc_figures_search(client, disease)
            for f in figs:
                if len(results) >= 5:
                    break
                if not any(r["url"] == f["url"] for r in results):
                    results.append(f)

    return results[:5]


async def _wiki_image(client: httpx.AsyncClient, disease: str) -> Optional[dict]:
    slug = disease.strip().replace(" ", "_")
    try:
        r = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
        if r.status_code != 200:
            return None
        d = r.json()
        src = d.get("thumbnail", {}).get("source", "")
        if not src:
            return None
        # Upgrade to higher resolution
        hi_res = re.sub(r"/\d+px-", "/600px-", src)
        return {
            "url": hi_res,
            "caption": d.get("description", disease).capitalize(),
            "source": "Wikipedia",
            "source_url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception as e:
        logger.debug("Wiki image failed for %s: %s", disease, e)
        return None


async def _pmcid_from_pmid(client: httpx.AsyncClient, pmid: str) -> Optional[str]:
    """Convert a PubMed ID to a PMC ID via elink."""
    try:
        r = await client.get(
            f"{_NCBI_BASE}/elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json&{_NCBI_PARAMS}"
        )
        if r.status_code != 200:
            return None
        data = r.json()
        for ls in data.get("linksets", []):
            for ld in ls.get("linksetdbs", []):
                if ld.get("dbto") == "pmc":
                    links = ld.get("links", [])
                    if links:
                        return str(links[0])
    except Exception:
        pass
    return None


async def _pmc_figures_from_pmid(client: httpx.AsyncClient, pmid: str) -> list[dict]:
    pmcid = await _pmcid_from_pmid(client, pmid)
    if not pmcid:
        return []
    return await _extract_pmc_figures(client, pmcid)


async def _pmc_figures_search(client: httpx.AsyncClient, disease: str) -> list[dict]:
    """Search PMC for open-access disease articles and extract figures."""
    try:
        r = await client.get(
            f"{_NCBI_BASE}/esearch.fcgi?db=pmc&term={disease}+AND+open+access[filter]"
            f"&retmax=4&retmode=json&{_NCBI_PARAMS}"
        )
        if r.status_code != 200:
            return []
        ids = r.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

    figs: list[dict] = []
    for pmcid in ids[:3]:
        if len(figs) >= 4:
            break
        found = await _extract_pmc_figures(client, pmcid)
        figs.extend(found[:2])
    return figs


async def _extract_pmc_figures(client: httpx.AsyncClient, pmcid: str) -> list[dict]:
    """Fetch full PMC XML and extract <fig> image URLs + captions."""
    try:
        r = await client.get(
            f"{_NCBI_BASE}/efetch.fcgi?db=pmc&id={pmcid}&rettype=full&retmode=xml&{_NCBI_PARAMS}",
            timeout=20,
        )
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
    except Exception as e:
        logger.debug("PMC fetch failed PMC%s: %s", pmcid, e)
        return []

    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
    figs: list[dict] = []

    for fig in root.iter("fig"):
        # Caption
        cap_el = fig.find(".//caption//p")
        if cap_el is None:
            cap_el = fig.find(".//caption")
        caption = ("".join(cap_el.itertext()).strip()[:160] if cap_el is not None else "")

        # Figure label (e.g. "Figure 1")
        label_el = fig.find("label")
        label = label_el.text.strip() if label_el is not None and label_el.text else ""

        for g in fig.iter("graphic"):
            href = g.get("{http://www.w3.org/1999/xlink}href", "")
            if not href:
                continue
            # href may or may not include extension
            if not re.search(r"\.(jpg|jpeg|png|gif|tif)$", href, re.IGNORECASE):
                href_jpg = href + ".jpg"
            else:
                href_jpg = href
            img_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/bin/{href_jpg}"
            figs.append({
                "url": img_url,
                "caption": f"{label + ': ' if label else ''}{caption}",
                "source": f"PMC{pmcid}",
                "source_url": article_url,
            })

    return figs[:3]
