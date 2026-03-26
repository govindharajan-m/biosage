"""
BioSage — FastAPI Backend v4.0
Perplexity-style biomedical intelligence platform.

Architecture
────────────
Chat Engine (fast):  /api/query/stream  — SSE streaming, parallel retrieval,
                                          PubMed fallback, instant first tokens.
Research Engine:     /api/variant, /api/vcf/upload, /api/acmg
                     Heavy work offloaded via BackgroundTasks or async gather.

Endpoints
─────────
GET  /                              Web UI
GET  /health                        Health + DB stats
POST /api/query/stream              Primary SSE streaming endpoint
POST /api/query                     Non-streaming (legacy / simple clients)
POST /api/chat                      Legacy chat alias

GET  /api/variant/{rsid}            Multi-DB variant aggregation + ACMG
GET  /api/acmg/{rsid}               Standalone ACMG report
GET  /api/pubmed                    Literature search
GET  /api/gene/{symbol}             Gene overview

POST /api/vcf/upload                VCF file → risk report (async annotate)
GET  /api/cache/stats               Cache diagnostics
POST /api/cache/flush               Invalidate cache

GET  /api/workspaces                List workspaces
POST /api/workspaces                Create workspace
GET  /api/workspaces/{id}           Workspace details
DELETE /api/workspaces/{id}         Delete workspace

GET  /api/analyses                  List analyses (?workspace_id= &limit=)
POST /api/analyses                  Save analysis
GET  /api/analyses/{id}             Get analysis
PATCH /api/analyses/{id}            Rename analysis
DELETE /api/analyses/{id}           Delete analysis
GET  /api/analyses/{id}/export      Export JSON download
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from cache import query_cache
from rag_engine import RAGEngine
from services.acmg_classifier import generate_report as acmg_report
from services.database import (
    DEFAULT_WORKSPACE_ID,
    create_workspace,
    delete_analysis,
    delete_workspace,
    get_analysis,
    get_workspace,
    init_db,
    list_analyses,
    list_workspaces,
    rename_analysis,
    save_analysis,
    workspace_stats,
)
from services.disease_engine import DiseaseEngine
from services.image_service import fetch_disease_images
from services.pubmed_service import PubMedService
from services.query_normalizer import fast_classify, fuzzy_correct, normalize_query
from services.variant_aggregator import VariantAggregator
from services.vcf_parser import build_risk_report, parse_vcf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

rag:        Optional[RAGEngine]        = None
aggregator: Optional[VariantAggregator] = None
pubmed:     Optional[PubMedService]    = None
disease:    Optional[DiseaseEngine]    = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global rag, aggregator, pubmed, disease
    init_db()
    logger.info("Initializing services…")
    rag        = RAGEngine()
    aggregator = VariantAggregator()
    pubmed     = PubMedService()
    disease    = DiseaseEngine()
    logger.info("BioSage v4.0 ready — chat engine online")
    yield


app = FastAPI(
    title="BioSage API",
    version="4.0.0",
    docs_url="/api/docs",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# ── Auth & Rate-limit ─────────────────────────────────────────────────────────

# Optional tester API-key gate.
# Set TESTER_KEYS="key1,key2,key3" in env to enable; leave blank to disable.
_TESTER_KEYS: set[str] = {
    k.strip() for k in os.getenv("TESTER_KEYS", "").split(",") if k.strip()
}

def _check_api_key(x_api_key: str = Header(None)):
    if _TESTER_KEYS and x_api_key not in _TESTER_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")

# Simple in-memory rate limiter: max N requests per IP per 60 s window.
_RATE_LIMIT   = int(os.getenv("RATE_LIMIT_RPM", "20"))  # requests per minute per IP
_rate_buckets: dict[str, list[float]] = defaultdict(list)

def _rate_limit(request: Request):
    if _RATE_LIMIT <= 0:
        return
    ip  = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _rate_buckets[ip]
    # Evict timestamps older than 60 s
    _rate_buckets[ip] = [t for t in bucket if now - t < 60]
    if len(_rate_buckets[ip]) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded — please wait a moment")
    _rate_buckets[ip].append(now)

# ── Pydantic models ───────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryItem]] = []
    workspace_id: Optional[str] = DEFAULT_WORKSPACE_ID
    save: Optional[bool] = False


class QueryRequest(BaseModel):
    q: str
    history: Optional[List[HistoryItem]] = []
    workspace_id: Optional[str] = DEFAULT_WORKSPACE_ID
    save: Optional[bool] = False
    bandwidth: Optional[str] = "high"   # "low" | "high"
    mode: Optional[str] = "research"    # "research" | "clinical" | "quick"


class SaveAnalysisRequest(BaseModel):
    query: str
    type: str
    result: dict
    name: Optional[str] = None
    workspace_id: Optional[str] = DEFAULT_WORKSPACE_ID


class WorkspaceCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    color: Optional[str] = "#3B82F6"


class RenameRequest(BaseModel):
    name: str


# ── Static / UI ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        stats = workspace_stats(DEFAULT_WORKSPACE_ID)
    except Exception:
        stats = {}
    return {"status": "ok", "version": "4.1.0", "workspace_stats": stats}


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Background save helper ────────────────────────────────────────────────────

def _bg_save(query: str, atype: str, result: dict,
             workspace_id: str = DEFAULT_WORKSPACE_ID) -> None:
    """Fire-and-forget analysis persistence (safe to run in thread pool)."""
    try:
        save_analysis(query, atype, result, workspace_id=workspace_id)
    except Exception as exc:
        logger.warning("bg_save failed for '%s': %s", query[:40], exc)


# ── Primary SSE streaming endpoint ───────────────────────────────────────────

@app.post("/api/query/stream", dependencies=[Depends(_check_api_key), Depends(_rate_limit)])
async def stream_query(request: QueryRequest):
    """
    Perplexity-style SSE streaming endpoint.

    Event sequence (disease example):
      meta → classified → data(genes) → data(variants) → data(drugs) →
      data(phenotypes) → token... → llm_done → followups → citations → done

    Chat (follow-up) sequence:
      meta → token... → llm_done → citations → followups → done

    Cached response:
      [all previous events replayed] → done(cached:true)
    """
    raw_q = request.q.strip()
    if not raw_q:
        raise HTTPException(400, "Query cannot be empty")

    # Normalise synonyms / abbreviations (e.g. "cf" → "cystic fibrosis")
    q = normalize_query(raw_q)
    if q != raw_q:
        logger.info("Query normalised: '%s' → '%s'", raw_q, q)

    history     = [{"role": h.role, "content": h.content} for h in (request.history or [])]
    workspace   = request.workspace_id or DEFAULT_WORKSPACE_ID
    loop        = asyncio.get_running_loop()
    has_history = bool(history)
    low_bw      = (request.bandwidth or "high") == "low"
    mode        = (request.mode or "research").lower()
    if mode not in ("research", "clinical", "quick"):
        mode = "research"

    # ── Async bridge: sync generator → async generator ────────────────────────
    async def _bridge(gen_fn, *args):
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _producer():
            try:
                for item in gen_fn(*args):
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result(timeout=10)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "message": str(exc)}), loop
                ).result(timeout=5)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)

        loop.run_in_executor(None, _producer)
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=45)
            if item is None:
                break
            yield item

    async def generate():
        nonlocal q
        try:
            # ── Cache check ───────────────────────────────────────────────────
            cache_key = query_cache.make_key("qs4", q, "h" if has_history else "", "lo" if low_bw else "", mode)
            cached    = query_cache.get(cache_key)
            if cached:
                logger.info("Cache hit: '%s'", q[:50])
                for ev in cached:
                    yield _sse(ev)
                yield _sse({"type": "done", "cached": True})
                return

            yield _sse({"type": "meta", "query": q, "original": raw_q})

            # ── Conversational follow-up (chat mode) ──────────────────────────
            if has_history:
                accumulated = ""
                async for ev in _bridge(rag.stream_chat_sync, q, history, 6, pubmed):
                    if ev["type"] == "token":
                        accumulated += ev["text"]
                    yield _sse(ev)
                    if ev["type"] == "llm_done":
                        cits = ev.get("citations", [])
                        yield _sse({"type": "citations", "citations": cits})
                        break

                followups = await loop.run_in_executor(
                    None, lambda: rag._generate_followups(q, accumulated[:400])
                )
                yield _sse({"type": "followups", "followups": followups})

                # Auto-save chat response
                loop.run_in_executor(None, _bg_save, q, "chat", {
                    "query": q, "query_type": "chat",
                    "answer": accumulated,
                    "followups": followups,
                    "citations": cits if "cits" in locals() else [],
                }, workspace)

                yield _sse({"type": "done"})
                return

            # ── Intent classification ─────────────────────────────────────────
            # Fast path: keyword/regex classifier — no LLM call required
            if re.match(r"^rs\d+$", q.lower()):
                qt = "variant"
            else:
                qt = fast_classify(q)
                if qt is None:
                    # Ambiguous — fall back to LLM classifier
                    qt = await loop.run_in_executor(None, lambda: rag.classify_query(q))
                    logger.info("LLM classified '%s' → %s", q[:50], qt)
                else:
                    logger.info("Fast-classified '%s' → %s", q[:50], qt)

            yield _sse({"type": "classified", "query_type": qt})

            # ── Variant ───────────────────────────────────────────────────────
            if qt == "variant":
                # Parallel lookup across ClinVar + Ensembl + dbSNP + UniProt
                data  = await aggregator.async_aggregate(q.lower())
                data["acmg"] = await loop.run_in_executor(None, lambda: acmg_report(data))
                yield _sse({"type": "variant_data", "data": data})

                accumulated = ""
                async for ev in _bridge(
                    rag.stream_chat_sync,
                    f"Explain the variant {q} and its clinical significance, "
                    f"including disease associations and ACMG classification.",
                    None, 6, pubmed,
                ):
                    if ev["type"] == "token":
                        accumulated += ev["text"]
                    yield _sse(ev)
                    if ev["type"] == "llm_done":
                        yield _sse({"type": "citations", "citations": ev.get("citations", [])})
                        break

                loop.run_in_executor(None, _bg_save, q, "variant", {
                    "query": q, "query_type": "variant",
                    "answer": accumulated,
                    "evidence": data,
                }, workspace)

                yield _sse({"type": "done"})
                return

            # ── Disease ───────────────────────────────────────────────────────
            if qt == "disease":
                # Fuzzy typo correction — attempt before fetching
                corrected_name, corr_score = await loop.run_in_executor(
                    None, fuzzy_correct, q
                )
                if corrected_name:
                    q = corrected_name
                    yield _sse({
                        "type": "meta", "query": q, "original": raw_q,
                        "correction": True,
                        "message": f'Interpreted as: "{q}"',
                    })

                # Progressive fetch — stream data events as each DB responds,
                # then get the complete evidence bundle for LLM synthesis.
                ev: dict = {}
                cache_events: list = []

                async for chunk in disease.stream_aggregate(q):
                    if chunk["type"] == "partial":
                        key  = chunk["key"]
                        data = chunk["data"]
                        # Map internal keys to frontend display keys
                        display_key = {
                            "opentargets": "genes", "disgenet": "genes",
                            "clinvar": "variants", "drugs": "drugs",
                            "phenotypes": "phenotypes", "pathways": "pathways",
                        }.get(key)
                        if display_key and data:
                            data_ev = {"type": "data", "key": display_key, "data": (data or [])[:14]}
                            yield _sse(data_ev)
                            cache_events.append(data_ev)
                    elif chunk["type"] == "complete":
                        ev = chunk["evidence"]

                # Quality gate — block hallucination for unknown/nonsense queries.
                # Require *hard biomedical evidence* (genes, variants, or ≥3 citations
                # from authoritative DBs). A bare MedGen definition is NOT enough —
                # MedGen returns spurious matches for nonsense queries.
                citations  = ev.get("all_citations", [])
                auth_cits  = [c for c in citations
                              if c.get("db", "") not in ("NCBI MedGen",)]
                overview   = ev.get("overview") or {}
                definition = overview.get("definition", "") or ""
                # The disease name must appear in the definition (prevents topic drift)
                name_in_def = q.split()[0].lower() in definition.lower() if definition else False
                has_real   = (
                    len(ev.get("genes", []))    > 0
                    or len(ev.get("variants", [])) > 0
                    or len(auth_cits)              >= 2
                    or (len(definition.strip())    >= 60 and name_in_def)
                )
                if not has_real:
                    hint = (f' Did you mean "{corrected_name}"?'
                            if corrected_name else
                            " Please check the spelling or use a recognised medical term.")
                    yield _sse({
                        "type": "error",
                        "message": (
                            f'"{raw_q}" is not a recognised disease or medical condition '
                            f'in any of our databases (ClinVar, OMIM, MedGen, Open Targets).{hint}'
                        ),
                    })
                    yield _sse({"type": "done"})
                    return

                # PubMed fallback: if evidence is thin, supplement with literature
                if (len(ev.get("genes", [])) < 3 and len(ev.get("variants", [])) < 3
                        and not low_bw):
                    try:
                        pm_papers = await loop.run_in_executor(
                            None, lambda: pubmed.search(q, max_results=6)
                        )
                        if pm_papers:
                            pm_cits = [
                                {"source": "PubMed", "label": p.get("title", "")[:80],
                                 "url": p.get("url", ""), "id": p.get("pmid", "")}
                                for p in pm_papers
                            ]
                            ev.setdefault("all_citations", []).extend(pm_cits)
                            citations = ev["all_citations"]
                            pm_ev = {"type": "data", "key": "literature", "data": pm_papers[:4]}
                            yield _sse(pm_ev)
                            cache_events.append(pm_ev)
                    except Exception as _pm_e:
                        logger.debug("PubMed fallback error: %s", _pm_e)

                # Kick off Immediate Response in parallel (skipped in low-bandwidth mode)
                ir_future = (
                    None if low_bw
                    else loop.run_in_executor(None, rag.generate_immediate_response_sync, q, mode)
                )

                # Stream LLM synthesis (starts immediately — DB results already sent)
                accumulated = ""
                async for token_ev in _bridge(rag.stream_synthesis_sync, ev, low_bw, mode):
                    if token_ev["type"] == "token":
                        accumulated += token_ev["text"]
                    yield _sse(token_ev)
                    if token_ev["type"] == "llm_done":
                        break

                parsed = rag._parse_structured_text(accumulated)
                # Low-bandwidth: trim citations to top 3 to reduce payload
                cits = ev.get("all_citations", [])
                if low_bw:
                    cits = cits[:3]

                fu_ev  = {"type": "followups", "followups": parsed.get("followups", [])}
                cit_ev = {"type": "citations",  "citations": cits}
                yield _sse(fu_ev)
                yield _sse(cit_ev)
                cache_events.extend([fu_ev, cit_ev])

                # Emit Immediate Response section (generated in parallel — likely already done)
                if ir_future is not None:
                    ir_text = await ir_future
                    if ir_text:
                        ir_ev = {"type": "immediate_response", "content": ir_text}
                        yield _sse(ir_ev)
                        cache_events.append(ir_ev)

                # Auto-save and cache
                loop.run_in_executor(None, _bg_save, q, "disease", {
                    "query": q, "query_type": "disease", "mode": mode,
                    "sections":  parsed.get("sections", []),
                    "followups": parsed.get("followups", []),
                    "citations": cits,
                    "evidence":  ev,
                    "immediate_response": ir_text if ir_future is not None and "ir_text" in locals() else "",
                }, workspace)
                query_cache.set(cache_key, cache_events)

                yield _sse({"type": "done"})
                return

            # ── Gene ──────────────────────────────────────────────────────────
            if qt == "gene":
                # Extract gene symbol from query (last ALL-CAPS token fallback)
                sym_match = re.search(r"\b([A-Z][A-Z0-9]{1,7})\b", q)
                symbol    = sym_match.group(1) if sym_match else q.split()[-1].upper()

                uniprot_data, papers = await asyncio.gather(
                    loop.run_in_executor(None, lambda: aggregator.query_uniprot(symbol)),
                    loop.run_in_executor(None, lambda: pubmed.search_for_gene(symbol)),
                )
                # Gene quality gate — reject nonsense symbols.
                # UniProt is authoritative; it uses protein_name/gene/accession keys.
                gene_known = bool(
                    uniprot_data and (
                        uniprot_data.get("gene")
                        or uniprot_data.get("protein_name")
                        or uniprot_data.get("accession")
                    )
                )
                if not gene_known:
                    yield _sse({
                        "type": "error",
                        "message": (
                            f'"{raw_q}" does not match a recognised gene symbol or name. '
                            f"Please use a standard HGNC gene symbol (e.g. BRCA1, CFTR, TP53)."
                        ),
                    })
                    yield _sse({"type": "done"})
                    return

                gene_ev = {
                    "type": "gene_data",
                    "data": {
                        "symbol":     symbol,
                        "uniprot":    uniprot_data,
                        "literature": (papers or [])[:8],
                    },
                }
                yield _sse(gene_ev)

                gene_prompt = (
                    f"Explain the gene {symbol}: its molecular function, protein product, "
                    f"known disease associations, pathogenic variants, and clinical significance."
                )
                accumulated = ""
                async for ev in _bridge(rag.stream_chat_sync, gene_prompt, None, 6, pubmed):
                    if ev["type"] == "token":
                        accumulated += ev["text"]
                    yield _sse(ev)
                    if ev["type"] == "llm_done":
                        yield _sse({"type": "citations", "citations": ev.get("citations", [])})
                        break

                followups = await loop.run_in_executor(
                    None, lambda: rag._generate_followups(q, accumulated[:400])
                )
                yield _sse({"type": "followups", "followups": followups})

                loop.run_in_executor(None, _bg_save, q, "gene", {
                    "query": q, "query_type": "gene",
                    "answer": accumulated,
                    "followups": followups,
                    "evidence": {"symbol": symbol, "uniprot": uniprot_data},
                }, workspace)

                yield _sse({"type": "done"})
                return

            # ── Drug / general / literature fallback ──────────────────────────
            accumulated = ""
            async for ev in _bridge(rag.stream_chat_sync, q, None, 6, pubmed):
                if ev["type"] == "token":
                    accumulated += ev["text"]
                yield _sse(ev)
                if ev["type"] == "llm_done":
                    yield _sse({"type": "citations", "citations": ev.get("citations", [])})
                    break

            followups = await loop.run_in_executor(
                None, lambda: rag._generate_followups(q, accumulated[:400])
            )
            yield _sse({"type": "followups", "followups": followups})

            loop.run_in_executor(None, _bg_save, q, "chat", {
                "query": q, "query_type": qt,
                "answer": accumulated,
                "followups": followups,
            }, workspace)

            yield _sse({"type": "done"})

        except Exception as exc:
            logger.error("Stream error for '%s': %s", q[:50], exc, exc_info=True)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ── Non-streaming unified query (legacy / simple clients) ────────────────────

@app.post("/api/query")
async def unified_query(request: QueryRequest):
    """
    Non-streaming version. Routes to the same logic as /api/query/stream
    but collects the full response before returning.
    """
    q = normalize_query(request.q.strip())
    if not q:
        raise HTTPException(400, "Query cannot be empty")

    history  = [{"role": h.role, "content": h.content} for h in (request.history or [])]
    loop     = asyncio.get_running_loop()
    workspace = request.workspace_id or DEFAULT_WORKSPACE_ID

    if history:
        rag_result = rag.answer(q, history=history, pubmed_svc=pubmed)
        result = {
            "query":      q,
            "query_type": "chat",
            "answer":     rag_result.get("answer", ""),
            "sections":   [],
            "followups":  rag_result.get("followups", []),
            "citations":  rag_result.get("citations", []),
            "evidence":   {},
        }
        if request.save:
            save_analysis(q, "chat", result, workspace_id=workspace)
        return result

    if re.match(r"^rs\d+$", q.lower()):
        qt = "variant"
    else:
        qt = fast_classify(q) or await loop.run_in_executor(None, lambda: rag.classify_query(q))

    if qt == "variant":
        data     = await aggregator.async_aggregate(q.lower())
        data["acmg"] = await loop.run_in_executor(None, lambda: acmg_report(data))
        rag_result   = rag.answer(f"Explain the variant {q} and its clinical significance",
                                  pubmed_svc=pubmed)
        result = {
            "query": q, "query_type": "variant",
            "answer": rag_result.get("answer", ""),
            "sections": [], "followups": rag_result.get("followups", []),
            "citations": rag_result.get("citations", []), "evidence": data,
        }
    elif qt == "disease":
        ev     = await disease.async_aggregate(q)
        report = rag.synthesize_disease_report(ev)
        result = {
            "query": q, "query_type": "disease",
            "answer": "",
            "sections":  report.get("sections", []),
            "followups": report.get("followups", []),
            "citations": ev.get("all_citations", []),
            "evidence":  ev,
        }
    elif qt == "gene":
        sym_match = re.search(r"\b([A-Z][A-Z0-9]{1,7})\b", q)
        symbol    = sym_match.group(1) if sym_match else q.split()[-1].upper()
        rag_result = rag.answer(
            f"Explain the gene {symbol}: function, protein product, disease associations.",
            pubmed_svc=pubmed,
        )
        result = {
            "query": q, "query_type": "gene",
            "answer": rag_result.get("answer", ""),
            "sections": [], "followups": rag_result.get("followups", []),
            "citations": rag_result.get("citations", []), "evidence": {},
        }
    else:
        rag_result = rag.answer(q, pubmed_svc=pubmed)
        result = {
            "query": q, "query_type": qt,
            "answer": rag_result.get("answer", ""),
            "sections": [], "followups": rag_result.get("followups", []),
            "citations": rag_result.get("citations", []), "evidence": {},
        }

    if request.save:
        save_analysis(q, qt, result, workspace_id=workspace)
    return result


# ── Cache management ──────────────────────────────────────────────────────────

@app.get("/api/cache/stats")
async def cache_stats_endpoint():
    return query_cache.stats()


@app.post("/api/cache/flush")
async def cache_flush_endpoint():
    query_cache.flush()
    return {"ok": True}


# ── Disease (non-streaming GET) ───────────────────────────────────────────────

@app.get("/api/disease/{name}")
async def get_disease(
    name: str,
    save: bool = Query(False),
    workspace_id: str = Query(DEFAULT_WORKSPACE_ID),
):
    """Full disease report: parallel multi-DB aggregation + LLM synthesis."""
    name = normalize_query(name.strip())
    if not name:
        raise HTTPException(400, "Disease name required")
    try:
        ev      = await disease.async_aggregate(name)
        report  = rag.synthesize_disease_report(ev)
        data    = {
            "query":      name,
            "query_type": "disease",
            "sections":   report.get("sections", []),
            "followups":  report.get("followups", []),
            "citations":  ev.get("all_citations", []),
            "evidence":   ev,
        }
        if save:
            save_analysis(name, "disease", data, workspace_id=workspace_id)
        return data
    except Exception as e:
        logger.error("Disease error '%s': %s", name, e, exc_info=True)
        raise HTTPException(500, f"Disease query failed: {e}")


# ── Legacy chat endpoint ──────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    q      = normalize_query(request.message.strip())
    result = rag.answer(q, pubmed_svc=pubmed)
    if request.save:
        try:
            save_analysis(q, "chat", result, workspace_id=request.workspace_id or DEFAULT_WORKSPACE_ID)
        except Exception as exc:
            logger.warning("Chat save error: %s", exc)
    return result


# ── Variant aggregation (REST, non-streaming) ─────────────────────────────────

@app.get("/api/variant/{rsid}")
async def get_variant(
    rsid: str,
    save: bool = Query(False),
    workspace_id: str = Query(DEFAULT_WORKSPACE_ID),
):
    rsid = rsid.strip().lower()
    if not rsid.startswith("rs"):
        raise HTTPException(400, "rsID must start with 'rs'")
    try:
        data         = await aggregator.async_aggregate(rsid)
        data["acmg"] = acmg_report(data)
        if save:
            save_analysis(rsid, "variant", data, workspace_id=workspace_id)
        return data
    except Exception as e:
        logger.error("Variant error '%s': %s", rsid, e, exc_info=True)
        raise HTTPException(500, f"Variant lookup failed: {e}")


# ── ACMG classification ───────────────────────────────────────────────────────

@app.get("/api/acmg/{rsid}")
async def get_acmg(rsid: str):
    rsid = rsid.strip().lower()
    if not rsid.startswith("rs"):
        raise HTTPException(400, "rsID must start with 'rs'")
    try:
        variant_data = await aggregator.async_aggregate(rsid)
        return acmg_report(variant_data)
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Literature ────────────────────────────────────────────────────────────────

@app.get("/api/pubmed")
async def search_pubmed(
    q:       Optional[str] = Query(None),
    gene:    Optional[str] = Query(None),
    rsid:    Optional[str] = Query(None),
    disease: Optional[str] = Query(None),
):
    loop = asyncio.get_running_loop()
    if q:
        results = await loop.run_in_executor(None, lambda: pubmed.search(q, max_results=12))
    elif rsid or gene or disease:
        results = await loop.run_in_executor(
            None,
            lambda: pubmed.search_for_variant(rsid=rsid or "", gene=gene or "", disease=disease or ""),
        )
    else:
        raise HTTPException(400, "Provide at least one of: q, gene, rsid, disease")
    return {"papers": results, "count": len(results)}


# ── Gene overview ─────────────────────────────────────────────────────────────

@app.get("/api/gene/{symbol}")
async def get_gene(symbol: str):
    symbol = symbol.strip().upper()
    loop   = asyncio.get_running_loop()
    try:
        uniprot, papers, rag_result = await asyncio.gather(
            loop.run_in_executor(None, lambda: aggregator.query_uniprot(symbol)),
            loop.run_in_executor(None, lambda: pubmed.search_for_gene(symbol)),
            loop.run_in_executor(
                None,
                lambda: rag.answer(
                    f"Explain the gene {symbol}: its function, protein product, "
                    f"known disease associations, and clinical significance.",
                    pubmed_svc=pubmed,
                ),
            ),
        )
        return {
            "symbol":     symbol,
            "uniprot":    uniprot,
            "literature": (papers or [])[:6],
            "ai_summary": rag_result.get("answer", ""),
            "citations":  rag_result.get("citations", []),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── VCF upload (heavy annotation runs in background) ─────────────────────────

@app.post("/api/vcf/upload")
async def upload_vcf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    save: bool = Query(False),
    workspace_id: str = Query(DEFAULT_WORKSPACE_ID),
):
    fname = file.filename or ""
    if not fname.lower().endswith(".vcf"):
        raise HTTPException(400, "File must be a .vcf file")

    raw     = await file.read()
    content = raw.decode("utf-8", errors="replace")
    variants = parse_vcf(content)

    if not variants:
        raise HTTPException(422, "No variants parsed from VCF file")

    # Annotate up to 25 rs-tagged variants synchronously (fast enough for API response)
    loop      = asyncio.get_running_loop()
    annotated = 0
    for v in variants:
        if v.rsid and annotated < 25:
            try:
                cv = await loop.run_in_executor(None, lambda vid=v.rsid: aggregator.query_clinvar(vid))
                if cv:
                    v.clinical_significance = cv.get("clinical_significance")
                    v.gene                  = cv.get("gene")
                    v.associated_disease    = (cv.get("associated_diseases") or [None])[0]
                    v.evidence_sources      = ["ClinVar"]
                annotated += 1
            except Exception:
                pass

    report                   = build_risk_report(variants)
    report["filename"]       = fname
    report["annotated_count"] = annotated

    if save:
        background_tasks.add_task(_bg_save, fname, "vcf", report, workspace_id)

    return report


# ── Workspaces ────────────────────────────────────────────────────────────────

@app.get("/api/workspaces")
async def api_list_workspaces():
    return {"workspaces": list_workspaces()}


@app.post("/api/workspaces", status_code=201)
async def api_create_workspace(body: WorkspaceCreateRequest):
    ws = create_workspace(body.name, body.description or "", body.color or "#3B82F6")
    if not ws:
        raise HTTPException(500, "Failed to create workspace")
    return ws


@app.get("/api/workspaces/{workspace_id}")
async def api_get_workspace(workspace_id: str):
    ws = get_workspace(workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    ws["stats"] = workspace_stats(workspace_id)
    return ws


@app.delete("/api/workspaces/{workspace_id}", status_code=204)
async def api_delete_workspace(workspace_id: str):
    if workspace_id == DEFAULT_WORKSPACE_ID:
        raise HTTPException(400, "Cannot delete the default workspace")
    if not delete_workspace(workspace_id):
        raise HTTPException(404, "Workspace not found")


# ── Analyses ──────────────────────────────────────────────────────────────────

@app.get("/api/analyses")
async def api_list_analyses(
    workspace_id: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    return {"analyses": list_analyses(workspace_id=workspace_id, limit=limit)}


@app.post("/api/analyses", status_code=201)
async def api_save_analysis(body: SaveAnalysisRequest):
    row = save_analysis(
        query=body.query,
        analysis_type=body.type,
        result=body.result,
        name=body.name,
        workspace_id=body.workspace_id or DEFAULT_WORKSPACE_ID,
    )
    if not row:
        raise HTTPException(500, "Failed to save analysis")
    return row


@app.get("/api/analyses/{analysis_id}")
async def api_get_analysis(analysis_id: str):
    row = get_analysis(analysis_id)
    if not row:
        raise HTTPException(404, "Analysis not found")
    return row


@app.patch("/api/analyses/{analysis_id}")
async def api_rename_analysis(analysis_id: str, body: RenameRequest):
    if not rename_analysis(analysis_id, body.name.strip()):
        raise HTTPException(404, "Analysis not found")
    return {"ok": True}


@app.delete("/api/analyses/{analysis_id}", status_code=204)
async def api_delete_analysis(analysis_id: str):
    if not delete_analysis(analysis_id):
        raise HTTPException(404, "Analysis not found")


@app.get("/api/analyses/{analysis_id}/export")
async def api_export_analysis(analysis_id: str):
    row = get_analysis(analysis_id)
    if not row:
        raise HTTPException(404, "Analysis not found")
    filename = f"biosage_{row['type']}_{analysis_id[:8]}.json"
    return JSONResponse(
        content=row,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Disease images endpoint ───────────────────────────────────────────────────

@app.get("/api/disease-images")
async def api_disease_images(
    q: str = Query(..., description="Disease name"),
    pmids: str = Query("", description="Comma-separated PubMed IDs from citations"),
):
    """
    Returns up to 5 images for a disease:
    1. Wikipedia thumbnail (always first)
    2. Figures extracted from PMC open-access articles cited in the response
    """
    pmid_list = [p.strip() for p in pmids.split(",") if p.strip()] if pmids else []
    images = await fetch_disease_images(q, pmid_list)
    return JSONResponse({"images": images})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
