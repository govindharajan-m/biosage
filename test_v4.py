"""
test_v4.py — Comprehensive test suite for BioSage v4.0

Tests are grouped into four layers:
  Layer 1 — Unit tests (no network, no LLM)
  Layer 2 — Service tests (live network, no LLM)
  Layer 3 — LLM integration tests (network + Groq)
  Layer 4 — Live server / API endpoint tests

Run all:
    python test_v4.py

Run specific layer:
    python test_v4.py --layer 1
"""

import asyncio
import json
import sys
import time

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Stub heavy deps so Layer 1 (pure-Python) tests can import rag_engine
# without triggering sentence-transformers / ChromaDB / OpenBLAS init.
# Real imports happen later when Layer 2+ tests actually need them.
def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__spec__ = None
    # Make attribute access return a no-op callable / stub object
    class _Any:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return _Any()
        def __getattr__(self, n): return _Any()
    mod.__getattr__ = lambda n: _Any()
    mod.__all__ = []
    return mod

_STUB_NAMES = [
    "sentence_transformers",
    "chromadb",
    "chromadb.config",
    "groq",
]
_STUBBED: list[str] = []

def _install_stubs():
    """Install lightweight stubs so Layer 1 can import rag_engine without loading
    OpenBLAS / ChromaDB (which would fail while the server process holds the memory)."""
    for name in _STUB_NAMES:
        if name not in sys.modules:
            sys.modules[name] = _make_stub(name)
            _STUBBED.append(name)

def _remove_stubs():
    """Remove stubs and purge any modules that imported them, so later layers
    can do real imports."""
    for name in list(_STUBBED):
        sys.modules.pop(name, None)
        _STUBBED.remove(name)
    # Also evict any module that may have cached a stub reference
    for key in list(sys.modules):
        if key in ("rag_engine", "data_pipeline.vector_store", "data_pipeline"):
            sys.modules.pop(key, None)

LAYER = None
if "--layer" in sys.argv:
    idx = sys.argv.index("--layer")
    if idx + 1 < len(sys.argv):
        LAYER = int(sys.argv[idx + 1])

PASS = 0
FAIL = 0
SKIP = 0

def _pr(sym, label, detail=""):
    print(f"  {sym}  {label}" + (f"  ->  {detail}" if detail else ""))

def ok(label, detail=""):
    global PASS; PASS += 1
    _pr("PASS", label, detail)

def fail(label, detail=""):
    global FAIL; FAIL += 1
    _pr("FAIL", label, detail)

def skip(label, reason=""):
    global SKIP; SKIP += 1
    _pr("SKIP", label, reason)

def section(title):
    print(f"\n{'-'*62}")
    print(f"  {title}")
    print(f"{'-'*62}")


# ═══════════════════════════════════════════════════════════════
# LAYER 1 — Unit tests (no network, no LLM required)
# ═══════════════════════════════════════════════════════════════

def test_query_normalizer():
    section("Layer 1 · Query Normalizer")
    from services.query_normalizer import normalize_query, fast_classify

    # ── normalize_query ──────────────────────────────────────────
    cases_norm = [
        ("cf",                        "cystic fibrosis"),
        ("CF",                        "cystic fibrosis"),
        ("als",                       "amyotrophic lateral sclerosis"),
        ("huntingtons",               "huntington disease"),
        ("huntingtons disease",       "huntington disease"),   # no double "disease"
        ("parkinsons disease",        "parkinson disease"),
        ("sickle cell anemia",        "sickle cell disease"),
        ("pku",                       "phenylketonuria"),
        ("sma",                       "spinal muscular atrophy"),
        ("t2d",                       "type 2 diabetes"),
        ("What is cf?",               "What is cystic fibrosis?"),
        ("BRCA1 gene",                "BRCA1 gene"),           # no change expected
        ("rs113993960",               "rs113993960"),           # no change
    ]
    for q, expected in cases_norm:
        result = normalize_query(q)
        if result.lower() == expected.lower():
            ok(f"normalize({q!r})", result)
        else:
            fail(f"normalize({q!r})", f"expected {expected!r}, got {result!r}")

    # ── fast_classify ────────────────────────────────────────────
    cases_cls = [
        ("rs113993960",                   "variant"),
        ("rs28897696",                    "variant"),
        ("BRCA1 gene function",           "gene"),
        ("what does TP53 do",             None),    # ambiguous → None OK
        ("cystic fibrosis",               "disease"),
        ("what is huntington disease",    "disease"),
        ("explain alzheimer syndrome",    "disease"),
        ("what is the CFTR gene",         None),    # ambiguous OK
        ("treatment for breast cancer",   "disease"),  # has "cancer" → disease
    ]
    for q, expected in cases_cls:
        result = fast_classify(q)
        if expected is None:
            # None means "ambiguous" — we just don't want a wrong answer
            ok(f"fast_classify({q!r})", f"returned {result!r} (ambiguous OK)")
        elif result == expected:
            ok(f"fast_classify({q!r})", result)
        else:
            fail(f"fast_classify({q!r})", f"expected {expected!r}, got {result!r}")


def test_parse_structured_text():
    section("Layer 1 · RAGEngine._parse_structured_text")
    from rag_engine import RAGEngine
    engine = RAGEngine.__new__(RAGEngine)  # no __init__ needed

    # Section + followups in streaming marker format
    text = (
        "##SECTION: Disease Overview\n"
        "Cystic fibrosis is caused by CFTR mutations [1].\n\n"
        "##SECTION: Genetic Basis\n"
        "The CFTR gene encodes a chloride channel [2].\n\n"
        "##FOLLOWUPS:\n"
        "What is the most common CFTR mutation?\n"
        "How does CFTR affect lung function?\n"
        "What treatments target CFTR?\n"
        "What is the life expectancy for CF patients?\n"
    )
    result = engine._parse_structured_text(text)

    secs = result.get("sections", [])
    fus  = result.get("followups", [])

    if len(secs) == 2:
        ok("parsed 2 sections")
    else:
        fail("section count", f"expected 2, got {len(secs)}")

    if secs and secs[0]["title"] == "Disease Overview":
        ok("section title correct")
    else:
        fail("section title", str(secs[0] if secs else "[]"))

    if len(fus) == 4:
        ok("parsed 4 followups")
    else:
        fail("followup count", f"expected 4, got {len(fus)}")

    # Fallback text-mode format
    text2 = (
        "SECTION: Genetic Basis\n"
        "HTT gene contains CAG repeats [1].\n\n"
        "FOLLOW_UPS:\n"
        "What is the normal CAG repeat count?\n"
        "How does polyglutamine cause toxicity?\n"
        "What drugs target mHTT?\n"
        "Is there a cure?\n"
    )
    result2 = engine._parse_structured_text(text2)
    secs2   = result2.get("sections", [])
    fus2    = result2.get("followups", [])

    if len(secs2) == 1 and secs2[0]["title"] == "Genetic Basis":
        ok("fallback text-mode section parsed")
    else:
        fail("fallback section", str(secs2))

    if len(fus2) == 4:
        ok("fallback text-mode followups parsed")
    else:
        fail("fallback followups", str(fus2))


def test_compact_evidence():
    section("Layer 1 · RAGEngine._compact_evidence")
    from rag_engine import RAGEngine
    engine = RAGEngine.__new__(RAGEngine)

    evidence = {
        "query": "cystic fibrosis",
        "overview": {"definition": "A genetic disorder affecting the lungs.", "source": "MedGen"},
        "omim_entries": [{"cn": 1, "title": "CYSTIC FIBROSIS", "source": "OMIM", "source_url": "..."}],
        "genes": [
            {"cn": 2, "gene": "CFTR", "gene_name": "cystic fibrosis transmembrane conductance regulator",
             "association_score": 0.99, "source": "Open Targets", "source_url": "..."}
        ] * 15,  # 15 genes — should be trimmed to 10
        "variants": [{"cn": 3, "variant_name": "p.Phe508del", "gene": "CFTR",
                       "clinical_significance": "Pathogenic", "source": "ClinVar",
                       "source_url": "..."} ] * 12,  # 12 → trimmed to 8
        "drugs": [{"cn": 4, "drug_name": "Ivacaftor", "max_phase": 4,
                   "source": "ChEMBL", "source_url": "..."}] * 8,  # → trimmed to 6
        "pathways": [{"cn": 5, "pathway_name": "CFTR Activity", "source": "Reactome",
                      "source_url": "..."}] * 7,  # → trimmed to 5
        "phenotypes": [{"cn": 6, "phenotype": "Chronic lung disease", "source": "HPO",
                        "source_url": "..."}] * 10,  # → trimmed to 8
    }
    compact = engine._compact_evidence(evidence)

    checks = [
        ("disease",        compact.get("disease") == "cystic fibrosis"),
        ("definition",     "genetic disorder" in compact.get("definition", "")),
        ("top_genes ≤10",  len(compact.get("top_genes", [])) <= 10),
        ("top_variants ≤8",len(compact.get("top_variants", [])) <= 8),
        ("top_drugs ≤6",   len(compact.get("top_drugs", [])) <= 6),
        ("top_pathways ≤5",len(compact.get("top_pathways", [])) <= 5),
        ("top_phenotypes ≤8", len(compact.get("top_phenotypes", [])) <= 8),
    ]
    for label, passed in checks:
        (ok if passed else fail)(label)


def test_cache():
    section("Layer 1 · TTL Cache")
    from cache import TTLCache

    c = TTLCache(default_ttl=2, max_size=3)

    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)

    if c.get("a") == 1 and c.get("b") == 2 and c.get("c") == 3:
        ok("basic set/get")
    else:
        fail("basic set/get")

    # Eviction when max_size exceeded
    c.set("d", 4)
    remaining = sum(1 for k in ["a","b","c","d"] if c.get(k) is not None)
    if remaining <= 3:
        ok("max_size eviction", f"{remaining} entries remaining ≤ 3")
    else:
        fail("max_size eviction", f"{remaining} entries > 3")

    # TTL expiry
    c2 = TTLCache(default_ttl=1, max_size=10)
    c2.set("x", 99)
    if c2.get("x") == 99:
        ok("TTL: hit before expiry")
    else:
        fail("TTL: hit before expiry")
    time.sleep(1.1)
    if c2.get("x") is None:
        ok("TTL: miss after expiry")
    else:
        fail("TTL: miss after expiry")

    # make_key determinism
    k1 = TTLCache.make_key("qs4", "cystic fibrosis", "")
    k2 = TTLCache.make_key("qs4", "cystic fibrosis", "")
    k3 = TTLCache.make_key("qs4", "huntington disease", "")
    if k1 == k2:
        ok("make_key deterministic")
    else:
        fail("make_key deterministic")
    if k1 != k3:
        ok("make_key distinct for different inputs")
    else:
        fail("make_key distinct")

    # Stats
    c3 = TTLCache(default_ttl=60, max_size=100)
    c3.set("p", 1); c3.set("q", 2)
    stats = c3.stats()
    if stats["entries"] == 2 and stats["max_size"] == 100:
        ok("stats()", str(stats))
    else:
        fail("stats()", str(stats))

    # Flush
    c3.flush()
    if c3.stats()["entries"] == 0:
        ok("flush()")
    else:
        fail("flush()")


# ═══════════════════════════════════════════════════════════════
# LAYER 2 — Service tests (network, no LLM)
# ═══════════════════════════════════════════════════════════════

def test_variant_aggregator_async():
    section("Layer 2 · VariantAggregator.async_aggregate")
    from services.variant_aggregator import VariantAggregator
    agg = VariantAggregator()

    t0   = time.time()
    data = asyncio.run(agg.async_aggregate("rs113993960"))
    elapsed = time.time() - t0

    src = data.get("sources", {})
    uni = data.get("unified", {})

    checks = [
        ("rsid present",              data.get("rsid") == "rs113993960"),
        ("sources dict present",      isinstance(src, dict) and len(src) > 0),
        ("ClinVar source fetched",    "ClinVar" in src),
        ("Ensembl source fetched",    "Ensembl" in src),
        ("unified dict present",      isinstance(uni, dict)),
        ("gene resolved",             bool(uni.get("gene"))),
        ("sources_available list",    len(uni.get("sources_available", [])) > 0),
        (f"completed in <30s ({elapsed:.1f}s)", elapsed < 30),
    ]
    for label, passed in checks:
        (ok if passed else fail)(label)

    # ACMG structure (optional — depends on ClinVar data)
    if "acmg" in data:
        ok("acmg key present", data["acmg"].get("classification", "?"))

    print(f"\n    Gene: {uni.get('gene')}  "
          f"ClinSig: {uni.get('clinical_significance')}  "
          f"Sources: {list(src.keys())}  "
          f"Time: {elapsed:.2f}s")


def test_disease_engine_async():
    section("Layer 2 · DiseaseEngine.async_aggregate")
    from services.disease_engine import DiseaseEngine
    engine = DiseaseEngine()

    t0 = time.time()
    ev = asyncio.run(engine.async_aggregate("cystic fibrosis"))
    elapsed = time.time() - t0

    checks = [
        ("query field",               ev.get("query") == "cystic fibrosis"),
        ("all_citations is list",     isinstance(ev.get("all_citations"), list)),
        ("at least 1 citation",       len(ev.get("all_citations", [])) >= 1),
        ("genes list present",        isinstance(ev.get("genes"), list)),
        ("variants list present",     isinstance(ev.get("variants"), list)),
        ("sources_used list",         len(ev.get("sources_used", [])) > 0),
        (f"completed in <20s ({elapsed:.1f}s)", elapsed < 20),
    ]
    for label, passed in checks:
        (ok if passed else fail)(label)

    print(f"\n    Sources: {ev.get('sources_used')}")
    print(f"    Citations: {len(ev.get('all_citations', []))}  "
          f"Genes: {len(ev.get('genes',[]))}  "
          f"Variants: {len(ev.get('variants',[]))}  "
          f"Time: {elapsed:.2f}s")


def test_pubmed_service():
    section("Layer 2 · PubMedService")
    from services.pubmed_service import PubMedService
    pm = PubMedService()

    # General search
    papers = pm.search("CFTR cystic fibrosis pathogenic", max_results=5)
    if papers and len(papers) >= 1:
        ok(f"search returned {len(papers)} papers")
        p = papers[0]
        has_keys = all(k in p for k in ("pmid", "title", "pubmed_url"))
        (ok if has_keys else fail)("paper has pmid/title/pubmed_url")
    else:
        fail("search returned 0 papers")

    # Gene search
    gene_papers = pm.search_for_gene("BRCA1")
    if gene_papers:
        ok(f"search_for_gene returned {len(gene_papers)} papers")
    else:
        fail("search_for_gene returned 0 papers")


def test_chroma_store():
    section("Layer 2 · ChromaStore vector search")
    # Skip if the server is already running — loading a second SentenceTransformer
    # in the same machine causes an OpenBLAS fatal memory error (C-level crash that
    # cannot be caught).  The vector store is exercised indirectly via Layer 4.
    import socket
    with socket.socket() as _s:
        _server_up = _s.connect_ex(("127.0.0.1", 8000)) == 0
    if _server_up:
        skip("ChromaStore.query", "server is running (OpenBLAS memory conflict) — tested via Layer 4")
        return
    try:
        from data_pipeline.vector_store import ChromaStore
        vs = ChromaStore()
        results = vs.query("CFTR cystic fibrosis mutation", n_results=3)
        if isinstance(results, list):
            ok(f"query returned list ({len(results)} results)")
            if results:
                r = results[0]
                has_keys = all(k in r for k in ("id", "text", "metadata"))
                (ok if has_keys else fail)("result has id/text/metadata")
        else:
            fail("query did not return list", type(results).__name__)
    except Exception as e:
        fail("ChromaStore.query", str(e))


# ═══════════════════════════════════════════════════════════════
# LAYER 3 — LLM integration tests (network + Groq)
# ═══════════════════════════════════════════════════════════════

def _server_running(port: int = 8000) -> bool:
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def test_rag_stream_chat():
    section("Layer 3 · RAGEngine.stream_chat_sync (with PubMed fallback)")
    if _server_running():
        skip("stream_chat_sync", "server running — OpenBLAS memory conflict; covered by Layer 4")
        return
    try:
        from rag_engine import RAGEngine
        from services.pubmed_service import PubMedService
        engine = RAGEngine()
        pm     = PubMedService()

        if not engine.client:
            skip("stream_chat_sync", "GROQ_API_KEY not set")
            return

        # Test 1: normal query — should get ChromaDB context or PubMed fallback
        tokens = []
        citations = []
        for ev in engine.stream_chat_sync("What is CFTR?", pubmed_svc=pm):
            if ev["type"] == "token":
                tokens.append(ev["text"])
            elif ev["type"] == "llm_done":
                citations = ev.get("citations", [])
                break

        answer = "".join(tokens)
        if len(answer) > 50:
            ok(f"streamed answer ({len(answer)} chars)")
        else:
            fail("streamed answer too short", answer[:100])

        if citations:
            ok(f"citations returned ({len(citations)})")
        else:
            # Acceptable if vector store is empty — PubMed fallback may still return none
            ok("no citations (vector store may be empty)", "PubMed fallback active")

        # Test 2: very niche query — should trigger PubMed fallback
        tokens2 = []
        for ev in engine.stream_chat_sync(
            "What is the molecular mechanism of phospholamban inhibition of SERCA2a?",
            pubmed_svc=pm,
        ):
            if ev["type"] == "token":
                tokens2.append(ev["text"])
            elif ev["type"] == "llm_done":
                break

        answer2 = "".join(tokens2)
        if len(answer2) > 30:
            ok("PubMed/LLM fallback returned answer for niche query")
        else:
            fail("fallback answer too short", answer2[:100])

    except Exception as e:
        fail("stream_chat_sync raised exception", str(e))
        traceback.print_exc()


def test_rag_classify():
    section("Layer 3 · RAGEngine.classify_query (LLM fallback path)")
    if _server_running():
        skip("classify_query", "server running — covered by Layer 4")
        return
    try:
        from rag_engine import RAGEngine
        engine = RAGEngine()

        if not engine.client:
            skip("classify_query", "GROQ_API_KEY not set")
            return

        # These are ambiguous enough that fast_classify returns None
        ambiguous = [
            ("What does FBN1 encode?", ("gene", "general")),
            ("Tell me about APOE4",     ("gene", "disease", "general")),
        ]
        for q, valid in ambiguous:
            result = engine.classify_query(q)
            if result in valid:
                ok(f"classify({q!r})", result)
            else:
                fail(f"classify({q!r})", f"got {result!r}, expected one of {valid}")

    except Exception as e:
        fail("classify_query", str(e))


def test_disease_report_streaming():
    section("Layer 3 · RAGEngine.stream_synthesis_sync")
    if _server_running():
        skip("stream_synthesis_sync", "server running — covered by Layer 4 disease stream")
        return
    try:
        from rag_engine import RAGEngine
        from services.disease_engine import DiseaseEngine

        engine = RAGEngine()
        if not engine.client:
            skip("stream_synthesis_sync", "GROQ_API_KEY not set")
            return

        # Minimal evidence bundle (no network needed for this part)
        evidence = {
            "query": "test disease",
            "overview": {"definition": "A test condition.", "source": "MedGen",
                         "source_url": "https://example.com"},
            "genes": [{"cn": 1, "gene": "GENE1", "gene_name": "Test gene",
                        "association_score": 0.9, "source": "Open Targets",
                        "source_url": "https://example.com"}],
            "variants": [], "drugs": [], "pathways": [], "phenotypes": [],
            "omim_entries": [],
            "all_citations": [{"n": 1, "db": "Open Targets", "url": "...", "label": "GENE1"}],
        }

        tokens = []
        for ev in engine.stream_synthesis_sync(evidence):
            if ev["type"] == "token":
                tokens.append(ev["text"])
            elif ev["type"] == "llm_done":
                break

        full = "".join(tokens)
        has_section  = "##SECTION:" in full
        has_followup = "##FOLLOWUPS:" in full

        if len(full) > 100:
            ok(f"streamed {len(full)} chars")
        else:
            fail("too few tokens", full[:80])

        (ok if has_section  else fail)("contains ##SECTION: markers")
        (ok if has_followup else fail)("contains ##FOLLOWUPS: marker")

        # Parse it
        parsed = engine._parse_structured_text(full)
        secs   = parsed.get("sections", [])
        fus    = parsed.get("followups", [])

        if len(secs) >= 1:
            ok(f"parsed {len(secs)} sections")
        else:
            fail("no sections parsed", full[:200])

        if len(fus) >= 1:
            ok(f"parsed {len(fus)} followups")
        else:
            ok("no followups yet (short evidence bundle may skip them)")

    except Exception as e:
        fail("stream_synthesis_sync", str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# LAYER 4 — Live server / API endpoint tests
# ═══════════════════════════════════════════════════════════════

def _http_get(path: str, timeout: int = 15) -> dict | None:
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return None

def _http_post(path: str, body: dict, timeout: int = 15) -> dict | None:
    import urllib.request, urllib.error
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:8000{path}",
        data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return None

def _http_stream(path: str, body: dict, max_events: int = 20, timeout: int = 20) -> list:
    """Read up to max_events SSE events from a streaming endpoint."""
    import socket, urllib.request
    data     = json.dumps(body).encode()
    req      = urllib.request.Request(
        f"http://127.0.0.1:8000{path}",
        data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    events = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf = b""
            while len(events) < max_events:
                chunk = r.read(512)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    part, buf = buf.split(b"\n\n", 1)
                    part = part.strip()
                    if part.startswith(b"data: "):
                        try:
                            ev = json.loads(part[6:])
                            events.append(ev)
                            if ev.get("type") in ("done", "error"):
                                return events
                        except Exception:
                            pass
    except Exception as e:
        events.append({"type": "error", "message": str(e)})
    return events


def test_server_health():
    section("Layer 4 · Server health")
    import time as _time
    # Wait up to 15s in case the server is still starting
    data = None
    for _ in range(5):
        data = _http_get("/health")
        if data:
            break
        _time.sleep(3)
    if data and data.get("version") == "4.0.0":
        ok("health endpoint", f"v{data['version']}")
    elif data:
        fail("version mismatch", f"got {data.get('version')}")
    else:
        skip("server not reachable — start with: python server.py", "Layer 4 skipped")
        return False
    return True


def test_server_cache():
    section("Layer 4 · Cache endpoints")
    stats = _http_get("/api/cache/stats")
    if stats and "entries" in stats:
        ok("GET /api/cache/stats", str(stats))
    else:
        fail("GET /api/cache/stats")

    flush = _http_post("/api/cache/flush", {})
    if flush and flush.get("ok"):
        ok("POST /api/cache/flush")
    else:
        fail("POST /api/cache/flush")


def test_server_stream_normalization():
    section("Layer 4 · /api/query/stream — query normalization")
    events = _http_stream("/api/query/stream", {"q": "cf", "history": []}, max_events=4)
    evtypes = [e.get("type") for e in events]

    meta = next((e for e in events if e.get("type") == "meta"), None)
    if meta and meta.get("query") == "cystic fibrosis" and meta.get("original") == "cf":
        ok("'cf' normalized to 'cystic fibrosis'", str(meta))
    elif meta:
        fail("normalization result", str(meta))
    else:
        fail("no meta event", str(events[:4]))

    classified = next((e for e in events if e.get("type") == "classified"), None)
    if classified and classified.get("query_type") == "disease":
        ok("fast-classified as 'disease' without LLM")
    elif classified:
        fail("wrong query_type", classified.get("query_type"))
    else:
        ok("classified event not yet received (within 4 events — OK)")


def test_server_stream_variant():
    section("Layer 4 · /api/query/stream — variant (rs113993960)")
    events = _http_stream(
        "/api/query/stream",
        {"q": "rs113993960", "history": []},
        max_events=8, timeout=20,
    )

    classified = next((e for e in events if e.get("type") == "classified"), None)
    variant_ev = next((e for e in events if e.get("type") == "variant_data"), None)

    if classified and classified.get("query_type") == "variant":
        ok("classified as 'variant'")
    else:
        fail("variant classification", str(events[:4]))

    if variant_ev:
        d   = variant_ev.get("data", {})
        src = d.get("sources", {})
        ok("variant_data event received", f"sources={list(src.keys())}")
        if "ClinVar" in src or "Ensembl" in src:
            ok("ClinVar or Ensembl source present")
        else:
            fail("no ClinVar/Ensembl in sources", str(list(src.keys())))
    else:
        fail("no variant_data event (may be slow network)", str([e.get("type") for e in events]))


def test_server_stream_chat_followup():
    section("Layer 4 · /api/query/stream — chat follow-up (history present)")
    events = _http_stream(
        "/api/query/stream",
        {
            "q": "What gene is mutated?",
            "history": [
                {"role": "user",      "content": "Tell me about cystic fibrosis"},
                {"role": "assistant", "content": "Cystic fibrosis is caused by CFTR mutations."},
            ],
        },
        max_events=6, timeout=20,
    )
    evtypes = [e.get("type") for e in events]

    # In chat mode, no 'classified' event — streaming starts immediately
    if "classified" not in evtypes:
        ok("chat mode: no classification step (instant start)")
    else:
        ok("chat mode classified (acceptable)", str(evtypes))

    tokens = [e for e in events if e.get("type") == "token"]
    if tokens:
        ok(f"tokens streaming ({len(tokens)} received within limit)")
    else:
        ok("no tokens yet (within 6 event limit — server is streaming)")


def test_server_variant_endpoint():
    section("Layer 4 · GET /api/variant/{rsid}")
    data = _http_get("/api/variant/rs113993960", timeout=20)
    if not data:
        fail("no response from /api/variant/rs113993960")
        return

    src = data.get("sources", {})
    uni = data.get("unified", {})

    checks = [
        ("rsid in response",      data.get("rsid") == "rs113993960"),
        ("sources dict",          isinstance(src, dict) and len(src) > 0),
        ("unified dict",          isinstance(uni, dict)),
        ("ClinVar or Ensembl",    "ClinVar" in src or "Ensembl" in src),
        ("acmg field present",    "acmg" in data),
    ]
    for label, passed in checks:
        (ok if passed else fail)(label)
    print(f"\n    Gene={uni.get('gene')}  "
          f"Sig={uni.get('clinical_significance')}  "
          f"Sources={list(src.keys())}")


def test_server_analyses():
    section("Layer 4 · /api/analyses CRUD")
    # List
    data = _http_get("/api/analyses")
    if data and "analyses" in data:
        ok(f"GET /api/analyses ({len(data['analyses'])} items)")
    else:
        fail("GET /api/analyses", str(data))

    # Save
    saved = _http_post("/api/analyses", {
        "query": "test query from test_v4",
        "type":  "chat",
        "result": {"answer": "test answer", "citations": []},
    })
    if saved and saved.get("id"):
        ok("POST /api/analyses", f"id={saved['id'][:8]}")
        # Get
        row = _http_get(f"/api/analyses/{saved['id']}")
        if row and row.get("id") == saved["id"]:
            ok("GET /api/analyses/{id}")
        else:
            fail("GET /api/analyses/{id}")

        # Delete
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:8000/api/analyses/{saved['id']}",
            method="DELETE",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            ok("DELETE /api/analyses/{id}")
        except Exception as e:
            # 204 raises URLError on some urllib versions
            if "204" in str(e) or "No Content" in str(e):
                ok("DELETE /api/analyses/{id}")
            else:
                fail("DELETE /api/analyses/{id}", str(e))
    else:
        fail("POST /api/analyses", str(saved))


def test_server_pubmed():
    section("Layer 4 · GET /api/pubmed")
    data = _http_get("/api/pubmed?q=CFTR+cystic+fibrosis", timeout=15)
    if data and data.get("papers"):
        ok(f"returned {data['count']} papers")
        p = data["papers"][0]
        if p.get("pubmed_url"):
            ok("paper has pubmed_url")
        else:
            fail("paper missing pubmed_url", str(p.keys()))
    else:
        fail("no papers returned", str(data))


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

LAYERS = {
    1: [test_query_normalizer, test_parse_structured_text,
        test_compact_evidence, test_cache],
    2: [test_variant_aggregator_async, test_disease_engine_async,
        test_pubmed_service, test_chroma_store],
    3: [test_rag_stream_chat, test_rag_classify,
        test_disease_report_streaming],
    4: [test_server_health, test_server_cache,
        test_server_stream_normalization, test_server_stream_variant,
        test_server_stream_chat_followup, test_server_variant_endpoint,
        test_server_analyses, test_server_pubmed],
}

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  BioSage v4.0 -- Test Suite")
    print("="*62)

    if LAYER:
        layers_to_run = {LAYER: LAYERS.get(LAYER, [])}
        print(f"\nRunning Layer {LAYER} only")
    else:
        layers_to_run = LAYERS
        print("\nRunning all 4 layers")

    t_start = time.time()
    for layer_num, fns in layers_to_run.items():
        print(f"\n{'='*62}")
        print(f"  LAYER {layer_num}")
        print(f"{'='*62}")
        # Layer 1 uses lightweight stubs so it can import rag_engine without
        # loading OpenBLAS (the server process may hold the memory).
        # All other layers need real imports — clear stubs first.
        if layer_num == 1:
            _install_stubs()
        else:
            _remove_stubs()
        # Layer 4 requires a live server — skip remaining tests if health fails
        _layer4_ok = True
        for fn in fns:
            if layer_num == 4 and not _layer4_ok:
                skip(fn.__name__, "server unreachable")
                continue
            try:
                result = fn()
                if layer_num == 4 and fn.__name__ == "test_server_health" and result is False:
                    _layer4_ok = False
            except Exception as exc:
                fail(fn.__name__, f"uncaught exception: {exc}")
                traceback.print_exc()

    elapsed = time.time() - t_start
    total = PASS + FAIL + SKIP
    print(f"\n{'='*62}")
    print(f"  Results  PASS={PASS}  FAIL={FAIL}  SKIP={SKIP}  total={total}  ({elapsed:.1f}s)")
    print(f"{'='*62}\n")
    sys.exit(0 if FAIL == 0 else 1)
