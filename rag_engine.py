"""
RAGEngine — hybrid retrieval + Groq LLM synthesis.

Operating modes
───────────────
  answer(query, history)              — conversational chat (non-streaming)
  stream_chat_sync(query, …)          — streaming chat with PubMed fallback
  synthesize_disease_report(evidence) — structured JSON report
  stream_synthesis_sync(evidence)     — streaming structured report
  classify_query(query, history)      — intent routing (LLM fallback path)

Retrieval fallback chain
────────────────────────
  1. ChromaDB vector search  (primary)
  2. PubMed full-text search (if ChromaDB returns < MIN_RESULTS results)
  3. LLM biomedical knowledge (if both are empty — with explicit disclaimer)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from data_pipeline.vector_store import ChromaStore

logger = logging.getLogger(__name__)

# Minimum ChromaDB hits before we trigger PubMed fallback
_MIN_CHROMA_RESULTS = 2

# ── Prompts ───────────────────────────────────────────────────────────────────

_CHAT_SYSTEM = """\
You are BioSage, a conversational biomedical research assistant with access to \
curated evidence from ClinVar, Open Targets, OMIM, ChEMBL, Reactome, PubMed, and related databases.

Answer the question using the retrieved evidence below. Every factual claim must \
include a citation marker [N] where N matches the source number.

If the evidence does not support the question, state that clearly rather than speculating. \
Never fabricate gene names, variant identifiers, drug names, or clinical claims.

Write in concise, precise scientific prose. No emojis. No bullet symbols.

RETRIEVED EVIDENCE (numbered sources):
{context}"""

_CHAT_SYSTEM_NO_DB = """\
You are BioSage, a conversational biomedical research assistant.

NOTE: No records were found in the curated databases for this query. \
You are answering from established biomedical literature knowledge. \
Prefix your response with: "Based on established biomedical knowledge \
(no specific database records found):" — then provide a thorough, \
accurate answer. Cite recognised sources such as textbooks or guideline \
bodies where appropriate, but do not fabricate specific study references.

Write in concise, precise scientific prose. No emojis. No bullet symbols."""

_DISEASE_JSON_SYSTEM = """\
You are BioSage generating a structured biomedical research report.
Output ONLY a valid JSON object — no text before or after the JSON.

Required schema:
{{
  "sections": [
    {{"title": "Disease Overview",     "content": "2-3 sentences with [N] citations."}},
    {{"title": "Genetic Basis",        "content": "Key genes, functions, association scores with [N] citations."}},
    {{"title": "Pathogenic Variants",  "content": "Most significant mutations and clinical significance with [N] citations."}},
    {{"title": "Biological Pathways",  "content": "Disrupted pathways with [N] citations."}},
    {{"title": "Clinical Features",    "content": "Major symptoms and phenotypes with [N] citations."}},
    {{"title": "Treatment Approaches", "content": "Known drugs, clinical phases, targets with [N] citations."}}
  ],
  "followups": [
    "Specific follow-up question 1",
    "Specific follow-up question 2",
    "Specific follow-up question 3",
    "Specific follow-up question 4"
  ]
}}

RULES:
- Every content sentence must include at least one [N] citation where N is from the evidence numbers.
- If no evidence available for a section: set content to "Insufficient evidence available for this section."
- followups must be specific, natural research questions a scientist would ask about this disease.
- No emojis. No # markdown headers in content.

EVIDENCE:
{evidence}"""

_CLASSIFY_SYSTEM = """\
Classify this biomedical query. Reply with ONLY one of these exact words:
disease, variant, gene, drug, general

A "disease" query asks about conditions, disorders, or syndromes.
A "variant" query asks about a specific genetic variant or rsID.
A "gene" query asks about a specific gene's function or associations.
A "drug" query asks about a medication or therapeutic.
"general" for anything else.

Prior conversation context: {history_summary}
Query: {query}"""

_FOLLOWUP_SYSTEM = """\
Generate exactly 4 specific follow-up research questions for this biomedical query.
Output only the 4 questions, one per line, no numbering, no leading symbols.
Make each question specific — use actual gene names, disease names, or drug names from the answer.

Query: {query}
Answer summary (first 400 chars): {summary}"""


class RAGEngine:
    """
    Retrieval-augmented generation using ChromaDB for vector search
    and Groq (llama-3.3-70b-versatile) for reasoning and synthesis.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — LLM functions disabled")
        self.vector_store = ChromaStore()
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    # ── Low-level Groq wrappers ───────────────────────────────────────────────

    def _chat(self, system: str, user: str, max_tokens: int = 1200,
              temperature: float = 0.1, json_mode: bool = False) -> str:
        if not self.client:
            return "LLM unavailable — GROQ_API_KEY not configured."
        kwargs = dict(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return "" if json_mode else f"LLM error: {e}"

    def _chat_messages(self, messages: list, max_tokens: int = 1200,
                       temperature: float = 0.1) -> str:
        """Send a full messages array (supports conversation history)."""
        if not self.client:
            return "LLM unavailable — GROQ_API_KEY not configured."
        try:
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq (messages) error: {e}")
            return f"LLM error: {e}"

    # ── Context / citation helpers ────────────────────────────────────────────

    def _format_chroma_context(self, results: list) -> str:
        parts = []
        for i, res in enumerate(results, 1):
            meta   = res.get("metadata", {})
            source = meta.get("source_db", "Unknown")
            rec_id = res.get("id", "?")
            text   = res.get("text", "")
            parts.append(f"[{i}] {source} ({rec_id})\n{text}")
        return "\n\n".join(parts)

    def _format_pubmed_context(self, papers: list, offset: int = 0) -> str:
        """Format PubMed papers as numbered evidence context."""
        parts = []
        for i, p in enumerate(papers, offset + 1):
            title    = p.get("title", "Untitled")
            authors  = p.get("authors", "")
            year     = p.get("year", "")
            abstract = p.get("abstract", "")[:400]
            parts.append(f"[{i}] PubMed — {title} ({authors}, {year})\n{abstract}")
        return "\n\n".join(parts)

    def _history_to_messages(self, history: list) -> list:
        """Convert [{role, content}] history to Groq messages (capped at 10)."""
        msgs = []
        for msg in (history or [])[-10:]:
            role    = msg.get("role", "user")
            content = msg.get("content") or ""
            if role in ("user", "assistant") and content:
                if role == "assistant" and len(content) > 800:
                    content = content[:800] + "..."
                msgs.append({"role": role, "content": str(content)})
        return msgs

    def _history_summary(self, history: list) -> str:
        if not history:
            return "None"
        recent = [m.get("content", "")[:80] for m in history[-4:] if m.get("content")]
        return " | ".join(recent)

    def _make_citations(self, results: list, pubmed_papers: list = None,
                        offset: int = 0) -> list:
        citations = [
            {
                "n":     i + 1,
                "db":    res.get("metadata", {}).get("source_db", ""),
                "label": res.get("id", ""),
                "url":   res.get("metadata", {}).get("url", ""),
            }
            for i, res in enumerate(results)
        ]
        if pubmed_papers:
            base = len(results) + offset
            for j, p in enumerate(pubmed_papers, base + 1):
                citations.append({
                    "n":     j,
                    "db":    "PubMed",
                    "label": p.get("title", "")[:80],
                    "url":   p.get("pubmed_url", ""),
                })
        return citations

    # ── Mode 1: Conversational chat (non-streaming) ───────────────────────────

    def answer(self, query: str, history: Optional[List[Dict]] = None,
               n_results: int = 6, pubmed_svc=None) -> Dict[str, Any]:
        """
        Retrieve from ChromaDB, fall back to PubMed if needed, then synthesize.
        Returns: {answer, sections, followups, citations, sources}
        """
        logger.info(f"RAG.answer: {query[:80]}")

        results = self.vector_store.query(query, n_results=n_results)

        pubmed_papers = []
        if len(results) < _MIN_CHROMA_RESULTS and pubmed_svc:
            try:
                pubmed_papers = pubmed_svc.search(query, max_results=5)
            except Exception as e:
                logger.warning("PubMed fallback error: %s", e)

        if results:
            context = self._format_chroma_context(results)
            if pubmed_papers:
                context += "\n\nADDITIONAL LITERATURE:\n" + \
                    self._format_pubmed_context(pubmed_papers, offset=len(results))
            system = _CHAT_SYSTEM.format(context=context)
        elif pubmed_papers:
            context = self._format_pubmed_context(pubmed_papers)
            system  = _CHAT_SYSTEM.format(context=context)
        else:
            system = _CHAT_SYSTEM_NO_DB

        messages = [{"role": "system", "content": system}]
        messages.extend(self._history_to_messages(history))
        messages.append({"role": "user", "content": query})

        answer_text = self._chat_messages(messages, max_tokens=1000)
        followups   = self._generate_followups(query, answer_text)
        citations   = self._make_citations(results, pubmed_papers)

        if not results and not pubmed_papers:
            logger.info("RAG.answer: no DB hits for '%s' — using LLM knowledge", query)

        return {
            "answer":    answer_text,
            "sections":  [],
            "followups": followups,
            "citations": citations,
            "sources":   results,
        }

    # ── Mode 1b: Streaming chat ───────────────────────────────────────────────

    def stream_chat_sync(self, query: str, history: list = None,
                         n_results: int = 6, pubmed_svc=None):
        """
        Sync generator — streams a conversational RAG answer token by token.

        Retrieval fallback chain:
          1. ChromaDB vector search
          2. PubMed (if ChromaDB returns < _MIN_CHROMA_RESULTS results)
          3. LLM biomedical knowledge (with disclaimer, if both empty)

        Yields: {"type": "token", "text": str}
        Final:  {"type": "llm_done", "citations": list}
        """
        if not self.client:
            yield {"type": "token", "text": "LLM unavailable — GROQ_API_KEY not configured."}
            yield {"type": "llm_done", "citations": []}
            return

        # ── Retrieval ─────────────────────────────────────────────────────────
        results = self.vector_store.query(query, n_results=n_results)

        pubmed_papers: list = []
        if len(results) < _MIN_CHROMA_RESULTS and pubmed_svc:
            try:
                pubmed_papers = pubmed_svc.search(query, max_results=5)
                logger.info("PubMed fallback for '%s': %d papers", query, len(pubmed_papers))
            except Exception as e:
                logger.warning("PubMed fallback error: %s", e)

        # ── Build context ─────────────────────────────────────────────────────
        if results:
            context = self._format_chroma_context(results)
            if pubmed_papers:
                context += "\n\nADDITIONAL LITERATURE:\n" + \
                    self._format_pubmed_context(pubmed_papers, offset=len(results))
            system = _CHAT_SYSTEM.format(context=context)
        elif pubmed_papers:
            context = self._format_pubmed_context(pubmed_papers)
            system  = _CHAT_SYSTEM.format(context=context)
        else:
            logger.info("stream_chat_sync: no DB/PubMed hits — LLM-only mode")
            system = _CHAT_SYSTEM_NO_DB

        citations = self._make_citations(results, pubmed_papers)

        messages = [{"role": "system", "content": system}]
        messages.extend(self._history_to_messages(history))
        messages.append({"role": "user", "content": query})

        # ── Stream LLM ────────────────────────────────────────────────────────
        try:
            stream = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield {"type": "token", "text": text}
        except Exception as e:
            logger.error("stream_chat_sync error: %s", e)
            yield {"type": "token", "text": f"[Error: {e}]"}

        yield {"type": "llm_done", "citations": citations}

    # ── Mode 2: Structured disease report (non-streaming) ────────────────────

    def synthesize_disease_report(self, evidence: dict) -> dict:
        """
        Convert DiseaseEngine evidence into a structured JSON report.
        Returns: {sections: [{title, content}], followups: [str]}
        """
        logger.info(f"RAG.synthesize: '{evidence.get('query', '')}'")

        compact = self._compact_evidence(evidence)
        system  = _DISEASE_JSON_SYSTEM.format(evidence=json.dumps(compact, ensure_ascii=False))
        user    = f"Generate the structured research report for: {evidence.get('query', '')}"

        raw = self._chat(system, user, max_tokens=1500, json_mode=True)
        if raw:
            try:
                parsed    = json.loads(raw)
                sections  = parsed.get("sections", [])
                followups = parsed.get("followups", [])[:4]
                if sections:
                    return {"sections": sections, "followups": followups}
            except json.JSONDecodeError:
                logger.warning("JSON mode parse failed — falling back to text parsing")

        # Text-mode fallback
        system_text = system.replace(
            "Output ONLY a valid JSON object — no text before or after the JSON.",
            "Output the report using 'SECTION: Title\\n[content]' format, "
            "followed by 'FOLLOW_UPS:\\n[one per line]'."
        )
        raw = self._chat(system_text, user, max_tokens=1500, json_mode=False)
        return self._parse_structured_text(raw)

    def _compact_evidence(self, evidence: dict) -> dict:
        """Reduce evidence to a token-efficient summary for the LLM context."""
        compact: dict = {
            "disease":        evidence.get("query", ""),
            "definition":     "",
            "omim":           [],
            "top_genes":      [],
            "top_variants":   [],
            "top_drugs":      [],
            "top_pathways":   [],
            "top_phenotypes": [],
        }

        if evidence.get("overview"):
            ov   = evidence["overview"]
            defn = ov.get("definition") or ""
            compact["definition"] = str(defn)[:300]

        for e in evidence.get("omim_entries", [])[:2]:
            compact["omim"].append({"cn": e.get("cn"), "title": e.get("title", "")})

        for g in evidence.get("genes", [])[:10]:
            compact["top_genes"].append({
                "cn": g.get("cn"), "gene": g.get("gene", ""),
                "name": g.get("gene_name", ""), "score": g.get("association_score"),
            })

        for v in evidence.get("variants", [])[:8]:
            compact["top_variants"].append({
                "cn": v.get("cn"), "variant": v.get("variant_name", ""),
                "gene": v.get("gene", ""), "significance": v.get("clinical_significance", ""),
            })

        for d in evidence.get("drugs", [])[:6]:
            compact["top_drugs"].append({
                "cn": d.get("cn"), "name": d.get("drug_name", ""),
                "phase": d.get("max_phase"),
            })

        for p in evidence.get("pathways", [])[:5]:
            compact["top_pathways"].append({
                "cn": p.get("cn"), "pathway": p.get("pathway_name", ""),
            })

        for ph in evidence.get("phenotypes", [])[:8]:
            compact["top_phenotypes"].append({
                "cn": ph.get("cn"), "phenotype": ph.get("phenotype", ""),
            })

        return compact

    def _parse_structured_text(self, text: str) -> dict:
        """
        Parse both streaming (##SECTION: / ##FOLLOWUPS:) and
        fallback text-mode (SECTION: / FOLLOW_UPS:) delimited output.
        """
        sections: list  = []
        followups: list = []

        for marker in ("##FOLLOWUPS:", "FOLLOW_UPS:"):
            if marker in text:
                text, fu_text = text.split(marker, 1)
                followups = [
                    q.strip().lstrip("0123456789.-) ")
                    for q in fu_text.strip().split("\n")
                    if q.strip() and len(q.strip()) > 8
                ][:4]
                break

        if "##SECTION:" in text:
            parts = re.split(r"(?:^|\n)##SECTION:\s*", text)
        else:
            parts = re.split(r"(?:^|\n)SECTION:\s*", text)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines   = part.split("\n", 1)
            title   = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""
            if title and content:
                sections.append({"title": title, "content": content})

        if not sections and text.strip():
            sections = [{"title": "Research Report", "content": text.strip()}]

        return {"sections": sections, "followups": followups}

    # ── Mode 2b: Streaming disease report ─────────────────────────────────────

    def stream_synthesis_sync(self, evidence: dict, low_bandwidth: bool = False):
        """
        Sync generator — streams a structured disease report token by token.
        Uses ##SECTION: / ##FOLLOWUPS: markers for client-side parsing.

        low_bandwidth=True returns a compact 3-section report (~600 tokens).

        Yields: {"type": "token", "text": str}
        Final:  {"type": "llm_done"}
        """
        if not self.client:
            yield {"type": "token", "text": "LLM unavailable — GROQ_API_KEY not configured."}
            yield {"type": "llm_done"}
            return

        compact       = self._compact_evidence(evidence)
        evidence_json = json.dumps(compact, ensure_ascii=False)

        if low_bandwidth:
            system = f"""You are BioSage generating a concise biomedical summary.
Use EXACTLY this format:

##SECTION: Disease Overview
[2 sentences max with [N] citation markers]

##SECTION: Key Points
[3-4 bullet points covering genetics, symptoms, and treatment]

##SECTION: Treatment Approaches
[1-2 sentences on main treatments with [N] citations]

##FOLLOWUPS:
[2 specific follow-up questions, one per line]

RULES:
- Be extremely concise — low-bandwidth mode
- Every sentence must include at least one [N] citation
- No emojis, no asterisks, no markdown except ##SECTION / ##FOLLOWUPS markers
- NEVER open with "[Disease] is a condition..." — lead with the most striking fact

EVIDENCE:
{evidence_json[:1800]}"""
            max_tok = 600
        else:
            system = f"""You are BioSage generating a biomedical research report.
Use EXACTLY this format — no other text outside these markers:

##SECTION: Disease Overview
[2-3 sentences with [N] citation markers]

##SECTION: Genetic Basis
[Key genes, functions, association scores with [N] citations]

##SECTION: Pathogenic Variants
[Most significant mutations and clinical significance with [N] citations]

##SECTION: Biological Pathways
[Disrupted pathways with [N] citations]

##SECTION: Clinical Features
[Major symptoms and phenotypes with [N] citations]

##SECTION: Treatment Approaches
[Known drugs, clinical phases, molecular targets with [N] citations]

##FOLLOWUPS:
[Exactly 4 specific follow-up questions, one per line, no numbering]

RULES:
- Every factual sentence must include at least one [N] citation (N from evidence numbers)
- If a section has no evidence write: Insufficient evidence available.
- No emojis, no asterisks, no markdown except ##SECTION / ##FOLLOWUPS markers
- Be concise and scientifically precise
- NEVER open the Disease Overview with "[Disease] is a condition..." or "[Disease] is a disease..."
  Instead open with the most clinically or genetically striking fact — for example:
  "Caused by...", "Affecting roughly X people...", "First described by...",
  "Mutations in [gene] underlie...", "A hallmark of [disease] is...",
  "Driven by...", "The primary defect in [disease] lies in..."
- Write in an engaging, direct scientific voice — not a dry encyclopaedia entry

EVIDENCE:
{evidence_json[:3500]}"""
            max_tok = 1800

        try:
            stream = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content":
                        f"Generate the structured report for: {evidence.get('query', '')}"},
                ],
                temperature=0.1,
                max_tokens=max_tok,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield {"type": "token", "text": text}
        except Exception as e:
            logger.error("stream_synthesis_sync error: %s", e)
            yield {"type": "token", "text": f"[Report generation error: {e}]"}

        yield {"type": "llm_done"}

    # ── Immediate Response (fast parallel call) ───────────────────────────────

    def generate_immediate_response_sync(self, disease_name: str) -> str:
        """
        Fast, non-streaming LLM call executed in parallel with main retrieval.
        Returns bullet-point general first-response guidance.
        Deliberately avoids specific medications or dosages.
        """
        if not self.client:
            return ""

        system = """You are a biomedical safety assistant providing general first-response guidance.

OUTPUT FORMAT — respond with ONLY a bullet list, no headings, no preamble:
• [action or observation point]
• ...
(5–8 bullets maximum)

STRICT RULES:
- General supportive care only — NO specific drug names, NO dosages, NO prescriptions
- Appropriate for a non-medical audience (patient, carer, or first responder)
- Focus on: when to seek help, monitoring, comfort measures, documentation
- Last bullet MUST be the disclaimer exactly as shown:
  • ⚠ This information is for general educational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider."""

        user = (
            f"Provide immediate first-action guidance for someone newly diagnosed with or "
            f"caring for a person with: {disease_name}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=450,
                stream=False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("generate_immediate_response error: %s", e)
            return ""

    # ── Mode 3: Follow-up question generator ─────────────────────────────────

    def _generate_followups(self, query: str, answer: str) -> list:
        """Generate 4 specific follow-up research questions."""
        system = _FOLLOWUP_SYSTEM.format(
            query=query,
            summary=answer[:400] if answer else "",
        )
        try:
            raw       = self._chat(system, "", max_tokens=200, temperature=0.3)
            questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in raw.strip().split("\n")
                if q.strip() and len(q.strip()) > 8
            ]
            return questions[:4]
        except Exception:
            return []

    # ── Mode 4: Intent classifier (LLM fallback) ──────────────────────────────

    def classify_query(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """
        LLM-based intent classification (fallback after fast_classify returns None).
        Returns one of: disease | variant | gene | drug | general
        """
        system = _CLASSIFY_SYSTEM.format(
            history_summary=self._history_summary(history),
            query=query,
        )
        result = self._chat(system, query, max_tokens=10, temperature=0.0).strip().lower()
        for cat in ("disease", "variant", "gene", "drug", "general"):
            if cat in result:
                return cat
        return "general"
