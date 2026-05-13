# BIOSAGE 

### AI-Powered Genomic Intelligence — B2C SaaS for Disease & Variant Interpretation

> A scalable, cloud-ready **B2C SaaS platform** that leverages **LLM-driven Retrieval-Augmented Generation (RAG)** and real-time aggregation from open-source genomic databases to deliver instant, clinically-grounded disease and variant insights — to researchers, clinicians, and curious minds alike.

---

##  What It Does

BIOSAGE is a **text-based AI assistant** that lets any user ask natural-language questions about diseases, genetic variants, and clinical significance — and get structured, citation-backed answers in seconds.

Under the hood, it combines:

- **Groq-accelerated LLM inference** (Llama 3.3 70B) for sub-second response times
- **RAG pipeline** over a curated knowledge base built from **ClinVar, OMIM, COSMIC, DisGeNET, Ensembl, and OMIA**
- **Multi-source variant aggregation** across ClinVar, Ensembl, dbSNP, and UniProt
- **ACMG/AMP 2015 classification engine** evaluating all 28 evidence criteria
- **VCF genome upload & risk reporting** for personal genomics workflows

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│               Next.js Frontend (SPA)                     │
│   Sidebar Nav · Variant Query · Genome Upload · Chat     │
└───────────────────────┬──────────────────────────────────┘
                        │  REST API
┌───────────────────────▼──────────────────────────────────┐
│              FastAPI Backend (Python)                     │
│                                                          │
│   RAG Engine (Groq)  ·  Variant Aggregator  ·  ACMG      │
│   PubMed Service     ·  VCF Parser          ·  SQLite     │
└───────────────────────┬──────────────────────────────────┘
                        │  HTTPS
┌───────────────────────▼──────────────────────────────────┐
│              External Databases & APIs                    │
│   ClinVar · Ensembl · dbSNP · UniProt · OMIM · PubMed    │
└──────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **AI Chat (RAG)** | Ask anything about diseases — answers are grounded in curated genomic data, not hallucinated |
| **Variant Lookup** | Enter any rsID and get a unified report aggregated from 4+ databases |
| **ACMG Classification** | Automated pathogenicity classification using all 28 ACMG/AMP criteria |
| **PubMed Mining** | Real-time literature search by gene, variant, or disease |
| **VCF Upload** | Upload a VCF file and receive a prioritized risk report |
| **Workspaces** | Save, organize, and export analyses across sessions |

---

## Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js, TypeScript, Tailwind CSS |
| **Backend** | FastAPI (Python), Uvicorn |
| **LLM Inference** | Groq API (Llama 3.3 70B Versatile) |
| **Vector Store** | ChromaDB with all-MiniLM-L6-v2 embeddings |
| **Data Sources** | ClinVar, Ensembl, dbSNP, UniProt, OMIM, COSMIC, DisGeNET, OMIA |
| **Database** | SQLite (PostgreSQL-ready schema) |
| **Literature** | NCBI E-utilities (PubMed) |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/govindharajan-m/biosage.git
cd biosage

# 2. Set up environment variables
cp backend/.env.example backend/.env
# Add your GROQ_API_KEY, NCBI_API_KEY, etc.

# 3. Launch (Windows)
# Double-click: START HERE.bat
# Or manually:
cd backend
pip install -r requirements.txt
python server.py

# 4. Open http://127.0.0.1:8000
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Groq API key for LLM inference |
| `NCBI_API_KEY` | Recommended | Increases NCBI rate limit (3 → 10 req/s) |
| `NCBI_EMAIL` | Recommended | Required by NCBI for API access |
| `OMIM_API_KEY` | For OMIM data | OMIM API access |

---

##  Scalability & Roadmap

DiseaseLLM is architected with **horizontal scalability** in mind. The modular, service-oriented backend is designed for a clear migration path from MVP to production-grade SaaS:

| Priority | Milestone | Impact |
|----------|-----------|--------|
| 🔴 High | **PostgreSQL + Redis caching** | Production-grade persistence & sub-ms variant lookups |
| 🔴 High | **Async job processing** (Celery + Redis) | Handle large VCF files without blocking the API |
| 🟡 Medium | **User authentication & multi-tenancy** | JWT-based auth with workspace isolation — true B2C readiness |
| 🟡 Medium | **OMIM deep integration** | Full phenotype-gene-disease mapping |
| 🟢 Low | **gnomAD population data** | Local MAF lookups for population-level insights |
| 🟢 Low | **Team workspaces** | Multi-user collaboration and shared analyses |

---

## API Reference

### Core Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | RAG-powered AI chat |
| `GET` | `/api/variant/{rsid}` | Full variant report (all DBs + ACMG) |
| `GET` | `/api/acmg/{rsid}` | Standalone ACMG classification |
| `GET` | `/api/pubmed?gene=&rsid=&disease=` | Literature search |
| `GET` | `/api/gene/{symbol}` | Gene overview + literature |
| `POST` | `/api/vcf/upload` | VCF upload → risk report |

### Workspace & Persistence

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/workspaces` | List workspaces |
| `POST` | `/api/workspaces` | Create workspace |
| `DELETE` | `/api/workspaces/{id}` | Delete workspace |
| `GET` | `/api/analyses?workspace_id=` | List saved analyses |
| `POST` | `/api/analyses` | Save analysis |
| `GET` | `/api/analyses/{id}/export` | Export as JSON |

📖 Interactive docs: `http://127.0.0.1:8000/api/docs`

---

##  How It Works

### RAG Pipeline
1. **Ingest** — Fetchers pull data from 6+ open-source genomic databases
2. **Normalize & Chunk** — Records are merged, deduplicated, and chunked for embedding
3. **Embed** — Chunks are embedded via `all-MiniLM-L6-v2` and stored in ChromaDB
4. **Query** — User questions trigger semantic search over the vector store
5. **Generate** — Retrieved context is fed to Groq's Llama 3.3 70B for grounded, cited answers

### ACMG Classification
Implements the full **ACMG/AMP 2015 variant interpretation guidelines** — evaluating 28 evidence criteria (PVS1, PS1–PS4, PM1–PM6, PP1–PP5, BA1, BS1–BS4, BP1–BP7) and applying combination rules to classify variants as Pathogenic, Likely Pathogenic, VUS, Likely Benign, or Benign.

---

<p align="center">
  <b>DiseaseLLM</b> — Democratizing genomic intelligence, one query at a time.<br/>
  Built with FastAPI · ChromaDB · Groq · NCBI APIs · UniProt
</p>
