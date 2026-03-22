"""
biosage_v2.py
──────────────────
Consolidated Phase 2 script for BioSage. 
Includes: Configuration, Vector Store management, Verification, and Interactive Search.

Usage:
    python biosage_v2.py            # Start interactive search
    python biosage_v2.py --verify   # Run Phase 2 verification
    python biosage_v2.py --populate # Embed and store chunks.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Optional dependencies for vector store
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError:
    print("❌ Missing dependencies. Please run: pip install chromadb sentence-transformers tqdm")
    sys.exit(1)

# --- Integrated Configuration ---
load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Ensure directories exist
for d in [RAW_DIR, PROC_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("BioSage-V2")

# --- Vector Store Logic ---
class ChromaStore:
    def __init__(self, collection_name: str = "biosage"):
        log.info(f"Initializing ChromaDB at {CHROMA_DIR}...")
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        
        log.info(f"Loading embedding model: {EMBED_MODEL}...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Disease variant embeddings"}
        )
        
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embed_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
        
    def populate(self, batch_size: int = 64):
        chunk_file = PROC_DIR / "chunks.json"
        if not chunk_file.exists():
            log.error(f"Input file not found: {chunk_file}. Run the chunker script first.")
            return
            
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        log.info(f"Loaded {len(chunks)} chunks to embed.")
        
        if self.collection.count() > 0:
            log.info("Wiping existing collection for a fresh load...")
            self.chroma_client.delete_collection(self.collection.name)
            self.collection = self.chroma_client.get_or_create_collection(self.collection.name)
            
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i : i + batch_size]
            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            embeddings = self.embed_texts(texts)
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
        log.info(f"✅ Stored {self.collection.count()} chunks.")

    def query(self, query_text: str, n_results: int = 5):
        query_embedding = self.embed_texts([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        return formatted

# --- Verification Logic ---
def run_verification():
    print("\nBIOSAGE -- PHASE 2 VERIFICATION")
    print("=" * 40)
    
    # Raw Data
    print("\n[1/3] Data Source Connectivity")
    raw_subdirs = ["clinvar", "omia", "omim", "cosmic", "disgenet"]
    for d in raw_subdirs:
        p = RAW_DIR / d
        files = list(p.glob("*.json")) if p.exists() else []
        print(f"  - {d:10}: {len(files)} records found")

    # Pipeline
    print("\n[2/3] Pipeline Output Status")
    proc_files = {
        "Merged Data": PROC_DIR / "all_records.json",
        "Text Chunks": PROC_DIR / "chunks.json"
    }
    for name, path in proc_files.items():
        status = "Found" if path.exists() else "Missing"
        print(f"  - {name:15}: {status}")

    # Vector Search Test
    print("\n[3/3] Vector Store Search Test")
    try:
        store = ChromaStore()
        test_q = "cystic fibrosis CFTR pathogenicity"
        print(f"  - Querying: '{test_q}'")
        results = store.query(test_q, n_results=1)
        if results:
            print(f"  - Match: {results[0]['text'][:80]}...")
        else:
            print("  - No results found. (Database might be empty)")
    except Exception as e:
        print(f"  - Error: {e}")

    print("\n" + "=" * 40)
    print("FINISHED")

# --- Interactive CLI ---
def interactive_search():
    try:
        store = ChromaStore()
    except Exception as e:
        log.error(f"Failed to load store: {e}")
        return

    print("\nSearch Engine Ready!")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("Search: ").strip()
            if not query: continue
            if query.lower() in ["quit", "exit", "q"]: break
            
            results = store.query(query, n_results=3)
            if not results:
                print("   No matches found.")
                continue
                
            for i, res in enumerate(results, 1):
                m = res["metadata"]
                print(f"\n   [{i}] {m.get('source_db')} | {m.get('disease_name')}")
                print(f"       ID: {res['id']} | Dist: {res['distance']:.4f}")
                print(f"       Text: {res['text'][:150]}...")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioSage V2 Unified Tool")
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    parser.add_argument("--populate", action="store_true", help="Embed and store data")
    parser.add_argument("--query", type=str, help="Run a single semantic query")
    
    args = parser.parse_args()
    
    if args.verify:
        run_verification()
    elif args.populate:
        store = ChromaStore()
        store.populate()
    elif args.query:
        store = ChromaStore()
        results = store.query(args.query)
        print(json.dumps(results, indent=2))
    else:
        interactive_search()
