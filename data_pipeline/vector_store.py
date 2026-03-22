"""
vector_store.py
───────────────
Loads chunks.json, embeds them using BioBERT, and stores them in a local ChromaDB.
Provides an interface to semantic query the database.

Usage:
    python -m data_pipeline.vector_store
"""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import PROC_DIR, CHROMA_DIR, EMBED_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Output dataset from chunker
IN_FILE = PROC_DIR / "chunks.json"

class ChromaStore:
    def __init__(self, collection_name: str = "biosage"):
        log.info(f"Initializing ChromaDB connection at {CHROMA_DIR}...")
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        
        # We manually use sentence-transformers instead of Chroma's built-in
        # so we can explicitly use BioBERT (better for medical literature)
        log.info(f"Loading embedding model: {EMBED_MODEL}...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Disease variant embeddings"}
        )
        
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate high-quality embeddings using BioBERT."""
        # Output is a numpy array, convert to list of floats for ChromaDB
        embeddings = self.embed_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
        
    def populate(self, batch_size: int = 64):
        """Read chunks.json and embed/store everything into ChromaDB."""
        if not IN_FILE.exists():
            log.error(f"Input file not found: {IN_FILE}. Run chunker first.")
            return
            
        with open(IN_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        log.info(f"Loaded {len(chunks)} chunks to embed.")
        
        # To avoid re-embedding everything if script crashes, check what's already there
        existing_count = self.collection.count()
        if existing_count > 0:
            log.info(f"Collection already has {existing_count} records. Wiping to do a fresh load.")
            # For simplicity in this demo, we do a fresh replace every time populate() runs
            self.chroma_client.delete_collection(self.collection.name)
            self.collection = self.chroma_client.get_or_create_collection(self.collection.name)
            
        # Process in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        log.info(f"Embedding and storing in {total_batches} batches...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[i : i + batch_size]
            
            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            
            embeddings = self.embed_texts(texts)
            
            # Upsert adds new or updates existing
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
        log.info(f"✅ Successfully stored {self.collection.count()} chunks in ChromaDB.")
        
    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Convert a natural language query into an embedding and search the DB."""
        # 1. Embed the user's question using the same BioBERT model
        query_embedding = self.embed_texts([query_text])[0]
        
        # 2. Search ChromaDB for the closest vectors
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # 3. Format the response nicely
        formatted_results = []
        
        # Chroma returns lists of lists since you can query multiple texts at once
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Distance is typically L2 or Cosine. Closer to 0 is better.
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": distance,
                    "score": 1.0 / (1.0 + distance) # Convert distance to a 0-1 similarity score
                })
                
        return formatted_results


if __name__ == "__main__":
    store = ChromaStore()
    store.populate()
