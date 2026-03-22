"""
query_demo.py
──────────────
Interactive CLI to test the ChromaDB vector embeddings.
Type a biological question, and see which chunks match best!

Usage:
    python query_demo.py
"""

from data_pipeline.vector_store import ChromaStore

def main():
    print("🧬 Loading BioSage Vector Store...")
    print("   Initializing BioBERT embeddings (this takes a few seconds)...")
    
    # Init store
    try:
        store = ChromaStore()
    except Exception as e:
        print(f"\n❌ Failed to load ChromaDB: {e}")
        print("   Did you run 'python -m data_pipeline.normalizer' -> 'chunker' -> 'vector_store' first?")
        return

    print("\n✅ Store loaded successfully!")
    print("\nTry queries like:")
    print(" - What SNPs cause cystic fibrosis?")
    print(" - Breast cancer BRAC1 pathogenic variants")
    print(" - Dog hip dysplasia mutations")
    print("\nType 'quit' or 'exit' to stop.")
    print("─" * 60)

    while True:
        try:
            query = input("\n🔍 Search query: ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                break
                
            results = store.query(query, n_results=3)
            
            if not results:
                print("   No matches found (is the database empty?)")
                continue
                
            print(f"\n   Top 3 matches for '{query}':")
            for i, res in enumerate(results, 1):
                meta = res["metadata"]
                dist = res["distance"]
                text = res["text"]
                
                print(f"\n   [{i}] {meta.get('source_db', 'Unknown')} ID: {meta.get('record_id', '?')}")
                print(f"       Distance : {dist:.4f} (lower is better)")
                print(f"       Disease  : {meta.get('disease_name', '?')}")
                print(f"       Gene     : {meta.get('gene', '?')} | Species: {meta.get('species_common', '?')}")
                # Print truncated text 
                short = text[:150] + "..." if len(text) > 150 else text
                print(f"       Context  : {short}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error during query: {e}")

if __name__ == "__main__":
    main()
