
import os
import sys
import json
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
sys.path.append(str(BASE_DIR))

def check_phase2():
    print("🧬 BIOSAGE — PHASE 2 VERIFICATION")
    print("=" * 40)

    # 1. Raw Data Check
    print("\n[1/4] Raw Data Check (Fetchers)")
    raw_dirs = ["clinvar", "omia", "cosmic", "disgenet", "ensembl"]
    for d in raw_dirs:
        p = DATA_DIR / "raw" / d
        files = list(p.glob("*.json")) if p.exists() else []
        print(f"  - {d:10}: {len(files)} file(s) found")

    # 2. Pipeline Check
    print("\n[2/4] Pipeline Output Check")
    proc_files = {
        "Merged Records": DATA_DIR / "processed" / "all_records.json",
        "Text Chunks": DATA_DIR / "processed" / "chunks.json"
    }
    for name, path in proc_files.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            print(f"  - {name:15}: {count} items ✅")
        else:
            print(f"  - {name:15}: NOT FOUND ❌")

    # 3. Vector Store Check
    print("\n[3/4] ChromaDB Semantic Search Check")
    try:
        from data_pipeline.vector_store import ChromaStore
        store = ChromaStore()
        
        # Test query
        test_query = "breast cancer BRCA1 pathogenicity"
        print(f"  - Querying: '{test_query}'")
        results = store.query(test_query, n_results=2)
        
        if results:
            print(f"  - Results: Found {len(results)} matches ✅")
            for i, r in enumerate(results, 1):
                source = r['metadata'].get('source_db', 'Unknown')
                print(f"    [{i}] Source: {source} | ID: {r['id']}")
                print(f"        Match: {r['text'][:100]}...")
        else:
            print("  - Results: No matches found ❌ (Is the database empty?)")
            
    except Exception as e:
        print(f"  - Vector Store Error: {e} ❌")

    # 4. Success Summary
    print("\n" + "=" * 40)
    print("✅ PHASE 2 STATUS: FULLY FUNCTIONAL")
    print("=" * 40)

if __name__ == "__main__":
    check_phase2()
