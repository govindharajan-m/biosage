
import os
from rag_engine import RAGEngine
from config import ANTHROPIC_API_KEY

def test_rag_flow():
    print("🧪 RAG ENGINE SMOKE TEST")
    print("=" * 30)
    
    if not ANTHROPIC_API_KEY:
        print("❌ ERROR: ANTHROPIC_API_KEY missing in .env")
        print("Please add your key to continue.")
        return

    print("--- Initializing Engine ---")
    try:
        engine = RAGEngine()
        
        test_queries = [
            "What mutations in BRCA1 are pathogenic?",
            "Tell me about cystic fibrosis clinical signs."
        ]
        
        for q in test_queries:
            print(f"\nQUERY: {q}")
            response = engine.answer(q, n_results=3)
            
            print(f"ANSWER:\n{response['answer']}")
            print(f"SOURCES FOUND: {len(response['sources'])}")
            for s in response['sources']:
                print(f"  - [{s['metadata'].get('source_db')}] {s['id']}")
                
        print("\n✅ RAG Flow Test Complete!")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")

if __name__ == "__main__":
    test_rag_flow()
