"""
main.py
========
World Cup IR System — main entry point

USAGE:
    # First time: build index from CSV
    python main.py --csv path/to/matches_1930_2022.csv

    # Interactive search (index already built)
    python main.py --search

    # Run evaluation
    python main.py --eval

    # Single query
    python main.py --query "messi final"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from document_builder import build_all_documents, save_documents, load_documents
from preprocessor     import preprocess_documents
from inverted_index   import InvertedIndex
from query_processor  import QueryProcessor, print_results
from evaluator        import Evaluator, EVAL_TOPICS

# ── file paths ────────────────────────────────────────────────
DATA_DIR       = "data"
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")
INDEX_PATH     = os.path.join(DATA_DIR, "index.json")
EVAL_RESULTS   = os.path.join(DATA_DIR, "eval_results.json")


# ── pipeline ──────────────────────────────────────────────────

def build_index(csv_path: str) -> tuple[InvertedIndex, QueryProcessor]:
    """Full pipeline: CSV → documents → preprocess → index."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "="*50)
    print("Step 1: Read CSV and build documents")
    print("="*50)
    documents = build_all_documents(csv_path)
    save_documents(documents, DOCUMENTS_PATH)

    print("\n" + "="*50)
    print("Step 2: Preprocess")
    print("="*50)
    documents = preprocess_documents(documents)

    print("\n" + "="*50)
    print("Step 3: Build Inverted Index")
    print("="*50)
    idx = InvertedIndex()
    idx.build(documents)
    idx.save(INDEX_PATH)
    idx.stats()

    return idx, QueryProcessor(idx)


def load_existing_index() -> tuple[InvertedIndex, QueryProcessor]:
    """Load a previously built index from disk."""
    if not os.path.exists(INDEX_PATH):
        print(f"[!] Index not found at {INDEX_PATH}")
        print("    Run first with:  python main.py --csv path/to/data.csv")
        sys.exit(1)
    idx = InvertedIndex()
    idx.load(INDEX_PATH)
    idx.stats()
    return idx, QueryProcessor(idx)


# ── modes ─────────────────────────────────────────────────────

def interactive_search(qp: QueryProcessor):
    """Interactive search loop."""
    print("\n" + "="*60)
    print("  World Cup IR Search System (1930-2022)")
    print("="*60)
    print("  Query types:")
    print("    keyword  : messi final")
    print("    boolean  : mbappe AND goal")
    print("    negation : argentina NOT france")
    print("    field    : team:Argentina  stage:Final  referee:Marciniak")
    print("    combined : team:Argentina stage:Final messi")
    print("    exit     : q  or  quit")
    print("="*60)

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("[exit]")
            break

        results = qp.search(query, top_k=10)
        print_results(results, query)


def run_evaluation(qp: QueryProcessor):
    """Run evaluation using the topics defined in evaluator.py."""
    ev      = Evaluator(qp)
    summary = ev.run(EVAL_TOPICS, top_k=10)
    ev.save(summary, EVAL_RESULTS)
    return summary


def run_single_query(qp: QueryProcessor, query: str):
    results = qp.search(query, top_k=10)
    print_results(results, query)


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="World Cup IR Search System")
    parser.add_argument("--csv",    type=str, help="Path to CSV dataset")
    parser.add_argument("--search", action="store_true", help="Interactive search")
    parser.add_argument("--eval",   action="store_true", help="Run evaluation")
    parser.add_argument("--query",  type=str, help="Run a single query")
    args = parser.parse_args()

    # load or build index
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"[!] CSV file not found: {args.csv}")
            sys.exit(1)
        idx, qp = build_index(args.csv)
    else:
        idx, qp = load_existing_index()

    # execute requested mode
    if args.query:
        run_single_query(qp, args.query)
    elif args.eval:
        run_evaluation(qp)
    else:
        # default: interactive search
        interactive_search(qp)


if __name__ == "__main__":
    main()