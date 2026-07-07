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

INSIDE INTERACTIVE SEARCH:
    messi                  -> top 10 results (default)
    messi /5               -> top 5 results
    team:Brazil /20        -> top 20 results
    stage:Final /all       -> ALL matching results
    mbappe AND goal        -> boolean search
    team:Argentina /all    -> all Argentina matches
"""

import os
import sys
import re
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from document_builder import build_all_documents, save_documents
from preprocessor     import preprocess_documents
from inverted_index   import InvertedIndex
from query_processor  import QueryProcessor
from evaluator        import Evaluator, EVAL_TOPICS

# ── file paths ────────────────────────────────────────────────
DATA_DIR       = "data"
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")
INDEX_PATH     = os.path.join(DATA_DIR, "index.json")
EVAL_RESULTS   = os.path.join(DATA_DIR, "eval_results.json")


# ── pipeline ──────────────────────────────────────────────────

def build_index(csv_path: str) -> tuple[InvertedIndex, QueryProcessor]:
    """Full pipeline: CSV -> documents -> preprocess -> index."""
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
    return idx, QueryProcessor(idx)


# ── display ───────────────────────────────────────────────────

def print_results(results, query: str, total_found: int):
    """Print ranked results in a clean table."""
    print(f"\n{'='*70}")
    print(f'  Query: "{query}"')
    print(f"  Showing {len(results)} of {total_found} total matches found")
    print(f"{'='*70}")

    if not results:
        print("  [no results found]")
        return

    print(f"  {'Rank':<5} {'DocID':<7} {'Home':<22} {'Away':<22} "
          f"{'Stage':<18} {'Score':<8} {'Year':<6} {'TF-IDF'}")
    print(f"  {'-'*5} {'-'*7} {'-'*22} {'-'*22} {'-'*18} {'-'*8} {'-'*6} {'-'*7}")

    for i, r in enumerate(results, 1):
        m     = r.metadata
        home  = m.get("home_team", "?")[:21]
        away  = m.get("away_team", "?")[:21]
        stage = m.get("stage",     "")[:17]
        score = m.get("score",     "")[:7]
        date  = m.get("date",      "")
        year  = date[:4] if date and date != "nan" else ""
        print(f"  {i:<5} {r.doc_id:<7} {home:<22} {away:<22} "
              f"{stage:<18} {score:<8} {year:<6} {r.score:.3f}")


# ── interactive search ────────────────────────────────────────

def interactive_search(qp: QueryProcessor):
    """Interactive search loop with variable result count."""
    print("\n" + "="*70)
    print("  World Cup IR Search System  (1930 - 2022)  |  964 matches")
    print("="*70)
    print("  QUERY TYPES:")
    print("    keyword   :  messi final")
    print("    boolean   :  mbappe AND goal  |  penalty OR extra time")
    print("    negation  :  argentina NOT france")
    print("    field     :  team:Brazil  stage:Final  referee:Marciniak")
    print("    combined  :  team:Argentina stage:Final messi")
    print()
    print("  RESULT COUNT  (add at the end of any query):")
    print("    messi /5          ->  show top 5")
    print("    team:Brazil /20   ->  show top 20")
    print("    stage:Final /all  ->  show ALL results")
    print("    (default is 10 if you don't add /N)")
    print()
    print("    exit: q  or  quit")
    print("="*70)

    while True:
        try:
            raw = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            print("[exit]")
            break

        # ── parse optional /N or /all suffix ──────────────────
        top_k = 10
        query = raw

        suffix_match = re.search(r"\s+/(\w+)$", raw)
        if suffix_match:
            val   = suffix_match.group(1).lower()
            query = raw[:suffix_match.start()].strip()
            if val == "all":
                top_k = 964
            else:
                try:
                    top_k = int(val)
                except ValueError:
                    print(f"  [!] '/{val}' not recognised — using default 10")
                    top_k = 10

        # ── run search ────────────────────────────────────────
        # always retrieve everything, then slice — so we can show
        # "X of Y total" even when the user asks for only 5
        all_results = qp.search(query, top_k=964)
        total_found = len(all_results)
        results     = all_results[:top_k]

        print_results(results, query, total_found)

        # helpful hint when results are truncated
        if total_found > top_k:
            remaining = total_found - top_k
            print(f"\n  [{remaining} more results not shown — "
                  f"add /{total_found} or /all to see everything]")


# ── other modes ───────────────────────────────────────────────

def run_evaluation(qp: QueryProcessor):
    ev      = Evaluator(qp)
    summary = ev.run(EVAL_TOPICS, top_k=10)
    ev.save(summary, EVAL_RESULTS)
    return summary


def run_single_query(qp: QueryProcessor, query: str, top_k: int = 10):
    # support /N and /all suffix in --query mode too
    suffix_match = re.search(r"\s+/(\w+)$", query)
    if suffix_match:
        val   = suffix_match.group(1).lower()
        query = query[:suffix_match.start()].strip()
        if val == "all":
            top_k = 964
        else:
            try:
                top_k = int(val)
            except ValueError:
                pass
    all_results = qp.search(query, top_k=964)
    total_found = len(all_results)
    results     = all_results[:top_k]
    print_results(results, query, total_found)
    if total_found > top_k:
        print(f"\n  [{total_found - top_k} more — add /all or /N to query to see more]")


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="World Cup IR Search System")
    parser.add_argument("--csv",    type=str,          help="Path to CSV dataset (builds index)")
    parser.add_argument("--search", action="store_true",help="Interactive search mode")
    parser.add_argument("--eval",   action="store_true",help="Run evaluation")
    parser.add_argument("--query",  type=str,          help="Run a single query")
    parser.add_argument("--topk",   type=int, default=10, help="Number of results (default 10)")
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
        run_single_query(qp, args.query, top_k=args.topk)
    elif args.eval:
        run_evaluation(qp)
    else:
        interactive_search(qp)


if __name__ == "__main__":
    main()