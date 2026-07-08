import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) or ".")

from inverted_index  import InvertedIndex
from query_processor import QueryProcessor



EVAL_TOPICS = [
    # Q01: matches where Messi played in a Final
    # doc=1  Argentina vs France, Final 2022       (Messi scored, Final)
    # doc=65 France vs Croatia, Final 2018         (no Messi)
    # doc=129 Germany vs Argentina, Final 2014     (Messi played)
    # doc=257 Italy vs France, Final 2006          (no Messi)
    {
        "id": "Q01",
        "query": "messi final",
        "relevant_ids": [1, 129, 553, 657, 947],
        "notes": "All World Cup Finals involving Argentina (Messi was captain/player)"
    },


    {
        "id": "Q02",
        "query": "mbappe goal",
        "relevant_ids": [1, 13, 79, 65, 43, 26, 60, 71, 107, 90],
        "notes": "All matches where Mbappé scored (all France 2018/2022 matches)"
    },


    {
        "id": "Q03",
        "query": "penalty shootout",
        "relevant_ids": [1, 8, 74, 77, 78, 200, 257, 334, 394, 458,
                          7, 9, 70, 131, 193, 262, 327, 392, 456, 503],
        "notes": "All matches decided on penalties (35 total in dataset)"
    },


    {
        "id": "Q04",
        "query": "own goal",
        "relevant_ids": [65, 77, 86, 114, 121, 124, 183, 192, 363, 411],
        "notes": "Matches with confirmed own goals"
    },


    {
        "id": "Q05",
        "query": "extra time goal",
        "relevant_ids": [77, 1, 8, 74, 208, 257, 334, 394, 457, 885],
        "notes": "Matches where a goal was scored after minute 90"
    },


    {
        "id": "Q06",
        "query": "yellow cards argentina",
        "relevant_ids": [8, 1, 131, 394, 504, 507, 657, 662, 666, 670],
        "notes": "Argentina matches where yellow cards were recorded"
    },


    {
        "id": "Q07",
        "query": "referee marciniak",
        "relevant_ids": [1, 16, 43, 102, 122],
        "notes": "Exact match on referee name — 5 matches"
    },


    {
        "id": "Q08",
        "query": "captain modric",
        "relevant_ids": [7, 39, 70, 78, 108, 124, 280, 292, 2],
        "notes": "All Croatia 2018/2022 matches (Modrić was captain)"
    },


    {
        "id": "Q09",
        "query": "team:Brazil stage:Quarter-finals",
        "relevant_ids": [7, 72, 136, 199, 262, 327, 392, 456, 559, 737],
        "notes": "Exact field filter — all Brazil quarter-final appearances"
    },


    {
        "id": "Q10",
        "query": "red card",
        "relevant_ids": [5, 8, 19, 34, 256, 297, 304, 339, 348, 460],
        "notes": "Matches where at least one player was sent off"
    },


    {
        "id": "Q11",
        "query": "penalty miss",
        "relevant_ids": [1, 8, 74, 77, 78, 200, 257, 334, 394, 458],
        "notes": "Matches with documented penalty misses"
    },

    # Q12: all Morocco matches (team filter)
    {
        "id": "Q12",
        "query": "team:Morocco",
        "relevant_ids": [2, 3, 5, 9, 21, 38, 53, 95, 109, 126],
        "notes": "All matches involving Morocco (field search)"
    },

]



#  METRIC FUNCTIONS


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:

    if k == 0:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:

    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[int], relevant: set[int]) -> float:

    if not relevant:
        return 0.0
    num_found = 0
    score = 0.0
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            num_found += 1
            score += num_found / rank
    return score / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:

    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


#  EVALUATOR


class Evaluator:
    def __init__(self, qp: QueryProcessor):
        self.qp = qp

    def run(self, topics: list[dict], top_k: int = 10) -> dict:

        print(f"\n{'='*68}")
        print(f"  EVALUATION RESULTS  ({len(topics)} queries, top_k={top_k})")
        print(f"{'='*68}")

        per_query = []
        all_ap = all_rr = all_p5 = all_p10 = all_r10 = 0.0

        for t in topics:
            qid      = t["id"]
            query    = t["query"]
            relevant = set(t["relevant_ids"])

            results, _info = self.qp.search(query, top_k=top_k)
            retrieved_ids = [r.doc_id for r in results]

            ap  = average_precision(retrieved_ids, relevant)
            rr  = reciprocal_rank(retrieved_ids, relevant)
            p5  = precision_at_k(retrieved_ids, relevant, k=5)
            p10 = precision_at_k(retrieved_ids, relevant, k=10)
            r10 = recall_at_k(retrieved_ids, relevant, k=10)

            all_ap  += ap
            all_rr  += rr
            all_p5  += p5
            all_p10 += p10
            all_r10 += r10


            if retrieved_ids and retrieved_ids[0] in relevant:
                mark = "✓"
            elif any(d in relevant for d in retrieved_ids):
                mark = "~"
            else:
                mark = "✗"

            print(f"\n  {qid} {mark}  \"{query}\"")
            print(f"       AP={ap:.3f}  P@5={p5:.3f}  P@10={p10:.3f}  R@10={r10:.3f}  RR={rr:.3f}")
            print(f"       top-5 retrieved: {retrieved_ids[:5]}")
            print(f"       relevant (sample): {sorted(relevant)[:5]}")

            per_query.append({
                "id": qid, "query": query,
                "relevant_ids": list(relevant),
                "retrieved_ids": retrieved_ids,
                "AP": round(ap,4), "RR": round(rr,4),
                "P@5": round(p5,4), "P@10": round(p10,4), "R@10": round(r10,4),
            })

        n = len(topics)
        summary = {
            "MAP":   round(all_ap  / n, 4),
            "MRR":   round(all_rr  / n, 4),
            "P@5":   round(all_p5  / n, 4),
            "P@10":  round(all_p10 / n, 4),
            "R@10":  round(all_r10 / n, 4),
            "num_queries": n,
            "per_query": per_query,
        }

        print(f"\n{'='*68}")
        print(f"  SUMMARY")
        print(f"{'='*68}")
        print(f"  MAP  (Mean Average Precision)  : {summary['MAP']:.4f}")
        print(f"  MRR  (Mean Reciprocal Rank)    : {summary['MRR']:.4f}")
        print(f"  P@5  (Precision at 5)          : {summary['P@5']:.4f}")
        print(f"  P@10 (Precision at 10)         : {summary['P@10']:.4f}")
        print(f"  R@10 (Recall at 10)            : {summary['R@10']:.4f}")
        print(f"{'='*68}")

        return summary

    def save(self, summary: dict, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved → {path}")



#  MAIN

if __name__ == "__main__":
    INDEX_PATH = "data/index.json"
    if not os.path.exists(INDEX_PATH):
        print("[!] data/index.json not found. Run main.py first.")
        sys.exit(1)

    idx = InvertedIndex()
    idx.load(INDEX_PATH)
    qp  = QueryProcessor(idx)
    ev  = Evaluator(qp)

    summary = ev.run(EVAL_TOPICS, top_k=10)
    ev.save(summary, "data/eval_results.json")

    print("""
HOW TO ADD YOUR OWN QUERIES FOR THE REPORT
-------------------------------------------
1. Run:  python evaluator.py
         and look at the printed doc_ids for each query.

2. Open data/eval_results.json to see every returned doc_id.

3. For each query, decide which doc_ids are truly relevant
   (look them up in data/documents.json if unsure).

4. Add a new entry to EVAL_TOPICS at the top of this file:
   {
       "id":           "Q13",
       "query":        "your query here",
       "relevant_ids": [doc_id1, doc_id2, ...],
       "notes":        "why these are relevant"
   }

5. Re-run python evaluator.py — MAP will update automatically.
""")