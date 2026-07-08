

import math
import re
from collections import defaultdict
from inverted_index import InvertedIndex
from preprocessor import preprocess, preprocess_query


# ============================================================
# Levenshtein Edit Distance
# ============================================================

def levenshtein_distance(s1: str, s2: str) -> int:

    n, m = len(s1), len(s2)


    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:

                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    return dp[n][m]


def find_spelling_suggestions(
    term: str,
    index: InvertedIndex,
    max_distance: int = 2,
    max_suggestions: int = 5
) -> list[tuple[str, int]]:

    term_len = len(term)
    suggestions = []

    all_terms = list(index.index.keys())

    for candidate in all_terms:
        cand_len = len(candidate)

        if abs(cand_len - term_len) > max_distance:
            continue

        dist = levenshtein_distance(term, candidate)
        if dist <= max_distance and dist > 0:
            suggestions.append((candidate, dist))

    suggestions.sort(key=lambda x: (x[1], -index.index[x[0]].df))

    return suggestions[:max_suggestions]


# SpellingCorrector


class SpellingCorrector:


    def __init__(self, index: InvertedIndex, max_distance: int = 2):
        self.index = index
        self.max_distance = max_distance

    def suggest(self, term: str, max_suggestions: int = 5) -> list[tuple[str, int]]:

        term = term.lower().strip()

        if self.index.lookup(term) is not None:
            return []

        return find_spelling_suggestions(
            term, self.index,
            max_distance=self.max_distance,
            max_suggestions=max_suggestions
        )

    def check_query_tokens(self, tokens: list[str]) -> dict[str, list[tuple[str, int]]]:

        corrections = {}
        for token in tokens:
            suggestions = self.suggest(token)
            if suggestions:
                corrections[token] = suggestions
        return corrections



# QueryResult


class QueryResult:


    def __init__(self, doc_id: int, score: float, metadata: dict):
        self.doc_id = doc_id
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        home = self.metadata.get("home_team", "?")
        away = self.metadata.get("away_team", "?")
        stage = self.metadata.get("stage", "")
        return f"[DocID:{self.doc_id}] {home} vs {away} | {stage} | score={self.score:.4f}"



# QueryProcessor


class QueryProcessor:

    def __init__(self, index: InvertedIndex):
        self.index = index
        self.corrector = SpellingCorrector(index, max_distance=2)


    # Wildcard Search


    def _expand_wildcards(self, tokens: list[str]) -> tuple[list[str], dict[str, list[str]]]:

        expanded = []
        wildcard_map = {}

        for token in tokens:
            if "*" in token:
                # بررسی که * فقط در انتها است
                if not token.endswith("*") or token.count("*") > 1:
                    # wildcard پیچیده -> نادیده بگیر
                    expanded.append(token.replace("*", ""))
                    continue

                prefix = token[:-1].lower()  # حذف *
                if not prefix:

                    continue

                matched = [
                    term for term in self.index.index.keys()
                    if term.startswith(prefix)
                ]

                wildcard_map[token] = matched

                if matched:
                    expanded.extend(matched)
                else:

                    expanded.append(prefix)
            else:
                expanded.append(token)

        return expanded, wildcard_map

    def wildcard_search(self, query: str, top_k: int = 10) -> tuple[list[QueryResult], dict]:

        tokens = preprocess(query.replace("*", "WILDCARD_MARKER*"), keep_stop_words=False)


        raw_tokens = query.lower().split()
        wildcard_tokens = [t for t in raw_tokens if "*" in t]
        normal_query_parts = [t for t in raw_tokens if "*" not in t]


        normal_tokens = preprocess(" ".join(normal_query_parts), keep_stop_words=False)


        wildcard_map = {}
        expanded_from_wildcards = []

        for wt in wildcard_tokens:
            if not wt.endswith("*") or wt.count("*") > 1:

                prefix = wt.replace("*", "")
                matched = [term for term in self.index.index.keys() if term.startswith(prefix)]
            else:
                prefix = wt[:-1]
                matched = [term for term in self.index.index.keys() if term.startswith(prefix)]

            wildcard_map[wt] = matched
            expanded_from_wildcards.extend(matched)

        all_tokens = normal_tokens + expanded_from_wildcards

        if not all_tokens:
            return [], wildcard_map


        candidate_docs = set()
        for token in all_tokens:
            candidate_docs.update(self.index.get_doc_ids_for_term(token))


        scored = self._score_documents(candidate_docs, all_tokens)
        return self._top_k(scored, top_k), wildcard_map


    # Spelling Correction


    def get_spelling_suggestions(self, query: str) -> dict[str, list[tuple[str, int]]]:


        if "*" in query:
            return {}

        tokens = preprocess(query, keep_stop_words=False)
        return self.corrector.check_query_tokens(tokens)


    def keyword_search(self, query: str, top_k: int = 10) -> list[QueryResult]:

        tokens = preprocess(query, keep_stop_words=False)
        if not tokens:
            return []


        candidate_docs = set()
        for token in tokens:
            candidate_docs.update(self.index.get_doc_ids_for_term(token))


        scored = self._score_documents(candidate_docs, tokens)
        return self._top_k(scored, top_k)



    def boolean_search(self, query: str) -> list[QueryResult]:

        result_ids = self._parse_boolean_query(query)

        if not result_ids:
            return []

        clean_query = re.sub(r"\b(AND|OR|NOT)\b", " ", query, flags=re.IGNORECASE)
        tokens = preprocess(clean_query, keep_stop_words=False)

        scored = self._score_documents(result_ids, tokens)
        return self._top_k(scored, top_k=len(result_ids))

    def _parse_boolean_query(self, query: str) -> set[int]:

        query = query.strip()

        if re.search(r"\bOR\b", query, re.IGNORECASE):
            parts = re.split(r"\bOR\b", query, flags=re.IGNORECASE)
            result = set()
            for part in parts:
                part_result = self._parse_boolean_query(part.strip())
                result = result.union(part_result)
            return result


        if re.search(r"\bAND\b", query, re.IGNORECASE):
            parts = re.split(r"\bAND\b", query, flags=re.IGNORECASE)
            result = None
            for part in parts:
                part_result = self._parse_boolean_query(part.strip())
                if result is None:
                    result = part_result
                else:
                    result = result.intersection(part_result)
            return result or set()

        if re.match(r"^NOT\s+", query, re.IGNORECASE):
            rest = re.sub(r"^NOT\s+", "", query, flags=re.IGNORECASE)
            not_docs = self._parse_boolean_query(rest.strip())
            all_docs = self.index.get_all_doc_ids()
            return all_docs - not_docs


        query_clean = query.strip("() ")
        tokens = preprocess(query_clean, keep_stop_words=False)

        if not tokens:
            return set()

        result = None
        for token in tokens:
            doc_ids = set(self.index.get_doc_ids_for_term(token))
            if result is None:
                result = doc_ids
            else:
                result = result.intersection(doc_ids)

        return result or set()



    def fielded_search(self, query: str, top_k: int = 10) -> list[QueryResult]:

        field_filters = {}
        remaining_query = query

        field_pattern = re.compile(
            r"(\w+):[\"']([^\"']+)[\"']|(\w+):(\S+)",
            re.IGNORECASE
        )

        for match in field_pattern.finditer(query):
            if match.group(1):
                field = match.group(1).lower()
                value = match.group(2).strip()
            else:
                field = match.group(3).lower()
                value = match.group(4).strip()


            if field == "team":
                field_filters["team"] = value
            elif field in ("round", "stage"):
                field_filters["stage"] = value
            elif field == "player":
                field_filters["player"] = value
            elif field == "referee":
                field_filters["referee"] = value
            elif field in self.index.field_indexes:
                field_filters[field] = value

            remaining_query = remaining_query.replace(match.group(0), "").strip()

        if not field_filters:
            return self.keyword_search(query, top_k)


        candidate_docs = None

        for field, value in field_filters.items():
            value_tokens = preprocess(value, keep_stop_words=False)
            if not value_tokens:
                continue

            field_docs = None
            for token in value_tokens:
                if field == "team":
                    home_pl = self.index.lookup_field("home_team", token)
                    away_pl = self.index.lookup_field("away_team", token)
                    home_ids = set(home_pl.get_doc_ids()) if home_pl else set()
                    away_ids = set(away_pl.get_doc_ids()) if away_pl else set()
                    token_docs = home_ids.union(away_ids)
                elif field == "player":
                    pl = self.index.lookup_field("goal_scorers", token)
                    token_docs = set(pl.get_doc_ids()) if pl else set()
                else:
                    pl = self.index.lookup_field(field, token)
                    token_docs = set(pl.get_doc_ids()) if pl else set()

                if field_docs is None:
                    field_docs = token_docs
                else:
                    field_docs = field_docs.intersection(token_docs)

            if field_docs is None:
                field_docs = set()

            if candidate_docs is None:
                candidate_docs = field_docs
            else:
                candidate_docs = candidate_docs.intersection(field_docs)

        if candidate_docs is None:
            candidate_docs = self.index.get_all_doc_ids()

        tokens = preprocess(remaining_query) if remaining_query else []

        if tokens:
            keyword_docs = set()
            for token in tokens:
                keyword_docs.update(self.index.get_doc_ids_for_term(token))
            candidate_docs = candidate_docs.intersection(keyword_docs)

        if not candidate_docs:
            return []

        scored = self._score_documents(candidate_docs, tokens if tokens else [])

        if not tokens:
            results = []
            for doc_id in sorted(candidate_docs):
                meta = self.index.get_doc_info(doc_id)
                results.append(QueryResult(doc_id=doc_id, score=1.0, metadata=meta))
            return results[:top_k]

        return self._top_k(scored, top_k)



    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> tuple[list[QueryResult], dict]:

        query = query.strip()
        info = {}

        #  Wildcard
        if "*" in query:
            results, wc_map = self.wildcard_search(query, top_k)
            info["wildcard_map"] = wc_map
            return results, info


        has_boolean = bool(re.search(r"\b(AND|OR|NOT)\b", query, re.IGNORECASE))
        has_field = bool(re.search(r"\w+:[\"']?[^\s]", query))

        if has_field:
            results = self.fielded_search(query, top_k)
        elif has_boolean:
            results = self.boolean_search(query)
            results = results[:top_k]
        else:
            results = self.keyword_search(query, top_k)


        if len(results) < 3:
            suggestions = self.get_spelling_suggestions(query)
            if suggestions:
                info["spelling"] = suggestions

        return results, info


    # محاسبه TF-IDF


    def _compute_tf(self, tf_raw: int, doc_length: int) -> float:

        if tf_raw == 0:
            return 0.0
        return 1 + math.log10(tf_raw)

    def _compute_idf(self, df: int) -> float:

        N = self.index.doc_count
        if df == 0 or N == 0:
            return 0.0
        return math.log10(N / df)

    def _score_documents(self, doc_ids: set[int], tokens: list[str]) -> list[QueryResult]:

        scores = defaultdict(float)

        for token in tokens:
            pl = self.index.lookup(token)
            if pl is None:
                continue

            idf = self._compute_idf(pl.df)

            for posting in pl.postings:
                if posting.doc_id not in doc_ids:
                    continue

                doc_len = self.index.doc_lengths.get(posting.doc_id, 1)
                tf = self._compute_tf(posting.tf, doc_len)
                scores[posting.doc_id] += tf * idf


        for doc_id in doc_ids:
            if doc_id not in scores:
                scores[doc_id] = 0.0

        results = []
        for doc_id, score in scores.items():
            meta = self.index.get_doc_info(doc_id)
            results.append(QueryResult(doc_id=doc_id, score=score, metadata=meta))

        return results

    def _top_k(self, results: list[QueryResult], top_k: int) -> list[QueryResult]:
        """نتایج را بر اساس امتیاز مرتب و top_k تای اول را برمی‌گرداند."""
        results.sort(key=lambda r: -r.score)
        return results[:top_k]



# Display helpers

def print_results(results: list[QueryResult], query: str):
    """نتایج را به شکل خوانا چاپ می‌کند."""
    print(f"\n=== نتایج برای: \"{query}\" ===")

    if not results:
        print("  [هیچ نتیجه‌ای پیدا نشد]")
        return

    print(f"  {len(results)} نتیجه پیدا شد:\n")
    for i, r in enumerate(results, 1):
        home = r.metadata.get("home_team", "?")
        away = r.metadata.get("away_team", "?")
        stage = r.metadata.get("stage", "")
        score = r.metadata.get("score", "")
        print(f"  Rank {i:2d} | DocID: {r.doc_id:3d} | {home} vs {away} | {stage} | Score: {r.score:.4f}")



if __name__ == "__main__":
    import json, os, sys

    sys.path.insert(0, ".")
    from document_builder import build_all_documents
    from preprocessor import preprocess_documents


    if os.path.exists("data/index.json"):
        idx = InvertedIndex()
        idx.load("data/index.json")
    else:
        print("[!] index.json پیدا نشد. ساخت مجدد...")

        with open("data/documents.json", encoding="utf-8") as f:
            docs = json.load(f)
        docs = preprocess_documents(docs)

        idx = InvertedIndex()
        idx.build(docs)
        os.makedirs("data", exist_ok=True)
        idx.save("data/index.json")

    qp = QueryProcessor(idx)


    print("\n=== تست Wildcard ===")
    wildcard_tests = ["mess*", "mba*", "pen*", "arg*"]
    for q in wildcard_tests:
        results, info = qp.search(q, top_k=5)
        wc_map = info.get("wildcard_map", {})
        for pattern, matched in wc_map.items():
            print(f"  {pattern} -> matched terms: {matched[:10]}")
        print_results(results, q)

    print("\n=== تست Spelling Correction ===")
    spelling_tests = ["messy", "mbape", "penalti", "argentna"]
    for q in spelling_tests:
        suggestions = qp.get_spelling_suggestions(q)
        if suggestions:
            for token, suggs in suggestions.items():
                best = suggs[0] if suggs else None
                if best:
                    print(f"  Did you mean '{best[0]}' instead of '{token}'? (distance={best[1]})")
        results, info = qp.search(q, top_k=3)
        print_results(results, q)
