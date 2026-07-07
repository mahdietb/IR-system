"""
query_processor.py
===================
بخش ۴ + ۵ پروژه: پردازش پرس‌وجو و رتبه‌بندی

سه نوع پرس‌وجو پشتیبانی می‌شود:
1. Keyword  : "messi final"         -> اسناد دارای هر دو کلمه
2. Boolean  : "mbappe AND goal"     -> عملگرهای AND/OR/NOT
3. Ranked   : TF-IDF score برای هر سند
4. Fielded  : "team:Argentina"      -> جستجو در فیلد خاص

TF-IDF چیست؟
-----------
TF  (Term Frequency)  : این کلمه چند بار در این سند آمده؟
                        هرچه بیشتر -> سند مرتبط‌تر است.
IDF (Inverse Document Frequency): این کلمه در چند سند کل آمده؟
                        هرچه نادرتر -> مهم‌تر است.

مثال: "goal" در ۶۰ از ۶۴ مسابقه -> IDF کم (کلمه رایج)
      "messi" در ۳ از ۶۴ مسابقه -> IDF بالا (کلمه خاص)

امتیاز نهایی = TF × IDF
"""

import math
import re
from collections import defaultdict
from inverted_index import InvertedIndex
from preprocessor import preprocess, preprocess_query


class QueryResult:
    """نتیجه یک پرس‌وجو."""

    def __init__(self, doc_id: int, score: float, metadata: dict):
        self.doc_id = doc_id
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        home = self.metadata.get("home_team", "?")
        away = self.metadata.get("away_team", "?")
        stage = self.metadata.get("stage", "")
        return f"[DocID:{self.doc_id}] {home} vs {away} | {stage} | score={self.score:.4f}"


class QueryProcessor:
    """
    پردازنده پرس‌وجو.

    از یک InvertedIndex استفاده می‌کند و پرس‌وجوها را پردازش می‌کند.
    """

    def __init__(self, index: InvertedIndex):
        self.index = index

    # ============================================================
    # ۱) جستجوی Keyword ساده
    # ============================================================

    def keyword_search(self, query: str, top_k: int = 10) -> list[QueryResult]:
        """
        جستجوی ساده کلمات کلیدی.
        اسنادی برمی‌گرداند که حداقل یکی از کلمات پرس‌وجو را دارند.
        نتایج بر اساس TF-IDF رتبه‌بندی می‌شوند.

        مثال: "messi final" -> اسناد دارای "messi" یا "final"
        """
        tokens = preprocess(query, keep_stop_words=False)
        if not tokens:
            return []

        # پیدا کردن همه اسناد مرتبط
        candidate_docs = set()
        for token in tokens:
            candidate_docs.update(self.index.get_doc_ids_for_term(token))

        # محاسبه امتیاز TF-IDF برای هر سند
        scored = self._score_documents(candidate_docs, tokens)
        return self._top_k(scored, top_k)

    # ============================================================
    # ۲) جستجوی Boolean
    # ============================================================

    def boolean_search(self, query: str) -> list[QueryResult]:
        """
        جستجوی Boolean با عملگرهای AND، OR، NOT.

        مثال‌ها:
            "mbappe AND goal"
            "penalty OR extra time"
            "argentina NOT brazil"
            "messi AND (final OR semifinal)"

        نتایج Boolean ترتیب ندارند (همه یکسان‌اند)،
        ولی ما TF-IDF هم اضافه می‌کنیم.
        """
        result_ids = self._parse_boolean_query(query)

        # اگر نتیجه‌ای نبود
        if not result_ids:
            return []

        # توکن‌های پرس‌وجو برای محاسبه امتیاز
        clean_query = re.sub(r"\b(AND|OR|NOT)\b", " ", query, flags=re.IGNORECASE)
        tokens = preprocess(clean_query, keep_stop_words=False)

        scored = self._score_documents(result_ids, tokens)
        return self._top_k(scored, top_k=len(result_ids))

    def _parse_boolean_query(self, query: str) -> set[int]:
        """
        پارسر ساده برای پرس‌وجوی Boolean.
        اولویت: NOT > AND > OR
        """
        # تبدیل به uppercase برای تشخیص عملگرها
        # اما کلمات جستجو را lowercase می‌کنیم

        # پشتیبانی از پرانتز با تقسیم
        query = query.strip()

        # ابتدا OR را پردازش می‌کنیم (پایین‌ترین اولویت)
        if re.search(r"\bOR\b", query, re.IGNORECASE):
            parts = re.split(r"\bOR\b", query, flags=re.IGNORECASE)
            result = set()
            for part in parts:
                part_result = self._parse_boolean_query(part.strip())
                result = result.union(part_result)
            return result

        # سپس AND را پردازش می‌کنیم
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

        # سپس NOT را پردازش می‌کنیم
        if re.match(r"^NOT\s+", query, re.IGNORECASE):
            rest = re.sub(r"^NOT\s+", "", query, flags=re.IGNORECASE)
            not_docs = self._parse_boolean_query(rest.strip())
            all_docs = self.index.get_all_doc_ids()
            return all_docs - not_docs

        # اگر هیچ عملگری نبود، keyword ساده
        query_clean = query.strip("() ")
        tokens = preprocess(query_clean, keep_stop_words=False)

        if not tokens:
            return set()

        # AND ضمنی: همه کلمات باید در سند باشند
        result = None
        for token in tokens:
            doc_ids = set(self.index.get_doc_ids_for_term(token))
            if result is None:
                result = doc_ids
            else:
                result = result.intersection(doc_ids)

        return result or set()

    # ============================================================
    # ۳) جستجوی فیلدی
    # ============================================================

    def fielded_search(self, query: str, top_k: int = 10) -> list[QueryResult]:
        """
        جستجو در فیلدهای خاص.

        فرمت‌های پشتیبانی‌شده:
            team:Argentina          -> home یا away
            home_team:Argentina     -> فقط تیم خانگی
            stage:Final             -> مرحله
            referee:Marciniak       -> داور
            round:Quarter-finals    -> (مترادف stage)

        مثال مرکب: "team:Argentina stage:Final"
        """
        # استخراج filter های فیلدی
        field_filters = {}
        remaining_query = query

        # پیدا کردن الگوهای field:value
        field_pattern = re.compile(
            r"(\w+):([\"']([^\"']+)[\"']|(\S+))",
            re.IGNORECASE
        )

        for match in field_pattern.finditer(query):
            field = match.group(1).lower()
            value = (match.group(3) or match.group(4) or "").strip()

            # نرمال‌سازی نام فیلدها
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
            # هیچ فیلتر فیلدی نبود، جستجوی عادی
            return self.keyword_search(query, top_k)

        # اعمال filter های فیلدی
        candidate_docs = None

        for field, value in field_filters.items():
            value_tokens = preprocess(value, keep_stop_words=False)
            if not value_tokens:
                continue

            field_docs = None
            for token in value_tokens:
                if field == "team":
                    # جستجو در هر دو تیم
                    home_pl = self.index.lookup_field("home_team", token)
                    away_pl = self.index.lookup_field("away_team", token)
                    home_ids = set(home_pl.get_doc_ids()) if home_pl else set()
                    away_ids = set(away_pl.get_doc_ids()) if away_pl else set()
                    token_docs = home_ids.union(away_ids)
                elif field == "player":
                    # جستجو در goal scorers
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

        # اگر query اضافی هم بود، رتبه‌بندی با آن
        tokens = preprocess(remaining_query) if remaining_query else []

        if tokens:
            # فقط اسناد field-filtered که keyword هم دارند
            keyword_docs = set()
            for token in tokens:
                keyword_docs.update(self.index.get_doc_ids_for_term(token))
            candidate_docs = candidate_docs.intersection(keyword_docs)

        if not candidate_docs:
            return []

        scored = self._score_documents(candidate_docs, tokens if tokens else [])

        # اگر توکنی نبود، امتیاز یکسان بده
        if not tokens:
            results = []
            for doc_id in sorted(candidate_docs):
                meta = self.index.get_doc_info(doc_id)
                results.append(QueryResult(doc_id=doc_id, score=1.0, metadata=meta))
            return results[:top_k]

        return self._top_k(scored, top_k)

    # ============================================================
    # ۴) جستجوی هوشمند (تشخیص نوع خودکار)
    # ============================================================

    def search(self, query: str, top_k: int = 10) -> list[QueryResult]:
        """
        تابع اصلی جستجو.
        نوع پرس‌وجو را تشخیص می‌دهد و مناسب‌ترین روش را اجرا می‌کند.

        - اگر دارای AND/OR/NOT بود -> Boolean
        - اگر دارای field:value بود -> Fielded
        - وگرنه -> Keyword + TF-IDF
        """
        query = query.strip()

        # تشخیص نوع
        has_boolean = bool(re.search(r"\b(AND|OR|NOT)\b", query, re.IGNORECASE))
        has_field = bool(re.search(r"\w+:[\"']?[^\s]", query))

        if has_field:
            return self.fielded_search(query, top_k)
        elif has_boolean:
            results = self.boolean_search(query)
            return results[:top_k]
        else:
            return self.keyword_search(query, top_k)

    # ============================================================
    # محاسبه TF-IDF
    # ============================================================

    def _compute_tf(self, tf_raw: int, doc_length: int) -> float:
        """
        TF نرمال‌شده.

        فرمول: log(1 + tf)
        چرا log؟ چون اگر messi 5 بار در سندی باشد،
        نه 5 برابر بلکه کمی بیشتر از 1 بار اهمیت دارد.
        """
        if tf_raw == 0:
            return 0.0
        return 1 + math.log10(tf_raw)

    def _compute_idf(self, df: int) -> float:
        """
        IDF: Inverse Document Frequency.

        فرمول: log(N / df)

        N  = تعداد کل اسناد
        df = تعداد اسنادی که این term دارند

        مثال:
        - "goals" در ۶۰ از ۶۴ سند: idf = log(64/60) ≈ 0.028 (کم)
        - "messi" در ۳ از ۶۴ سند:  idf = log(64/3)  ≈ 1.33  (زیاد)

        چرا این term های نادر مهم‌ترند؟
        چون اگر کاربر "messi" جستجو کند، می‌خواهد بازی‌های مسی را ببیند،
        نه همه بازی‌هایی که کلمه "goals" دارند.
        """
        N = self.index.doc_count
        if df == 0 or N == 0:
            return 0.0
        return math.log10(N / df)

    def _score_documents(self, doc_ids: set[int], tokens: list[str]) -> list[QueryResult]:
        """
        TF-IDF امتیاز هر سند را محاسبه می‌کند.

        امتیاز نهایی = مجموع (TF × IDF) برای همه کلمات پرس‌وجو
        """
        scores = defaultdict(float)

        for token in tokens:
            pl = self.index.lookup(token)
            if pl is None:
                continue

            idf = self._compute_idf(pl.df)

            # فقط اسناد candidate
            for posting in pl.postings:
                if posting.doc_id not in doc_ids:
                    continue

                doc_len = self.index.doc_lengths.get(posting.doc_id, 1)
                tf = self._compute_tf(posting.tf, doc_len)
                scores[posting.doc_id] += tf * idf

        # اگر اسنادی candidate بودند اما توکن نداشتند، امتیاز ۰
        for doc_id in doc_ids:
            if doc_id not in scores:
                scores[doc_id] = 0.0

        # ساخت QueryResult
        results = []
        for doc_id, score in scores.items():
            meta = self.index.get_doc_info(doc_id)
            results.append(QueryResult(doc_id=doc_id, score=score, metadata=meta))

        return results

    def _top_k(self, results: list[QueryResult], top_k: int) -> list[QueryResult]:
        """نتایج را بر اساس امتیاز مرتب و top_k تای اول را برمی‌گرداند."""
        results.sort(key=lambda r: -r.score)
        return results[:top_k]


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


# ==========================================
# اجرای مستقیم برای تست
# ==========================================
if __name__ == "__main__":
    import json, os, sys

    sys.path.insert(0, ".")
    from document_builder import build_all_documents
    from preprocessor import preprocess_documents

    # ---- بارگذاری داده ----
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

    # ---- تست با پرس‌وجوهای مختلف ----
    test_queries = [
        "messi",
        "mbappe",
        "penalty",
        "final",
        "messi final",
        "mbappe AND goal",
        "penalty AND quarter",
        "argentina NOT france",
        "team:Argentina",
        "team:Argentina stage:Final",
    ]

    for q in test_queries:
        results = qp.search(q, top_k=5)
        print_results(results, q)