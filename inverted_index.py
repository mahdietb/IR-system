

import json
import os
import pickle
from collections import defaultdict


class Posting:


    def __init__(self, doc_id: int, tf: int = 0, positions: list = None):
        self.doc_id = doc_id
        self.tf = tf
        self.positions = positions or []

    def __repr__(self):
        return f"Posting(doc={self.doc_id}, tf={self.tf}, pos={self.positions[:3]})"

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "tf": self.tf,
            "positions": self.positions
        }


class PostingList:


    def __init__(self):
        self.df = 0
        self.postings = []  # list of Posting, sorted by doc_id

    def add(self, posting: Posting):
        self.postings.append(posting)
        self.df += 1

    def get_doc_ids(self) -> list[int]:

        return [p.doc_id for p in self.postings]

    def get_posting(self, doc_id: int) -> Posting | None:

        lo, hi = 0, len(self.postings) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.postings[mid].doc_id == doc_id:
                return self.postings[mid]
            elif self.postings[mid].doc_id < doc_id:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def __repr__(self):
        return f"PostingList(df={self.df}, docs={self.get_doc_ids()[:5]})"

    def to_dict(self):
        return {
            "df": self.df,
            "postings": [p.to_dict() for p in self.postings]
        }


class InvertedIndex:


    def __init__(self):
        self.index = {}  # term -> PostingList
        self.field_indexes = {}  # field_name -> {term -> PostingList}
        self.doc_count = 0
        self.doc_lengths = {}  # doc_id -> تعداد توکن‌ها
        self.doc_metadata = {}  # doc_id -> اطلاعات خلاصه سند


    # ساخت ایندکس


    def build(self, documents: list[dict]):

        print(f"[*] ایندکس‌گذاری {len(documents)} سند...")

        self.doc_count = len(documents)

        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = doc.get("tokens", [])


            self.doc_lengths[doc_id] = len(tokens)


            fields = doc.get("fields", {})
            self.doc_metadata[doc_id] = {
                "home_team": fields.get("home_team", ""),
                "away_team": fields.get("away_team", ""),
                "stage": fields.get("stage", ""),
                "score": fields.get("score", ""),
                "date": fields.get("date", ""),
            }


            term_positions = defaultdict(list)
            for pos, token in enumerate(tokens):
                term_positions[token].append(pos)

            for term, positions in term_positions.items():
                if term not in self.index:
                    self.index[term] = PostingList()

                posting = Posting(
                    doc_id=doc_id,
                    tf=len(positions),
                    positions=positions
                )
                self.index[term].add(posting)

            field_tokens = doc.get("field_tokens", {})
            for field_name, f_tokens in field_tokens.items():
                if field_name not in self.field_indexes:
                    self.field_indexes[field_name] = {}

                f_term_positions = defaultdict(list)
                for pos, token in enumerate(f_tokens):
                    f_term_positions[token].append(pos)

                for term, positions in f_term_positions.items():
                    if term not in self.field_indexes[field_name]:
                        self.field_indexes[field_name][term] = PostingList()

                    posting = Posting(doc_id=doc_id, tf=len(positions), positions=positions)
                    self.field_indexes[field_name][term].add(posting)

        print(f"[✓] ایندکس ساخته شد: {len(self.index)} term یکتا")
        print(f"    تعداد اسناد: {self.doc_count}")


        top_terms = sorted(
            [(t, pl.df) for t, pl in self.index.items()],
            key=lambda x: -x[1]
        )[:10]
        print(f"    پرتکرارترین term‌ها: {top_terms}")


    # جستجو در ایندکس


    def lookup(self, term: str) -> PostingList | None:

        return self.index.get(term, None)

    def lookup_field(self, field: str, term: str) -> PostingList | None:

        field_idx = self.field_indexes.get(field, {})
        return field_idx.get(term, None)

    def get_doc_ids_for_term(self, term: str) -> list[int]:

        pl = self.lookup(term)
        if pl is None:
            return []
        return pl.get_doc_ids()

    def get_all_doc_ids(self) -> set[int]:

        return set(self.doc_metadata.keys())

    def get_doc_info(self, doc_id: int) -> dict:

        return self.doc_metadata.get(doc_id, {})


    # ذخیره و بارگذاری


    def save(self, path: str):

        data = {
            "doc_count": self.doc_count,
            "doc_lengths": self.doc_lengths,
            "doc_metadata": self.doc_metadata,
            "index": {
                term: pl.to_dict()
                for term, pl in self.index.items()
            },
            "field_indexes": {
                field: {
                    term: pl.to_dict()
                    for term, pl in field_idx.items()
                }
                for field, field_idx in self.field_indexes.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[✓] ایندکس در {path} ذخیره شد.")

    def load(self, path: str):

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.doc_count = data["doc_count"]
        self.doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        self.doc_metadata = {int(k): v for k, v in data["doc_metadata"].items()}


        self.index = {}
        for term, pl_data in data["index"].items():
            pl = PostingList()
            pl.df = pl_data["df"]
            for p_data in pl_data["postings"]:
                pl.postings.append(Posting(
                    doc_id=p_data["doc_id"],
                    tf=p_data["tf"],
                    positions=p_data["positions"]
                ))
            self.index[term] = pl


        self.field_indexes = {}
        for field, field_data in data.get("field_indexes", {}).items():
            self.field_indexes[field] = {}
            for term, pl_data in field_data.items():
                pl = PostingList()
                pl.df = pl_data["df"]
                for p_data in pl_data["postings"]:
                    pl.postings.append(Posting(
                        doc_id=p_data["doc_id"],
                        tf=p_data["tf"],
                        positions=p_data["positions"]
                    ))
                self.field_indexes[field][term] = pl

        print(f"[✓] ایندکس از {path} بارگذاری شد: {len(self.index)} term")


    # اطلاعات آماری


    def stats(self):

        print(f"\n=== آمار Inverted Index ===")
        print(f"تعداد اسناد       : {self.doc_count}")
        print(f"تعداد term یکتا   : {len(self.index)}")

        total_postings = sum(pl.df for pl in self.index.values())
        print(f"تعداد کل posting  : {total_postings}")

        avg_postings = total_postings / len(self.index) if self.index else 0
        print(f"میانگین df per term: {avg_postings:.2f}")

        # توزیع df
        df_one = sum(1 for pl in self.index.values() if pl.df == 1)
        df_all = sum(1 for pl in self.index.values() if pl.df == self.doc_count)
        print(f"term هایی که فقط در 1 سند: {df_one}")
        print(f"term هایی که در همه اسناد: {df_all}")

        # پرتکرارترین term‌ها
        top = sorted(self.index.items(), key=lambda x: -x[1].df)[:5]
        print(f"\nپرتکرارترین term‌ها:")
        for term, pl in top:
            print(f"  '{term}': df={pl.df}")



# اجرای مستقیم برای تست

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from document_builder import build_all_documents, load_documents
    from preprocessor import preprocess_documents


    if os.path.exists("data/documents.json"):
        with open("data/documents.json", encoding="utf-8") as f:
            import json

            docs = json.load(f)
    else:
        print("[!] فایل documents.json پیدا نشد. ابتدا document_builder.py را اجرا کنید.")
        exit(1)


    docs = preprocess_documents(docs)

    idx = InvertedIndex()
    idx.build(docs)
    idx.stats()

    print("\n=== تست جستجو ===")

    test_terms = ["messi", "mbappe", "penalty", "final", "lusail"]
    for term in test_terms:
        pl = idx.lookup(term)
        if pl:
            print(f"'{term}': df={pl.df}, docs={pl.get_doc_ids()}")
        else:
            print(f"'{term}': پیدا نشد")


    print("\n--- جستجوی فیلدی ---")
    pl = idx.lookup_field("home_team", "argentina")
    if pl:
        print(f"home_team='argentina': {pl.get_doc_ids()}")

    os.makedirs("data", exist_ok=True)
    idx.save("data/index.json")