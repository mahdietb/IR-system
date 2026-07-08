"""
preprocessor.py
================
بخش ۲ پروژه: پیش‌پردازش متن

این ماژول متن خام را تمیز و نرمال می‌کند تا
برای ایندکس‌گذاری آماده شود.

مراحل به ترتیب:
1. تبدیل به حروف کوچک (lowercase)
2. حذف کاراکترهای HTML خاص (مثل &rsquor;)
3. نرمال‌سازی نام‌های خاص (accent normalization)
4. حذف علائم نگارشی غیرضروری
5. Tokenize کردن (شکستن به کلمات)
6. حذف stop words (کلمات بی‌معنی)
"""

import re
import unicodedata
import json

# ---- Stop words ----
# کلماتی که در تقریباً همه متن‌ها هستند و
# هیچ اطلاعات جستجویی نمی‌دهند
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "up", "about",
    "into", "through", "during", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might",
    "vs", "v", "via", "per", "as", "it", "its", "this", "that",
    "these", "those", "i", "we", "you", "he", "she", "they",
    "their", "our", "your", "his", "her", "then", "than",
    "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "each", "more", "most", "other", "some", "such",
    "only", "own", "same", "than", "too", "very", "just", "can",
    "also", "s", "d", "m", "ll", "re", "ve",
    # کلمات مربوط به فوتبال که خیلی رایج‌اند و معنی خاصی ندارند
    "match", "game", "played", "team", "player", "won", "lost",
    "score", "result", "date", "stadium", "stage", "referee",
}

# ---- نرمال‌سازی کاراکترهای خاص HTML ----
HTML_ENTITIES = {
    "&rsquor;": "'",
    "&lsquor;": "'",
    "&rdquor;": '"',
    "&ldquor;": '"',
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&nbsp;": " ",
    "&apos;": "'",
    "&quot;": '"',
    "&#39;": "'",
    "&#34;": '"',
}

# ---- نرمال‌سازی نام‌های خاص ----
# برخی نام‌های بازیکنان دارای حرف‌های لهجه‌دار هستند
# برای اینکه جستجوی "Mbappe" و "Mbappé" هر دو کار کنند
# این فرهنگ نام‌های رایج جام جهانی ۲۰۲۲ را پوشش می‌دهد
NAME_NORMALIZATIONS = {
    "mbappé": "mbappe",
    "griezmann": "griezmann",
    "müller": "muller",
    "schürrle": "schurrle",
    "özil": "ozil",
    "en-nesyri": "en nesyri",
    "hakimi": "hakimi",
    "modrić": "modric",
    "perišić": "perisic",
    "gvardiol": "gvardiol",
    "stanković": "stankovic",
    "szymański": "szymanski",
    "świderski": "swiderski",
    "mitrović": "mitrovic",
    "vlahović": "vlahovic",
    "živković": "zivkovic",
    "júnior": "junior",
    "thiago": "thiago",
    "vinicius": "vinicius",
    "rodrygo": "rodrygo",
    "joão": "joao",
    "gonçalo": "goncalo",
    "félipe": "felipe",
    "martínez": "martinez",
    "garcía": "garcia",
    "álvarez": "alvarez",
    "ángel": "angel",
    "di maría": "di maria",
}


def remove_html_entities(text: str) -> str:
    """
    کاراکترهای خاص HTML را حذف یا جایگزین می‌کند.

    مثال: "it&rsquor;s" -> "it's"
    """
    for entity, replacement in HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    # حذف هر entity باقیمانده
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    return text


def normalize_unicode(text: str) -> str:
    """
    حروف لهجه‌دار را به معادل ASCII تبدیل می‌کند.

    مثال: "é" -> "e", "ü" -> "u", "ó" -> "o"

    چرا؟ تا "Mbappe" و "Mbappé" هر دو پیدا شوند.
    """
    # NFD: کاراکتر پایه + diacritic را جدا می‌کند
    # مثال: é = e + ́ (combining acute accent)
    normalized = unicodedata.normalize("NFD", text)
    # فقط کاراکترهای ASCII را نگه می‌داریم
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text


def lowercase(text: str) -> str:
    """تبدیل همه حروف به کوچک."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    علائم نگارشی را با فاصله جایگزین می‌کند.

    استثناء: خط تیره بین کلمات را نگه می‌داریم
    تا "extra-time" به "extra time" تبدیل شود نه "extratime"
    """
    # خط تیره و اپستروف بین حروف را با فاصله جایگزین می‌کنیم
    text = re.sub(r"[-']", " ", text)
    # بقیه علائم نگارشی را حذف می‌کنیم
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    """
    متن را به لیستی از توکن (کلمه) تبدیل می‌کند.

    مثال: "goals by messi" -> ["goals", "by", "messi"]
    """
    # شکستن روی فاصله‌ها و کاراکترهای غیرمفید
    tokens = re.findall(r"\b[a-z][a-z0-9]*\b", text)
    return tokens


def remove_stop_words(tokens: list[str]) -> list[str]:
    """
    کلمات بی‌معنی را از لیست توکن‌ها حذف می‌کند.

    مثال: ["goals", "by", "the", "messi"] -> ["goals", "messi"]
    """
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def apply_name_normalizations(tokens: list[str]) -> list[str]:
    """
    نام‌های خاص را نرمال می‌کند.

    مثال: ["mbappé", "goals"] -> ["mbappe", "goals"]
    """
    result = []
    i = 0
    while i < len(tokens):
        # سعی کن دو توکن پشت هم رو بررسی کن (مثل "di maria")
        if i < len(tokens) - 1:
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in NAME_NORMALIZATIONS:
                result.extend(NAME_NORMALIZATIONS[bigram].split())
                i += 2
                continue

        token = tokens[i]
        if token in NAME_NORMALIZATIONS:
            result.extend(NAME_NORMALIZATIONS[token].split())
        else:
            result.append(token)
        i += 1

    return result


def preprocess(text: str, keep_stop_words: bool = False) -> list[str]:
    """
    تابع اصلی پیش‌پردازش:
    همه مراحل را به ترتیب اجرا می‌کند.

    ورودی:  "Goals by Kylian Mbappé 80' (P), Angel Di Maria 36'"
    خروجی: ["goals", "kylian", "mbappe", "80", "angel", "di", "maria", "36"]

    پارامتر keep_stop_words:
    - False (پیش‌فرض): برای ایندکس‌گذاری
    - True: برای نمایش به کاربر
    """
    if not text:
        return []

    # مرحله ۱: حذف کاراکترهای HTML
    text = remove_html_entities(text)

    # مرحله ۲: نرمال‌سازی Unicode (é -> e)
    text = normalize_unicode(text)

    # مرحله ۳: تبدیل به lowercase
    text = lowercase(text)

    # مرحله ۴: اعمال نرمال‌سازی نام‌ها (قبل از حذف علائم)
    # نکته: اینجا باید روی متن کامل کار کنیم
    for accented, plain in NAME_NORMALIZATIONS.items():
        text = text.replace(accented, plain)

    # مرحله ۵: حذف علائم نگارشی
    text = remove_punctuation(text)

    # مرحله ۶: توکنایز
    tokens = tokenize(text)

    # مرحله ۷: حذف stop words (اختیاری)
    if not keep_stop_words:
        tokens = remove_stop_words(tokens)

    return tokens


def preprocess_query(query: str) -> list[str]:
    """
    پیش‌پردازش پرس‌وجو.
    مشابه preprocess اما stop words را نگه می‌داریم
    تا عملگرهای Boolean درست پارس شوند.
    سپس عملگرها (AND, OR, NOT) را جدا می‌کنیم.
    """
    # ابتدا بررسی کن که آیا پرس‌وجوی Boolean است
    upper_query = query.upper()
    has_boolean = " AND " in upper_query or " OR " in upper_query or " NOT " in upper_query

    if has_boolean:
        # توکن‌های Boolean را حفظ می‌کنیم
        return preprocess(query, keep_stop_words=True)
    else:
        return preprocess(query, keep_stop_words=False)


def preprocess_documents(documents: list[dict]) -> list[dict]:
    """
    همه اسناد را پیش‌پردازش می‌کند.
    توکن‌های پردازش‌شده را به هر سند اضافه می‌کند.
    """
    for doc in documents:
        # پیش‌پردازش متن کامل سند
        doc["tokens"] = preprocess(doc["text"])

        # پیش‌پردازش فیلدهای جداگانه (برای جستجوی فیلدی)
        doc["field_tokens"] = {
            "home_team": preprocess(doc["fields"].get("home_team", "")),
            "away_team": preprocess(doc["fields"].get("away_team", "")),
            "stage": preprocess(doc["fields"].get("stage", "")),
            "stadium": preprocess(doc["fields"].get("stadium", "")),
            "referee": preprocess(doc["fields"].get("referee", "")),
            "goal_scorers": preprocess(" ".join(doc["fields"].get("goal_scorers", []))),
        }

    print(f"[✓] {len(documents)} سند پیش‌پردازش شد.")
    return documents



# اجرای مستقیم برای تست

if __name__ == "__main__":
    print("=== تست پیش‌پردازش ===\n")


    test_cases = [
        "Goals by Kylian Mbappé 80' (P), Angel Di Maria 36'",
        "Argentina vs France, Final, Lusail Stadium",
        "Referee: Szymon Marciniak. Yellow cards: En-Nesyri",
        "Match decided on penalties &rsquor; extra time goal",
        "Captain Luka Modrić, Croatia vs Brazil",
    ]

    print("--- preprocess() ---")
    for text in test_cases:
        tokens = preprocess(text)
        print(f"ورودی : {text[:60]}")
        print(f"خروجی : {tokens}")
        print()

    print("--- preprocess_query() ---")
    query_tests = [
        "messi final",
        "mbappe AND goal",
        "penalty AND quarter-finals",
        "yellow cards argentina",
    ]
    for q in query_tests:
        tokens = preprocess_query(q)
        print(f"پرس‌وجو: {q}")
        print(f"توکن‌ها: {tokens}")
        print()


    doc_path = "data/documents.json"
    import os

    if os.path.exists(doc_path):
        with open(doc_path, encoding="utf-8") as f:
            docs = json.load(f)

        sample = docs[0]
        tokens = preprocess(sample["text"])
        print(f"--- نمونه سند ---")
        print(f"متن   : {sample['text'][:150]}...")
        print(f"توکن‌ها: {tokens[:20]}...")