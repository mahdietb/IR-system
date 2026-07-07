"""
document_builder.py  —  Part 1
================================
Reads the World Cup 1930-2022 CSV and builds one text document per match.

HOW TO RUN:
    python document_builder.py
    (uses the default CSV path hardcoded below)

HOW TO VERIFY (checklist printed at the end):
    [1] 964 documents built
    [2] doc_id=1 is Argentina vs France Final, has_penalties=True
    [3] Goal scorers parsed correctly (Messi, Di María, Mbappé visible)
    [4] Yellow cards parsed correctly
    [5] Red cards parsed correctly
    [6] data/documents.json exists and is non-empty
"""

import csv
import re
import json
import os

CSV_PATH = r"C:\Users\ASUS\Downloads\matches_1930_2022.csv"


# ═══════════════════════════════════════════════════════════════
#  LOW-LEVEL PARSERS
# ═══════════════════════════════════════════════════════════════

def parse_event_list(cell: str) -> list[str]:
    """
    The goal/card/sub columns store data as a Python-like list string:
        "['36&rsquor;|2:0|Ángel Di María|Assist:|Mac Allister', '108&rsquor;|3:2|Lionel Messi']"

    We CANNOT use ast.literal_eval because after replacing &rsquor; → '
    the apostrophes break Python string parsing.

    Strategy: strip outer brackets, then split on  ', '  boundary,
    then clean each entry individually.
    """
    raw = str(cell).strip()
    if raw.lower() in ("nan", "none", ""):
        return []

    # strip surrounding [ ]
    if raw.startswith("["):
        raw = raw[1:]
    if raw.endswith("]"):
        raw = raw[:-1]

    # split entries — they are separated by  ', '
    entries = re.split(r"',\s*'", raw)

    result = []
    for e in entries:
        e = e.strip(" '")
        # now safe to replace HTML entities
        e = e.replace("&rsquor;", "'").replace("&lsquor;", "'").replace("&amp;", "&")
        if e:
            result.append(e)
    return result


def player_from_event(event: str) -> str:
    """
    Each event string:  "36'|2:0|Ángel Di María|Assist:|Mac Allister"
    Player name is always at pipe-index 2.
    """
    parts = event.split("|")
    return parts[2].strip() if len(parts) >= 3 else ""


def players_from_event_list(cell: str) -> list[str]:
    """Parse an event-list column and return player names."""
    return [n for n in (player_from_event(e) for e in parse_event_list(cell)) if n]


def players_from_plain(cell: str) -> list[str]:
    """
    Red-card and yellow-red-card columns use a simpler format:
        "Paulo Bento · 90+11"   or   "Player1 · 45|Player2 · 78"
    Returns list of names.
    """
    raw = str(cell).strip()
    if raw.lower() in ("nan", "none", ""):
        return []
    players = []
    for part in raw.split("|"):
        name = re.sub(r"\s*·\s*[\d+]+.*$", "", part).strip()
        if name:
            players.append(name)
    return players


def players_from_penalty_goal(cell: str) -> list[str]:
    """
    home_penalty_goal column:  "Lionel Messi (P) · 23|Kylian Mbappé (P) · 80"
    """
    raw = str(cell).strip()
    if raw.lower() in ("nan", "none", ""):
        return []
    players = []
    for part in raw.split("|"):
        name = re.sub(r"\s*\(P\)", "", part)
        name = re.sub(r"\s*·\s*[\d+]+.*$", "", name).strip()
        if name:
            players.append(name)
    return players


def players_from_shootout(cell: str) -> list[str]:
    """
    Penalty-shootout columns use pipe-index 2, but no apostrophe:
        "2|1:1|Lionel Messi"
    """
    raw = str(cell).strip()
    if raw.lower() in ("nan", "none", ""):
        return []
    # try event-list first
    entries = parse_event_list(raw)
    if entries:
        return [n for n in (player_from_event(e) for e in entries) if n]
    # fallback: direct pipe split
    players = []
    for part in raw.split(","):
        p = part.strip().strip("[]'\"")
        cols = p.split("|")
        if len(cols) >= 3:
            players.append(cols[2].strip())
    return players


def get_referee(officials: str, referee: str) -> str:
    r = str(referee).strip()
    if r and r.lower() not in ("nan", "none", ""):
        return r
    m = re.search(r"([^·]+)\(Referee\)", str(officials))
    return m.group(1).strip() if m else ""


# ═══════════════════════════════════════════════════════════════
#  DOCUMENT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_document(row: dict, doc_id: int) -> dict:
    """Convert one CSV row into a searchable document."""

    # ── basic fields ─────────────────────────────────────────────────────
    home_team = str(row.get("home_team", "")).strip()
    away_team = str(row.get("away_team", "")).strip()
    round_ = str(row.get("Round", "")).strip()
    date = str(row.get("Date", "")).strip()
    venue = str(row.get("Venue", "")).strip()
    score = str(row.get("Score", "")).strip()
    notes = str(row.get("Notes", "")).strip()
    year = str(row.get("Year", "")).strip()
    host = str(row.get("Host", "")).strip()
    home_score = str(row.get("home_score", "")).strip()
    away_score = str(row.get("away_score", "")).strip()
    home_manager = str(row.get("home_manager", "")).strip()
    away_manager = str(row.get("away_manager", "")).strip()
    home_captain = str(row.get("home_captain", "")).strip()
    away_captain = str(row.get("away_captain", "")).strip()
    referee = get_referee(row.get("Officials", ""), row.get("Referee", ""))

    # ── events ────────────────────────────────────────────────────────────
    # goal scorers
    home_scorers = players_from_event_list(row.get("home_goal_long", ""))
    if not home_scorers:
        # fallback to simple column "Name · minute|Name · minute"
        home_scorers = players_from_plain(row.get("home_goal", ""))
    away_scorers = players_from_event_list(row.get("away_goal_long", ""))
    if not away_scorers:
        away_scorers = players_from_plain(row.get("away_goal", ""))
    all_scorers = home_scorers + away_scorers

    # in-match penalty goals (spot kicks during normal/extra time)
    pen_scorers = (players_from_penalty_goal(row.get("home_penalty_goal", "")) +
                   players_from_penalty_goal(row.get("away_penalty_goal", "")))

    # own goals
    own_goals = (players_from_plain(row.get("home_own_goal", "")) +
                 players_from_plain(row.get("away_own_goal", "")))

    # yellow cards
    yellow = (players_from_event_list(row.get("home_yellow_card_long", "")) +
              players_from_event_list(row.get("away_yellow_card_long", "")))

    # red cards (plain format: "Name · minute")
    red = (players_from_plain(row.get("home_red_card", "")) +
           players_from_plain(row.get("away_red_card", "")))

    # yellow-red (second yellow)
    yellow_red = (players_from_plain(row.get("home_yellow_red_card", "")) +
                  players_from_plain(row.get("away_yellow_red_card", "")))

    # substitutes (players coming ON)
    subs = (players_from_event_list(row.get("home_substitute_in_long", "")) +
            players_from_event_list(row.get("away_substitute_in_long", "")))

    # penalty shootout
    so_goals = (players_from_shootout(row.get("home_penalty_shootout_goal_long", "")) +
                players_from_shootout(row.get("away_penalty_shootout_goal_long", "")))
    so_miss = (players_from_shootout(row.get("home_penalty_shootout_miss_long", "")) +
               players_from_shootout(row.get("away_penalty_shootout_miss_long", "")))

    # penalty misses during normal/extra time
    pen_miss = (players_from_event_list(row.get("home_penalty_miss_long", "")) +
                players_from_event_list(row.get("away_penalty_miss_long", "")))

    # ── flags ─────────────────────────────────────────────────────────────
    has_penalties = bool(so_goals or so_miss) or "penalty kicks" in notes.lower()
    has_extra_time = ("extra time" in notes.lower() or
                      re.search(r"\b(9[1-9]|1[0-2]\d)\b",
                                str(row.get("home_goal_long", "")) +
                                str(row.get("away_goal_long", ""))) is not None)
    has_own_goal = bool(own_goals)
    has_red_card = bool(red or yellow_red)
    has_penalty_miss = bool(pen_miss or so_miss)

    # ── build full text ───────────────────────────────────────────────────
    def nv(s):
        """Return s if not nan/empty, else ''."""
        return s if s and s.lower() not in ("nan", "none") else ""

    parts = []
    parts.append(f"{home_team} vs {away_team}")
    parts.append(f"Year {year}")
    if nv(host):   parts.append(f"Host {host}")
    if nv(round_): parts.append(f"Round {round_} Stage {round_}")
    if nv(venue):  parts.append(f"Venue {venue}")
    if nv(score):  parts.append(f"Score {score} {home_score} {away_score}")
    if nv(referee): parts.append(f"Referee {referee}")
    if nv(notes):  parts.append(notes)

    if nv(home_manager): parts.append(f"Manager {home_manager} Coach {home_manager}")
    if nv(away_manager): parts.append(f"Manager {away_manager} Coach {away_manager}")
    if nv(home_captain): parts.append(f"Captain {home_captain}")
    if nv(away_captain): parts.append(f"Captain {away_captain}")

    if all_scorers:   parts.append("Goals by " + " ".join(all_scorers))
    if pen_scorers:   parts.append("Penalty goal by " + " ".join(pen_scorers))
    if own_goals:     parts.append("Own goal by " + " ".join(own_goals) + " own goal")
    if yellow:        parts.append("Yellow card " + " ".join(yellow))
    if red:           parts.append("Red card sent off " + " ".join(red))
    if yellow_red:    parts.append("Second yellow red card " + " ".join(yellow_red))
    if subs:          parts.append("Substitution " + " ".join(subs))
    if so_goals:      parts.append("Penalty shootout goals " + " ".join(so_goals))
    if so_miss:       parts.append("Penalty shootout miss " + " ".join(so_miss))
    if pen_miss:      parts.append("Penalty miss " + " ".join(pen_miss))

    if has_penalties:    parts.append("penalty shootout penalties decided on penalties")
    if has_extra_time:   parts.append("extra time aet overtime")
    if has_own_goal:     parts.append("own goal")
    if has_red_card:     parts.append("red card dismissal")
    if has_penalty_miss: parts.append("missed penalty penalty miss")

    text = ". ".join(filter(None, parts))

    # ── assemble ──────────────────────────────────────────────────────────
    return {
        "doc_id": doc_id,
        "text": text,
        "fields": {
            "home_team": home_team,
            "away_team": away_team,
            "stage": round_,
            "venue": venue,
            "score": score,
            "referee": referee,
            "date": date,
            "year": year,
            "host": host,
            "home_manager": home_manager,
            "away_manager": away_manager,
            "home_captain": home_captain,
            "away_captain": away_captain,
            "goal_scorers": all_scorers,
            "yellow_cards": yellow,
            "red_cards": red + yellow_red,
            "substitutes": subs,
            "has_penalties": has_penalties,
            "has_extra_time": has_extra_time,
            "has_own_goal": has_own_goal,
            "has_red_card": has_red_card,
            "has_penalty_miss": has_penalty_miss,
        },
        "raw": {k: str(v) for k, v in row.items()},
    }


def build_all_documents(csv_path: str) -> list[dict]:
    docs = []
    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            docs.append(build_document(row, doc_id=i + 1))
    print(f"[OK] Built {len(docs)} documents")
    return docs


def save_documents(docs: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved → {path}")


def load_documents(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    print(f"[OK] Loaded {len(docs)} documents from {path}")
    return docs


# ═══════════════════════════════════════════════════════════════
#  MAIN — run to build + verify
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    docs = build_all_documents(CSV_PATH)
    save_documents(docs, "data/documents.json")

    print()
    print("=" * 60)
    print("VERIFICATION CHECKLIST  (compare expected vs actual)")
    print("=" * 60)

    # [1] total count
    print(f"\n[1] Total documents : {len(docs)}  | expected: 964")

    # [2] spot-check doc 1
    d = docs[0]
    f = d["fields"]
    print(f"\n[2] doc_id=1 spot-check (Argentina vs France Final 2022)")
    print(f"    home_team     : {f['home_team']:<20} expected: Argentina")
    print(f"    away_team     : {f['away_team']:<20} expected: France")
    print(f"    stage         : {f['stage']:<20} expected: Final")
    print(f"    referee       : {f['referee']:<30} expected: Szymon Marciniak")
    print(f"    has_penalties : {f['has_penalties']:<10} expected: True")
    print(f"    has_extra_time: {f['has_extra_time']:<10} expected: True")
    print(f"    has_own_goal  : {f['has_own_goal']:<10} expected: False")

    # [3] goal scorers
    print(f"\n[3] Goal scorers doc_id=1:")
    print(f"    {f['goal_scorers']}")
    print(f"    expected: Ángel Di María, Lionel Messi, Kylian Mbappé visible")

    # [4] yellow cards
    print(f"\n[4] Yellow cards doc_id=1:")
    print(f"    {f['yellow_cards'][:4]}")
    print(f"    expected: Enzo Fernández, Marcos Acuña, Leandro Paredes, ...")

    # [5] red cards — find a match with one
    red_doc = next((d for d in docs if d["fields"]["red_cards"]), None)
    if red_doc:
        rf = red_doc["fields"]
        print(f"\n[5] Red card check  (doc_id={red_doc['doc_id']}):")
        print(f"    {rf['home_team']} vs {rf['away_team']}, {rf['stage']}")
        print(f"    red_cards: {rf['red_cards']}")
    else:
        print(f"\n[5] Red cards: NONE FOUND — check parsers!")

    # [6] coverage stats
    has_goals = sum(1 for d in docs if d["fields"]["goal_scorers"])
    has_pen = sum(1 for d in docs if d["fields"]["has_penalties"])
    has_et = sum(1 for d in docs if d["fields"]["has_extra_time"])
    has_red = sum(1 for d in docs if d["fields"]["has_red_card"])
    has_subs = sum(1 for d in docs if d["fields"]["substitutes"])
    print(f"\n[6] Coverage across all {len(docs)} documents:")
    print(f"    goal scorers parsed : {has_goals}")
    print(f"    penalty shootouts   : {has_pen}")
    print(f"    extra time matches  : {has_et}")
    print(f"    red card matches    : {has_red}")
    print(f"    substitutions       : {has_subs}")
    print(f"    expected roughly    : 866 / 35 / 70+ / 50+ / 400+")

    print("\n[DONE] Part 1 complete if all checks pass.")
