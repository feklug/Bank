"""
simulate/is_there_a_pattern.py
===============================
Prüft ob eine kategorisierte Transaktion zu einem bekannten Muster
in Firestore patterns_db passt.

Match-Kriterien (alle drei müssen erfüllt sein):
  1. COUNTERPARTY  : IBAN exact  ODER  Gegenpartei ≥ 85% Ähnlichkeit
  2. BETRAG        : |tx_betrag - pattern_avg| ≤ max(15% von avg, 2×std, 10 EUR)
  3. DATUM         : |tx_datum - next_expected_date| ≤ Datums-Toleranz (nach Intervall)

Datums-Toleranzen:
  WEEKLY      : ± 3 Tage
  BIWEEKLY    : ± 4 Tage
  MONTHLY     : ± 7 Tage
  BIMONTHLY   : ± 8 Tage
  QUARTERLY   : ±12 Tage
  SEMIANNUAL  : ±15 Tage
  ANNUAL      : ±20 Tage
  CUSTOM      : ±10% des Intervalls (min 5, max 20 Tage)
  Kein Datum  : ±10 Tage (Fallback)

Wenn Match (JA):
  → Bestätigt das Pattern in patterns_db:
      - last_confirmed aktualisieren
      - next_expected_date neu berechnen
      - confirmation_count erhöhen
  → Gibt (True, pattern_doc_id) zurück

Wenn kein Match (NEIN):
  → Speichert TX in distributions_db (identisches Format wie organisational.py)
  → Gibt (False, None) zurück

Importierbar für simulate/pipeline.py:
  from is_there_a_pattern import check_pattern
  matched, doc_id = check_pattern(db, tx_flat)
"""

import hashlib
import os
import pathlib
import re
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# ─────────────────────────────────────────────
# KONSTANTEN & TOLERANZEN
# ─────────────────────────────────────────────

COLLECTION_PATTERNS      = "patterns_db"
COLLECTION_DISTRIBUTIONS = "distributions_db"

# Betrags-Toleranz
AMOUNT_TOLERANCE_PCT  = 0.15   # ±15% vom Durchschnitt
AMOUNT_TOLERANCE_MIN  = 10.0   # Mindest-Toleranz ±10 EUR (für kleine Fixbeträge)
AMOUNT_STD_MULTIPLIER = 2.0    # alternativ: 2× Standardabweichung des Patterns

# Datums-Toleranzen je Intervall (in Tagen)
DATE_TOLERANCE_BY_INTERVAL: dict[str, int] = {
    "WEEKLY":      3,
    "BIWEEKLY":    4,
    "MONTHLY":     7,
    "BIMONTHLY":   8,
    "QUARTERLY":  12,
    "SEMIANNUAL": 15,
    "ANNUAL":     20,
    "CUSTOM":      0,   # wird dynamisch berechnet: 10% des Intervalls
}
DATE_TOLERANCE_FALLBACK = 10   # wenn kein Intervall bekannt

# Gegenpartei-Ähnlichkeit
COUNTERPARTY_MIN_SIMILARITY = 0.85   # ≥85% Token-Übereinstimmung


# ─────────────────────────────────────────────
# DATUMS-HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    """Parst ISO-Datum oder ISO-Datetime robust."""
    if not s or s == "-":
        return None
    s = str(s).strip()
    try:
        if "T" in s or " " in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T"))
            return dt.date()
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _date_tolerance(interval_label: Optional[str], interval_days: Optional[int]) -> int:
    """Gibt die Datums-Toleranz in Tagen basierend auf dem Intervall zurück."""
    if not interval_label:
        return DATE_TOLERANCE_FALLBACK

    label = interval_label.split(" ")[0].upper()   # "MONTHLY (~30d)" → "MONTHLY"

    if label in DATE_TOLERANCE_BY_INTERVAL:
        tol = DATE_TOLERANCE_BY_INTERVAL[label]
        if label == "CUSTOM" and interval_days:
            tol = max(5, min(20, round(interval_days * 0.10)))
        return tol

    # Unbekanntes Intervall: 10% des Intervalls
    if interval_days:
        return max(5, min(20, round(interval_days * 0.10)))

    return DATE_TOLERANCE_FALLBACK


def _next_expected_date(
    confirmed_date: date,
    interval_label: Optional[str],
    interval_days: Optional[int],
    anchor_dom: Optional[int],
) -> str:
    """
    Berechnet das nächste erwartete Datum nach einer bestätigten TX.
    Nutzt denselben Algorithmus wie detect_patterns.py.
    """
    label = (interval_label or "").split(" ")[0].upper()

    if label == "MONTHLY" and anchor_dom:
        m = confirmed_date.month + 1 if confirmed_date.month < 12 else 1
        y = confirmed_date.year if confirmed_date.month < 12 else confirmed_date.year + 1
        day = min(anchor_dom, monthrange(y, m)[1])
        next_d = date(y, m, day)
    elif label == "QUARTERLY" and anchor_dom:
        m = confirmed_date.month + 3
        y = confirmed_date.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        day = min(anchor_dom, monthrange(y, m)[1])
        next_d = date(y, m, day)
    elif interval_days:
        next_d = confirmed_date + timedelta(days=interval_days)
    else:
        next_d = confirmed_date + timedelta(days=30)   # Fallback: +30 Tage

    return next_d.isoformat()


# ─────────────────────────────────────────────
# COUNTERPARTY-MATCHING
# ─────────────────────────────────────────────

def _iban_match(tx_iban: Optional[str], pattern_iban: Optional[str]) -> bool:
    """Exakter IBAN-Vergleich (normalisiert: keine Leerzeichen, Großbuchstaben)."""
    if not tx_iban or not pattern_iban:
        return False
    norm = lambda s: re.sub(r"\s+", "", str(s)).upper()
    return norm(tx_iban) == norm(pattern_iban)


def _counterparty_similarity(name_a: str, name_b: str) -> float:
    """
    Token-basierte Ähnlichkeit zwischen zwei Gegenpartei-Namen.
    Teilt in Wörter auf und berechnet Jaccard-Koeffizient.
    "Bürowelt GmbH" vs "Bürowelt GmbH Wien" → 2/3 = 0.67
    "Bürowelt GmbH" vs "Bürowelt GmbH"      → 3/3 = 1.0
    """
    if not name_a or not name_b:
        return 0.0
    norm  = lambda s: re.sub(r"[^\w]", " ", str(s).lower()).split()
    tok_a = set(norm(name_a))
    tok_b = set(norm(name_b))
    if not tok_a or not tok_b:
        return 0.0
    intersection = tok_a & tok_b
    union        = tok_a | tok_b
    return len(intersection) / len(union)


def _counterparty_match(tx: dict, pattern: dict) -> bool:
    """
    Prüft Counterparty-Übereinstimmung:
    1. IBAN-Match (exact) hat Vorrang
    2. Fallback: Gegenpartei-Ähnlichkeit ≥ COUNTERPARTY_MIN_SIMILARITY
    """
    tx_iban      = tx.get("iban") or ""
    pattern_iban = pattern.get("iban") or ""

    # Wenn beide IBANs vorhanden → nur IBAN vergleichen
    if tx_iban and pattern_iban and pattern_iban != "-":
        # Manche Patterns enthalten " WECHSEL"-Suffix → abschneiden
        clean_pattern_iban = pattern_iban.split()[0]
        return _iban_match(tx_iban, clean_pattern_iban)

    # Fallback: Gegenpartei-Name
    tx_name      = tx.get("gegenpartei", "") or ""
    pattern_name = pattern.get("gegenpartei", "") or ""

    sim = _counterparty_similarity(tx_name, pattern_name)
    return sim >= COUNTERPARTY_MIN_SIMILARITY


# ─────────────────────────────────────────────
# BETRAGS-MATCHING
# ─────────────────────────────────────────────

def _amount_match(tx_betrag: float, pattern: dict) -> bool:
    """
    Prüft ob der Betrag der TX im Toleranzbereich des Patterns liegt.
    Toleranz = max(15% von avg, 2×std, 10 EUR)
    """
    avg = abs(pattern.get("amount_avg", 0.0))
    std = abs(pattern.get("amount_std", 0.0))
    tx_abs = abs(tx_betrag)

    tolerance = max(
        avg * AMOUNT_TOLERANCE_PCT,
        AMOUNT_STD_MULTIPLIER * std,
        AMOUNT_TOLERANCE_MIN,
    )

    return abs(tx_abs - avg) <= tolerance


# ─────────────────────────────────────────────
# DATUMS-MATCHING
# ─────────────────────────────────────────────

def _date_match(tx_date: date, pattern: dict) -> bool:
    """
    Prüft ob das TX-Datum nahe am next_expected_date des Patterns liegt.
    Wenn kein next_expected_date gesetzt → immer True (Pattern wird bestätigt).
    """
    next_exp_str   = pattern.get("next_expected_date", "") or ""
    next_exp_date  = _parse_date(next_exp_str.split(" ")[0])   # "2025-02-01 08:30:00" → date

    if not next_exp_date:
        return True   # Kein Erwartungsdatum → Datum-Check überspringen

    interval_label = pattern.get("recurrence_interval", "") or pattern.get("recurrence_interval_label", "")
    interval_days  = pattern.get("recurrence_interval_days")

    tolerance = _date_tolerance(interval_label, interval_days)
    delta     = abs((tx_date - next_exp_date).days)

    return delta <= tolerance


# ─────────────────────────────────────────────
# FIRESTORE: PATTERN BESTÄTIGEN
# ─────────────────────────────────────────────

def _confirm_pattern(db, doc_ref, pattern: dict, tx: dict, tx_date: date):
    """
    Bestätigt ein gefundenes Pattern:
      - last_confirmed = jetzt
      - confirmation_count + 1
      - next_expected_date neu berechnen
      - letzte TX dem transactions-Array hinzufügen
    """
    interval_label = pattern.get("recurrence_interval", "")
    interval_days  = pattern.get("recurrence_interval_days")
    anchor_dom     = pattern.get("recurrence_day_of_month")

    new_next = _next_expected_date(tx_date, interval_label, interval_days, anchor_dom)
    now_iso  = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # TX dem Pattern hinzufügen
    existing_txs = pattern.get("transactions", [])
    new_tx_entry = {
        "datum":            tx.get("datum", ""),
        "betrag":           tx.get("betrag", 0),
        "verwendungszweck": tx.get("verwendungszweck", ""),
        "gegenpartei":      tx.get("gegenpartei", ""),
        "iban":             tx.get("iban"),
        "category_level1":  tx.get("category_level1", ""),
        "confidence":       tx.get("confidence", 0.0),
        "source":           "simulation",
    }

    db.document(doc_ref.path).update({
        "last_confirmed":      now_iso,
        "next_expected_date":  new_next,
        "confirmation_count":  pattern.get("confirmation_count", 0) + 1,
        "transaction_count":   len(existing_txs) + 1,
        "transactions":        existing_txs + [new_tx_entry],
    })


# ─────────────────────────────────────────────
# FIRESTORE: IN DISTRIBUTIONS_DB SPEICHERN
# ─────────────────────────────────────────────

def _make_doc_id(tx: dict) -> str:
    seed = f"{tx.get('datum','')}|{tx.get('betrag','')}|{tx.get('gegenpartei','')}"
    return hashlib.sha1(seed.encode()).hexdigest()[:20]


def _store_in_distributions(db, tx: dict):
    """
    Speichert eine TX ohne Pattern-Match in distributions_db.
    Format identisch mit organisational.py → _tx_to_standalone_doc().
    """
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    doc = {
        "datum":            tx.get("datum", ""),
        "betrag":           tx.get("betrag", 0),
        "verwendungszweck": tx.get("verwendungszweck", ""),
        "gegenpartei":      tx.get("gegenpartei", ""),
        "iban":             tx.get("iban"),
        "category_level1":  tx.get("category_level1", "SONDERKATEGORIEN"),
        "confidence":       tx.get("confidence", 0.0),
        "reasoning":        tx.get("reasoning", ""),
        "status":           tx.get("status", "BOOKED"),
        "tink_id":          tx.get("tink_id", ""),
        "stored_at":        now_iso,
        "source":           "simulation",
    }
    doc_id = _make_doc_id(tx)
    db.collection(COLLECTION_DISTRIBUTIONS).document(doc_id).set(doc)
    return doc_id


# ─────────────────────────────────────────────
# HAUPT-FUNKTION  (importierbar)
# ─────────────────────────────────────────────

def check_pattern(db, tx: dict) -> tuple[bool, Optional[str]]:
    """
    Prüft ob tx zu einem bekannten Pattern in patterns_db passt.

    Parameter:
        db  : Firestore-Client
        tx  : Flaches TX-Dict (Output von categorize_one())

    Rückgabe:
        (True,  pattern_doc_id)  → Match gefunden, Pattern bestätigt
        (False, None)            → kein Match, TX in distributions_db gespeichert
    """
    tx_date   = _parse_date(tx.get("datum", ""))
    tx_betrag = tx.get("betrag", 0.0)

    if not tx_date:
        # Datum nicht parsbar → direkt in distributions_db
        _store_in_distributions(db, tx)
        return False, None

    # Alle Patterns laden
    pattern_docs = list(db.collection(COLLECTION_PATTERNS).stream())

    best_match     = None
    best_match_ref = None
    best_score     = -1.0   # höhere Scores = besserer Match

    for doc in pattern_docs:
        pattern = doc.to_dict()

        # ── 1. Counterparty ───────────────────────────────────────
        if not _counterparty_match(tx, pattern):
            continue

        # ── 2. Betrag ─────────────────────────────────────────────
        if not _amount_match(tx_betrag, pattern):
            continue

        # ── 3. Datum ──────────────────────────────────────────────
        if not _date_match(tx_date, pattern):
            continue

        # ── Score berechnen (für bestes Match bei mehreren Treffern) ──
        # Score = Counterparty-Ähnlichkeit + inverse Betrags-Abweichung
        tx_iban      = tx.get("iban", "") or ""
        pattern_iban = (pattern.get("iban", "") or "").split()[0]
        if tx_iban and pattern_iban and pattern_iban != "-":
            cp_score = 1.0   # Exakter IBAN-Match = perfekt
        else:
            cp_score = _counterparty_similarity(
                tx.get("gegenpartei", ""),
                pattern.get("gegenpartei", ""),
            )

        avg        = abs(pattern.get("amount_avg", 1.0)) or 1.0
        amt_score  = 1.0 - min(1.0, abs(abs(tx_betrag) - avg) / avg)
        score      = 0.6 * cp_score + 0.4 * amt_score

        if score > best_score:
            best_score     = score
            best_match     = pattern
            best_match_ref = doc.reference

    if best_match is not None:
        # Match → Pattern bestätigen
        _confirm_pattern(db, best_match_ref, best_match, tx, tx_date)
        return True, best_match_ref.id

    # Kein Match → in distributions_db speichern
    doc_id = _store_in_distributions(db, tx)
    return False, None


# ─────────────────────────────────────────────
# DIREKT-AUFRUF (für Tests)
# ─────────────────────────────────────────────

def main():
    """
    Test-Modus: Liest letzte TX aus data/simulate_categorized.json
    und prüft ob sie zu einem Pattern passt.
    """
    import firebase_admin
    from firebase_admin import credentials, firestore as fs

    _ROOT  = pathlib.Path(__file__).parent
    _DATA  = _ROOT.parent / "data"
    cat_file = _DATA / "simulate_categorized.json"

    if not cat_file.exists():
        print(f"❌  {cat_file} nicht gefunden. Erst categorize_simulation.py ausführen.")
        raise SystemExit(1)

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        print("❌  GOOGLE_APPLICATION_CREDENTIALS nicht gesetzt.")
        raise SystemExit(1)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(creds_path))
    db = fs.client()

    import json
    with open(cat_file, encoding="utf-8") as f:
        all_tx = json.load(f)

    if not all_tx:
        print("❌  Keine Transaktionen in simulate_categorized.json")
        raise SystemExit(1)

    tx = all_tx[-1]   # Letzte TX testen
    print(f"\n  Test-TX: {tx.get('datum','')[:10]}  {tx.get('betrag',0):.2f} EUR  {tx.get('gegenpartei','')}")

    matched, doc_id = check_pattern(db, tx)

    if matched:
        print(f"  ✅  Match gefunden  →  patterns_db/{doc_id}")
    else:
        print(f"  ➡️   Kein Match  →  in distributions_db gespeichert")


if __name__ == "__main__":
    main()
