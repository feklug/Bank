"""
simulate/detect_patterns2.py
=============================
Liest distributions_db aus Firestore und prüft ob neue Muster entstanden sind.
Verwendet dieselbe Erkennungslogik wie training/detect_patterns.py.

Ablauf:
  1. Alle TX aus distributions_db laden
  2. Muster-Erkennung (RECURRING, SEASONAL, SEQUENTIAL) — identisch zu detect_patterns.py
  3. Für jedes neu gefundene Muster:
       → Eintrag in patterns_db schreiben
       → Zugehörige TX aus distributions_db löschen
  4. Kein Muster gefunden → nichts ändern

Schwellenwerte (etwas lockerer als Training, da weniger Datenpunkte):
  MIN_OCCURRENCES   : mind. 2 TX nötig für ein Muster (vs. 2 im Training)
  MIN_DATE_CONF     : 0.55 (vs. 0.60 im Training)
  MIN_FORECAST_CONF : 0.55 (vs. 0.60 im Training)

Importierbar für simulate/pipeline.py:
  from detect_patterns2 import detect_new_patterns
  result = detect_new_patterns(db)
  # result = {"new_patterns": int, "tx_moved": int}
"""

import hashlib
import os
import pathlib
import sys
from calendar import monthrange
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta, timezone
from statistics import mean, stdev, linear_regression
from typing import Optional

# ─────────────────────────────────────────────
# PFADE & KONSTANTEN
# ─────────────────────────────────────────────

_ROOT = pathlib.Path(__file__).parent        # simulate/
_DATA = _ROOT.parent / "data"                # data/

COLLECTION_DISTRIBUTIONS = "distributions_db"
COLLECTION_PATTERNS      = "patterns_db"

# Schwellenwerte (etwas lockerer als Training — weniger Datenpunkte in Simulation)
MIN_OCCURRENCES        = 2      # mind. 2 TX mit gleichem Muster
MIN_DATE_CONF          = 0.55
MIN_FORECAST_CONF      = 0.55
CUSTOM_MIN_DATE_CONF   = 0.50

# ─────────────────────────────────────────────
# ÖSTERREICHISCHER FEIERTAGSKALENDER
# (identisch mit detect_patterns.py)
# ─────────────────────────────────────────────

def _easter(year: int) -> date:
    a = year % 19; b = year // 100; c = year % 100
    d = b // 4;    e = b % 4;       f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4;    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day   = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


_holidays_cache: dict[int, set[date]] = {}

def _get_holidays(year: int) -> set[date]:
    if year not in _holidays_cache:
        easter = _easter(year)
        _holidays_cache[year] = {
            date(year,  1,  1), date(year,  1,  6),
            easter + timedelta(days=1),
            date(year,  5,  1),
            easter + timedelta(days=39),
            easter + timedelta(days=50),
            easter + timedelta(days=60),
            date(year,  8, 15), date(year, 10, 26),
            date(year, 11,  1), date(year, 12,  8),
            date(year, 12, 25), date(year, 12, 26),
        }
    return _holidays_cache[year]


def _next_banking_day(d: date) -> date:
    while True:
        if d.weekday() < 5 and d not in _get_holidays(d.year):
            return d
        d += timedelta(days=1)


# ─────────────────────────────────────────────
# DATUM-HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    s = str(s).strip()
    try:
        if "T" in s or " " in s:
            return datetime.fromisoformat(
                s.replace("Z", "+00:00").replace(" ", "T")
            ).date()
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _next_month_same_day(last: date, dom: int) -> date:
    m = last.month + 1 if last.month < 12 else 1
    y = last.year if last.month < 12 else last.year + 1
    return date(y, m, min(dom, monthrange(y, m)[1]))


def _next_quarter_same_dom(last: date, dom: int) -> date:
    m = last.month + 3
    y = last.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    return date(y, m, min(dom, monthrange(y, m)[1]))


def _compute_typical_time(transactions: list[dict]) -> Optional[str]:
    secs = []
    for t in transactions:
        d = t.get("datum", "")
        if "T" in d:
            try:
                dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
                secs.append(dt.hour * 3600 + dt.minute * 60 + dt.second)
            except ValueError:
                pass
    if not secs:
        return None
    avg = round(mean(secs))
    return f"{avg//3600:02d}:{(avg%3600)//60:02d}:{avg%60:02d}"


def _compute_next_expected(
    dates: list[date],
    interval_label: Optional[str],
    interval_days: Optional[int],
    anchor_dom: Optional[int],
) -> Optional[str]:
    if not dates or not interval_days:
        return None
    last  = dates[-1]
    label = (interval_label or "").split(" ")[0].upper()
    if label == "MONTHLY" and anchor_dom:
        raw = _next_month_same_day(last, anchor_dom)
    elif label == "QUARTERLY" and anchor_dom:
        raw = _next_quarter_same_dom(last, anchor_dom)
    else:
        raw = last + timedelta(days=interval_days)
    return _next_banking_day(raw).isoformat()


# ─────────────────────────────────────────────
# INTERVALL-ERKENNUNG
# (identisch mit detect_patterns.py)
# ─────────────────────────────────────────────

INTERVAL_LABELS = {
    7:   ("WEEKLY",      7,   2),
    14:  ("BIWEEKLY",   14,   3),
    30:  ("MONTHLY",    30,   5),
    60:  ("BIMONTHLY",  60,   5),
    90:  ("QUARTERLY",  90,  10),
    180: ("SEMIANNUAL", 180, 20),
    365: ("ANNUAL",     365, 30),
}


def _detect_interval(dates: list[date]) -> tuple[Optional[str], Optional[int], float]:
    if len(dates) < 2:
        return None, None, 0.0
    gaps    = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    avg_gap = mean(gaps)

    for interval, (label, _, tol) in INTERVAL_LABELS.items():
        if abs(avg_gap - interval) <= tol:
            in_tol = sum(1 for g in gaps if abs(g - interval) <= tol)
            return label, interval, round(in_tol / len(gaps), 2)

    for base_days, label, base_tol in [(30,"MONTHLY",5),(7,"WEEKLY",2),(90,"QUARTERLY",10),(365,"ANNUAL",30)]:
        ok = sum(1 for g in gaps if (n := round(g/base_days)) >= 1 and abs(g - n*base_days) <= base_tol*n)
        if ok / len(gaps) >= 0.60:
            return label, base_days, round(ok/len(gaps), 2)

    best_int = round(avg_gap)
    best_tol = max(5, round(avg_gap * 0.18))
    in_tol   = sum(1 for g in gaps if abs(g - best_int) <= best_tol)
    return "CUSTOM", best_int, round(in_tol/len(gaps), 2)


def _is_seasonal(dates: list[date], interval_label: Optional[str]) -> bool:
    if len(dates) < 2 or interval_label in ("MONTHLY", "WEEKLY", "BIWEEKLY"):
        return False
    month_set = set(d.month for d in dates)
    return len(month_set) <= max(2, len(dates) // 2)


# ─────────────────────────────────────────────
# ANOMALIE-ERKENNUNG
# ─────────────────────────────────────────────

def _amount_anomalies(amounts: list[float]) -> list[int]:
    if len(amounts) < 3:
        return []
    avg = mean(amounts)
    try:
        sd = stdev(amounts)
    except Exception:
        return []
    return [] if sd == 0 else [i for i, a in enumerate(amounts) if abs(a-avg)/sd > 2.0]


def _amount_trend(amounts: list[float]) -> Optional[float]:
    if len(amounts) < 3:
        return None
    try:
        slope, _ = linear_regression(range(len(amounts)), amounts)
        return round(slope, 4)
    except Exception:
        return None


# ─────────────────────────────────────────────
# FORECAST-KONFIDENZ
# ─────────────────────────────────────────────

def _forecast_confidence(date_conf: float, amounts: list[float], txs: list[dict]) -> float:
    if len(amounts) >= 2:
        avg = mean(amounts)
        try:
            sd = stdev(amounts)
        except Exception:
            sd = 0.0
        cv          = (sd / avg) if avg > 0 else 1.0
        amount_conf = max(0.0, 1.0 - cv / 0.20)
    else:
        amount_conf = 1.0

    conf = 0.6 * date_conf + 0.4 * amount_conf

    ibans = [t.get("iban") or "" for t in txs]
    valid = [ib for ib in ibans if ib]
    if valid and len(set(valid)) == 1:
        conf = min(1.0, conf + 0.15)
    if len(amounts) >= 3 and (stdev(amounts) if len(amounts) >= 2 else 1.0) == 0.0:
        conf = min(1.0, conf + 0.10)

    return round(conf, 2)


# ─────────────────────────────────────────────
# GROUP KEY  (identisch mit detect_patterns.py)
# ─────────────────────────────────────────────

def _group_key(t: dict) -> tuple:
    iban = t.get("iban") or ""
    cat  = t.get("category_level1", "SONDERKATEGORIEN")
    return ("__IBAN__", iban, cat) if iban else (t.get("gegenpartei", ""), cat, "")


# ─────────────────────────────────────────────
# PATTERN BAUEN
# ─────────────────────────────────────────────

def _build_pattern(txs: list[dict], seasonal: bool = False) -> Optional[dict]:
    """Baut ein Pattern-Dict — identische Struktur zu detect_patterns.py."""
    # date_obj hinzufügen wenn noch nicht vorhanden
    for t in txs:
        if "date_obj" not in t:
            t["date_obj"] = _parse_date(t.get("datum", "")) or date.today()
        if "datum_display" not in t:
            t["datum_display"] = str(t["date_obj"])

    dates   = [t["date_obj"] for t in txs]
    amounts = [abs(t.get("betrag", 0.0)) for t in txs]

    interval_label, interval_days, date_conf = _detect_interval(dates)
    if not interval_label:
        return None

    # Konfidenz-Schwelle prüfen
    min_dc = CUSTOM_MIN_DATE_CONF if interval_label == "CUSTOM" else MIN_DATE_CONF
    if date_conf < min_dc:
        return None

    dom_counts: defaultdict[int, int] = defaultdict(int)
    dow_counts: defaultdict[int, int] = defaultdict(int)
    for d in dates:
        dom_counts[d.day]     += 1
        dow_counts[d.weekday()] += 1

    anchor_dom = max(dom_counts, key=dom_counts.get) if dom_counts else None
    dow_num    = max(dow_counts, key=dow_counts.get) if dow_counts else None
    dow_names  = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"]
    dow        = dow_names[dow_num] if dow_num is not None else None

    fcast_conf    = _forecast_confidence(date_conf, amounts, txs)
    next_expected = _compute_next_expected(dates, interval_label, interval_days, anchor_dom)
    typical_time  = _compute_typical_time(txs)

    if next_expected and typical_time:
        next_expected_str = f"{next_expected} {typical_time}"
    elif next_expected:
        next_expected_str = next_expected
    else:
        next_expected_str = "-"

    ibans_in_group = [t.get("iban") or "" for t in txs]
    non_empty      = [ib for ib in ibans_in_group if ib]
    dominant_iban  = non_empty[0] if non_empty else "-"
    if non_empty and len(set(non_empty)) > 1:
        dominant_iban = Counter(non_empty).most_common(1)[0][0] + "  WECHSEL"

    amount_std = round(stdev(amounts), 2) if len(amounts) >= 2 else 0.0
    sample     = txs[0]

    anomaly_idxs = set(_amount_anomalies(amounts))
    anomalies = [
        {
            "date":        txs[i]["datum_display"],
            "amount":      txs[i]["betrag"],
            "type":        "AMOUNT",
            "description": txs[i].get("verwendungszweck", ""),
        }
        for i in sorted(anomaly_idxs)
    ]

    pattern = {
        "pattern_type":               "SEASONAL" if seasonal else "RECURRING",
        "gegenpartei":                sample.get("gegenpartei", ""),
        "iban":                       dominant_iban,
        "category":                   sample.get("category_level1", ""),
        "amount_avg":                 round(mean(amounts), 2),
        "amount_std":                 amount_std,
        "amount_min":                 round(min(amounts), 2),
        "amount_max":                 round(max(amounts), 2),
        "amount_sum":                 round(sum(amounts), 2),
        "amount_trend_per_period":    _amount_trend(amounts),
        "first_seen":                 txs[min(range(len(dates)), key=lambda i: dates[i])]["datum_display"],
        "last_seen":                  txs[max(range(len(dates)), key=lambda i: dates[i])]["datum_display"],
        "next_expected_date":         next_expected_str,
        "recurrence_interval":        f"{interval_label} (~{interval_days}d)" if interval_label else "UNBEKANNT",
        "recurrence_interval_days":   interval_days,
        "recurrence_date_confidence": round(date_conf, 2),
        "recurrence_confidence":      fcast_conf,
        "recurrence_day_of_month":    anchor_dom,
        "recurrence_day_of_week":     dow,
        "recurrence_has_gaps":        False,
        "recurrence_sample_size":     len(txs),
        "transaction_count":          len(txs),
        "transactions": [
            {k: v for k, v in t.items() if k not in ("date_obj", "datum_display")}
            for t in txs
        ],
        "anomalies":      anomalies,
        "source":         "simulation",
        "detected_at":    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "confirmation_count": 0,
    }

    if seasonal:
        months_seen = sorted(set(d.month for d in dates))
        month_names = ["Jan","Feb","Mar","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
        pattern["seasonal_months"] = [month_names[m-1] for m in months_seen]
        pattern["seasonal_years"]  = sorted(set(d.year for d in dates))

    return pattern if fcast_conf >= MIN_FORECAST_CONF else None


# ─────────────────────────────────────────────
# FIRESTORE: PATTERN SCHREIBEN
# ─────────────────────────────────────────────

def _make_pattern_id(pattern: dict) -> str:
    seed = (
        f"{pattern.get('pattern_type','')}|"
        f"{pattern.get('gegenpartei','')}|"
        f"{pattern.get('iban','')}|"
        f"{pattern.get('first_seen','')}"
    )
    return hashlib.sha1(seed.encode()).hexdigest()[:20]


def _write_pattern_to_firestore(db, pattern: dict) -> str:
    doc_id = _make_pattern_id(pattern)
    db.collection(COLLECTION_PATTERNS).document(doc_id).set(pattern)
    return doc_id


def _delete_from_distributions(db, doc_ids: list[str]):
    """Löscht TX-Dokumente aus distributions_db in Batches."""
    BATCH_SIZE = 400
    for i in range(0, len(doc_ids), BATCH_SIZE):
        batch = db.batch()
        for doc_id in doc_ids[i:i + BATCH_SIZE]:
            batch.delete(db.collection(COLLECTION_DISTRIBUTIONS).document(doc_id))
        batch.commit()


# ─────────────────────────────────────────────
# HAUPT-FUNKTION  (importierbar)
# ─────────────────────────────────────────────

def detect_new_patterns(db) -> dict:
    """
    Liest distributions_db aus Firestore, erkennt neue Muster,
    schreibt sie in patterns_db und löscht die TX aus distributions_db.

    Rückgabe:
        {
          "new_patterns": int,   # neu gefundene Muster
          "tx_moved":     int,   # TX aus distributions_db entfernt
        }
    """
    # ── 1. distributions_db laden ─────────────────────────────────
    dist_docs = list(db.collection(COLLECTION_DISTRIBUTIONS).stream())
    if not dist_docs:
        return {"new_patterns": 0, "tx_moved": 0}

    # Transaktionen mit Firestore-Doc-ID vorbereiten
    transactions: list[dict] = []
    for doc in dist_docs:
        t = doc.to_dict()
        t["_firestore_id"] = doc.id
        d = _parse_date(t.get("datum", ""))
        t["date_obj"]      = d or date.today()
        t["datum_display"] = str(d) if d else ""
        transactions.append(t)

    # Chronologisch sortieren
    transactions.sort(key=lambda t: t["date_obj"])

    # ── 2. Bereits bekannte Patterns laden (für Duplikat-Check) ───
    existing_patterns = {
        doc.to_dict().get("gegenpartei", "") + "|" + doc.to_dict().get("iban", "")
        for doc in db.collection(COLLECTION_PATTERNS).stream()
    }

    # ── 3. Gruppieren ─────────────────────────────────────────────
    groups: defaultdict[tuple, list] = defaultdict(list)
    for t in transactions:
        groups[_group_key(t)].append(t)

    # ── 4. Muster-Erkennung ───────────────────────────────────────
    used_firestore_ids: set[str] = set()
    new_patterns:       list[dict] = []

    for key, group_txs in groups.items():
        if len(group_txs) < MIN_OCCURRENCES:
            continue

        # Bereits bekanntes Pattern? (verhindert Duplikate in patterns_db)
        gegenpartei = group_txs[0].get("gegenpartei", "")
        iban        = group_txs[0].get("iban", "") or ""
        lookup_key  = f"{gegenpartei}|{iban}"
        if lookup_key in existing_patterns:
            continue

        seasonal = _is_seasonal([t["date_obj"] for t in group_txs], None)
        pattern  = _build_pattern(group_txs, seasonal=seasonal)

        if pattern is None:
            continue

        new_patterns.append((pattern, [t["_firestore_id"] for t in group_txs]))

    # ── 5. Ergebnisse in Firestore schreiben ──────────────────────
    total_moved = 0
    for pattern, firestore_ids in new_patterns:
        doc_id = _write_pattern_to_firestore(db, pattern)
        _delete_from_distributions(db, firestore_ids)
        total_moved += len(firestore_ids)
        print(
            f"  ✅  Neues Pattern [{pattern['pattern_type']}]  "
            f"{pattern['gegenpartei'][:35]:<35}  "
            f"{len(firestore_ids)} TX  →  patterns_db/{doc_id}"
        )

    return {
        "new_patterns": len(new_patterns),
        "tx_moved":     total_moved,
    }


# ─────────────────────────────────────────────
# DIREKTAUFRUF (für Tests)
# ─────────────────────────────────────────────

def main():
    import firebase_admin
    from firebase_admin import credentials, firestore as fs

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        print("❌  GOOGLE_APPLICATION_CREDENTIALS nicht gesetzt.")
        raise SystemExit(1)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(creds_path))
    db = fs.client()

    print(f"\n{'='*60}")
    print("  DETECT PATTERNS 2  –  Neue Muster in distributions_db")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    result = detect_new_patterns(db)

    print(f"\n{'─'*60}")
    if result["new_patterns"] == 0:
        print("  ℹ️   Keine neuen Muster gefunden — distributions_db unverändert")
    else:
        print(f"  ✅  {result['new_patterns']} neue Muster → patterns_db")
        print(f"  🗑️   {result['tx_moved']} TX aus distributions_db entfernt")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
