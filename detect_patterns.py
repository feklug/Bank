"""
detect_patterns.py
==================
Erkennt Muster (RECURRING, SEASONAL, SEQUENTIAL) in Banktransaktionen.
Speichert Ergebnisse direkt in Firestore:
  - patterns_db    → jedes Pattern als eigenes Dokument (inkl. Transaktionen)
  - distributions_db → jede Transaktion ohne Muster als eigenes Dokument

PUBLIC API (für pipeline.py):
    from detect_patterns import detect_patterns
    result = detect_patterns(categorized)   # categorized = Output von categorize.py
    # result = {"total_patterns": int, "distributions": int}

CLI (Standalone):
    python detect_patterns.py [tink_categorized.json]

Änderungen gegenüber Vorversion:
  BUG-FIXES:
    [B1] get("iban", "") → get("iban") or ""  (None-safe bei "iban": null in JSON)
    [B2] group_key IBAN-first: gleiche IBAN = gleiche Gegenpartei, Kategorie ignoriert
    [B3] Österreichischer Feiertagskalender ersetzt durch Schweizer (Bund + ZH)
    [B4] next_expected_date für MONTHLY via next_month_same_day() – kein Datums-Drift

  VERBESSERUNGEN:
    [V1] Anomalie-Datums-Toleranz proportional (10 % des Intervalls, min. 3 Tage)
    [V2] amount_std im Pattern gespeichert (für Forecasting-Konfidenzintervalle)
    [V3] Betragstrend (linearer Slope) gespeichert – erkennt Gehaltserhöhungen
    [V4] IBAN-Konsistenz hebt Konfidenz an (+0.10 wenn alle IBAN identisch)
    [V5] IBAN-Wechsel als neue Anomalie-Kategorie [IBAN]
    [V6] Duplikat-Erkennung als Anomalie-Kategorie [DUPLICATE]
    [V7] Kombinierte Konfidenz (60 % Datum + 40 % Betrag)
    [V8] CUSTOM-Muster-Mindestkonfidenz auf 0.70 angehoben
    [V9] INTERNAL-Pattern entfernt; interne Umbuchungen landen in OHNE MUSTER
    [V10] Konfidenz-Schwelle 75 %: Datum-Konf. UND Forecast-Konf. müssen >= 0.75 sein,
          sonst wird das Pattern verworfen und die Transaktionen als OHNE MUSTER gefuehrt

  PIPELINE-INTEGRATION:
    [P1] detect_patterns() als öffentliche Funktion – nimmt In-Memory-Liste entgegen
    [P2] Firestore-Speicherung direkt integriert (keine organisational.py nötig)
    [P3] Deterministische Dokument-IDs – mehrfache Ausführungen überschreiben statt duplizieren
    [P4] date_obj und interne Felder (_transactions, _idxs) werden vor dem Speichern entfernt
"""

import json
import re
import sys
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, timedelta
from statistics import linear_regression, mean, stdev

# ─────────────────────────────────────────────
# 1. SCHWEIZER FEIERTAGSKALENDER  [B3]
# ─────────────────────────────────────────────

def _easter(year: int) -> date:
    """Berechnet Ostersonntag nach Gauss/Butcher."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def get_swiss_holidays_zh(year: int) -> set[date]:
    """Gibt alle Bankfeiertage fuer den Kanton Zuerich zurueck."""
    easter = _easter(year)
    return {
        date(year, 1, 1),                        # Neujahr
        date(year, 1, 2),                        # Berchtoldstag (ZH)
        easter - timedelta(days=2),              # Karfreitag
        easter + timedelta(days=1),              # Ostermontag
        date(year, 5, 1),                        # Tag der Arbeit (ZH)
        easter + timedelta(days=39),             # Auffahrt (Christi Himmelfahrt)
        easter + timedelta(days=50),             # Pfingstmontag
        date(year, 8, 1),                        # Bundesfeiertag
        date(year, 12, 25),                      # Weihnachten
        date(year, 12, 26),                      # Stephanstag (ZH)
    }


_holidays_cache: dict[int, set[date]] = {}


def next_banking_day(d: date) -> date:
    """Gibt den naechsten Schweizer Bankarbeitstag zurueck (kein Sa/So/Feiertag ZH)."""
    current = d
    while True:
        year = current.year
        if year not in _holidays_cache:
            _holidays_cache[year] = get_swiss_holidays_zh(year)
        if current.weekday() < 5 and current not in _holidays_cache[year]:
            return current
        current += timedelta(days=1)


def adjust_to_banking_day(d: date) -> date:
    return next_banking_day(d)


# ─────────────────────────────────────────────
# 2. NAECHSTEN MONATSTAG BERECHNEN  [B4]
# ─────────────────────────────────────────────

def next_month_same_day(last_date: date, anchor_dom: int) -> date:
    m = last_date.month + 1 if last_date.month < 12 else 1
    y = last_date.year if last_date.month < 12 else last_date.year + 1
    day = min(anchor_dom, monthrange(y, m)[1])
    return date(y, m, day)


def next_quarter_same_dom(last_date: date, anchor_dom: int) -> date:
    m = last_date.month + 3
    y = last_date.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    day = min(anchor_dom, monthrange(y, m)[1])
    return date(y, m, day)


def compute_next_expected(dates: list[date], interval_label: str | None,
                          interval_days: int | None, anchor_dom: int | None) -> date | None:
    if not interval_days or not dates:
        return None
    last = dates[-1]
    if interval_label == "MONTHLY" and anchor_dom:
        raw = next_month_same_day(last, anchor_dom)
    elif interval_label == "QUARTERLY" and anchor_dom:
        raw = next_quarter_same_dom(last, anchor_dom)
    else:
        raw = last + timedelta(days=interval_days)
    return adjust_to_banking_day(raw)


# ─────────────────────────────────────────────
# 3. DATEN LADEN & VORBEREITEN
# ─────────────────────────────────────────────

def _prepare_transactions(data: list[dict]) -> list[dict]:
    """
    Fügt date_obj hinzu und sortiert die Liste chronologisch.
    Verändert die Originaldaten nicht (arbeitet auf Kopien).
    """
    result = []
    for t in data:
        entry = dict(t)
        entry["date_obj"] = date.fromisoformat(t["datum"])
        result.append(entry)
    return sorted(result, key=lambda x: x["date_obj"])


def load_transactions(path: str) -> list[dict]:
    """Lädt Transaktionen aus einer JSON-Datei (CLI-Verwendung)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _prepare_transactions(data)


def group_key(t: dict) -> tuple:
    """
    Gruppierschluessel fuer Transaktionen.  [B1] [B2]
    Mit IBAN : ("__IBAN__", iban, category_level1)
    Ohne IBAN: (gegenpartei, category_level1, "")
    """
    iban = t.get("iban") or ""
    cat  = t["category_level1"]
    if iban:
        return ("__IBAN__", iban, cat)
    return (t["gegenpartei"], cat, "")


# ─────────────────────────────────────────────
# 4. INTERVALL-ERKENNUNG
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


def detect_interval(dates: list[date]) -> tuple[str | None, int | None, float]:
    if len(dates) < 2:
        return None, None, 0.0
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    if not gaps:
        return None, None, 0.0
    avg_gap = mean(gaps)
    best_label, best_interval, best_tolerance = None, None, 0
    for interval, (label, _, tolerance) in INTERVAL_LABELS.items():
        if abs(avg_gap - interval) <= tolerance:
            best_label     = label
            best_interval  = interval
            best_tolerance = tolerance
            break
    if not best_label:
        best_label     = "CUSTOM"
        best_interval  = round(avg_gap)
        best_tolerance = max(5, round(avg_gap * 0.18))
    in_tolerance = sum(1 for g in gaps if abs(g - best_interval) <= best_tolerance)
    confidence = in_tolerance / len(gaps)
    return best_label, best_interval, round(confidence, 2)


def is_seasonal(dates: list[date], interval_label: str | None) -> bool:
    if len(dates) < 2:
        return False
    if interval_label in ("MONTHLY", "WEEKLY", "BIWEEKLY"):
        return False
    month_set = set(d.month for d in dates)
    return len(month_set) <= max(2, len(dates) // 2)


# ─────────────────────────────────────────────
# 5. ANOMALIE-ERKENNUNG
# ─────────────────────────────────────────────

def detect_amount_anomalies(amounts: list[float], tolerance_factor: float = 2.0) -> list[int]:
    if len(amounts) < 3:
        return []
    avg = mean(amounts)
    try:
        sd = stdev(amounts)
    except Exception:
        return []
    if sd == 0:
        return []
    return [i for i, a in enumerate(amounts) if abs(a - avg) / sd > tolerance_factor]


def detect_date_anomalies(dates: list[date], interval_days: int) -> list[int]:
    tolerance = max(3, round(interval_days * 0.10))
    anomalies = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if abs(gap - interval_days) > tolerance:
            anomalies.append(i)
    return anomalies


def detect_iban_anomalies(transactions: list[dict]) -> list[int]:
    ibans = [(i, t.get("iban") or "") for i, t in enumerate(transactions)]
    non_empty = [(i, ib) for i, ib in ibans if ib]
    if len(non_empty) < 2:
        return []
    dominant = max(set(ib for _, ib in non_empty), key=lambda x: sum(1 for _, ib in non_empty if ib == x))
    return [i for i, ib in non_empty if ib != dominant]


def detect_duplicates(transactions: list[dict]) -> list[int]:
    duplicates = set()
    for i in range(len(transactions)):
        for j in range(i + 1, len(transactions)):
            ti, tj = transactions[i], transactions[j]
            days_apart = abs((tj["date_obj"] - ti["date_obj"]).days)
            if days_apart > 10:
                break
            amt_i = abs(ti["betrag"])
            amt_j = abs(tj["betrag"])
            if amt_i == 0:
                continue
            if abs(amt_i - amt_j) / amt_i <= 0.01:
                if ti["gegenpartei"] == tj["gegenpartei"]:
                    duplicates.add(j)
    return sorted(duplicates)


def compute_amount_trend(amounts: list[float]) -> float | None:
    if len(amounts) < 3:
        return None
    try:
        slope, _ = linear_regression(range(len(amounts)), amounts)
        return round(slope, 4)
    except Exception:
        return None


# ─────────────────────────────────────────────
# 6. KOMBINIERTE KONFIDENZ  [V7]
# ─────────────────────────────────────────────

def forecast_confidence(date_conf: float, amounts: list[float],
                        transactions: list[dict]) -> float:
    sd = 0.0
    if len(amounts) >= 2:
        avg = mean(amounts)
        try:
            sd = stdev(amounts)
        except Exception:
            sd = 0.0
        cv = (sd / avg) if avg > 0 else 1.0
        amount_conf = max(0.0, 1.0 - cv / 0.20)
    else:
        amount_conf = 1.0
    conf = 0.6 * date_conf + 0.4 * amount_conf
    ibans = [t.get("iban") or "" for t in transactions]
    non_empty = [ib for ib in ibans if ib]
    if non_empty and len(set(non_empty)) == 1:
        conf = min(1.0, conf + 0.15)
    if len(amounts) >= 3 and sd == 0.0:
        conf = min(1.0, conf + 0.10)
    return round(conf, 2)


# ─────────────────────────────────────────────
# 7. RECURRING PATTERN BUILDER
# ─────────────────────────────────────────────

def build_recurring(transactions: list[dict], seasonal: bool = False) -> dict:
    dates   = [t["date_obj"] for t in transactions]
    amounts = [abs(t["betrag"]) for t in transactions]
    interval_label, interval_days, date_conf = detect_interval(dates)

    dom_counts: defaultdict[int, int] = defaultdict(int)
    dow_counts: defaultdict[int, int] = defaultdict(int)
    for d in dates:
        dom_counts[d.day] += 1
        dow_counts[d.weekday()] += 1
    anchor_dom = max(dom_counts, key=dom_counts.get) if dom_counts else None
    dow_num    = max(dow_counts, key=dow_counts.get) if dow_counts else None
    dow_names  = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    dow        = dow_names[dow_num] if dow_num is not None else None

    fcast_conf    = forecast_confidence(date_conf, amounts, transactions)
    next_expected = compute_next_expected(dates, interval_label, interval_days, anchor_dom)

    amount_anomaly_idxs = set(detect_amount_anomalies(amounts))
    date_anomaly_idxs   = set(detect_date_anomalies(dates, interval_days or 30))
    iban_anomaly_idxs   = set(detect_iban_anomalies(transactions))
    duplicate_idxs      = set(detect_duplicates(transactions))
    all_anomaly_idxs    = amount_anomaly_idxs | date_anomaly_idxs | iban_anomaly_idxs | duplicate_idxs

    anomalies = []
    for idx in sorted(all_anomaly_idxs):
        t = transactions[idx]
        anomaly_type = []
        if idx in duplicate_idxs:      anomaly_type.append("DUPLICATE")
        if idx in iban_anomaly_idxs:   anomaly_type.append("IBAN")
        if idx in amount_anomaly_idxs: anomaly_type.append("AMOUNT")
        if idx in date_anomaly_idxs:   anomaly_type.append("DATE")
        anomalies.append({
            "date":        t["datum"],
            "amount":      t["betrag"],
            "type":        "+".join(anomaly_type),
            "description": t["verwendungszweck"],
        })

    has_gaps = False
    if interval_days and len(dates) > 1:
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        has_gaps = any(g > interval_days * 1.8 for g in gaps)

    amount_std   = round(stdev(amounts), 2) if len(amounts) >= 2 else 0.0
    amount_trend = compute_amount_trend(amounts)

    sample         = transactions[0]
    ibans_in_group = [t.get("iban") or "" for t in transactions]
    non_empty_ib   = [ib for ib in ibans_in_group if ib]
    dominant_iban  = non_empty_ib[0] if non_empty_ib else "-"
    if non_empty_ib and len(set(non_empty_ib)) > 1:
        dominant_iban = Counter(non_empty_ib).most_common(1)[0][0] + "  WECHSEL"

    pattern_type = "SEASONAL" if seasonal else "RECURRING"

    result = {
        "pattern_type":               pattern_type,
        "gegenpartei":                sample["gegenpartei"],
        "iban":                       dominant_iban,
        "category":                   sample["category_level1"],
        "amount_avg":                 round(mean(amounts), 2),
        "amount_std":                 amount_std,
        "amount_min":                 round(min(amounts), 2),
        "amount_max":                 round(max(amounts), 2),
        "amount_sum":                 round(sum(amounts), 2),
        "amount_trend_per_period":    amount_trend,
        "first_seen":                 min(dates).isoformat(),
        "last_seen":                  max(dates).isoformat(),
        "next_expected_date":         next_expected.isoformat() if next_expected else "-",
        "recurrence_interval":        f"{interval_label} (~{interval_days}d)" if interval_label else "UNBEKANNT",
        "recurrence_interval_days":   interval_days,
        "recurrence_date_confidence": round(date_conf, 2),
        "recurrence_confidence":      fcast_conf,
        "recurrence_day_of_month":    anchor_dom,
        "recurrence_day_of_week":     dow,
        "recurrence_has_gaps":        has_gaps,
        "recurrence_sample_size":     len(transactions),
        "anomalies":                  anomalies,
        "_transactions":              transactions,   # intern, wird nicht in Firestore gespeichert
    }

    if seasonal:
        months_seen = sorted(set(d.month for d in dates))
        month_names = ["Jan","Feb","Mar","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
        result["seasonal_months"] = [month_names[m - 1] for m in months_seen]
        result["seasonal_years"]  = sorted(set(d.year for d in dates))

    return result


# ─────────────────────────────────────────────
# 8. SEQUENTIAL PATTERN ERKENNUNG
# ─────────────────────────────────────────────

SEQUENTIAL_RULES = [
    {
        "name":             "Lohnzahlung -> Sozialversicherung",
        "trigger_category": "AUSGABEN – PERSONAL",
        "follow_category":  "AUSGABEN – SOZIALVERSICHERUNGEN",
        "max_days":         10,
        "description":      "Lohnzahlungen loesen typischerweise SV-Beitraege aus",
    },
    {
        "name":             "Quartalsabrechnung -> Steuerueberweisung",
        "trigger_keyword":  "Q",
        "trigger_category": "AUSGABEN – BETRIEBSKOSTEN",
        "follow_category":  "AUSGABEN – STEUERN & ABGABEN",
        "max_days":         15,
        "description":      "Quartalsabrechnung fuehrt oft zu Steuerzahlungen",
    },
    {
        "name":             "Einnahme -> Interne Umbuchung",
        "trigger_category": "EINNAHMEN",
        "follow_category":  "NEUTRALE / INTERNE BEWEGUNGEN",
        "max_days":         5,
        "description":      "Einnahmen werden oft auf andere Konten umgebucht",
    },
    {
        "name":             "Investition -> Kreditbelastung",
        "trigger_category": "AUSGABEN – INVESTITIONEN",
        "follow_category":  "AUSGABEN – FINANZEN & BANKING",
        "max_days":         30,
        "description":      "Investitionen koennen Kreditkosten ausloesen",
    },
]


def find_sequential_patterns(transactions: list[dict], used_ids: set[int]) -> list[dict]:
    patterns = []
    for rule in SEQUENTIAL_RULES:
        matched_pairs = []
        triggers = [
            (i, t) for i, t in enumerate(transactions)
            if i not in used_ids
            and t["category_level1"] == rule.get("trigger_category", "")
            and ("trigger_keyword" not in rule or rule["trigger_keyword"] in t["verwendungszweck"])
        ]
        follows = [
            (i, t) for i, t in enumerate(transactions)
            if i not in used_ids
            and t["category_level1"] == rule.get("follow_category", "")
        ]
        used_follows: set[int] = set()
        for ti, trigger in triggers:
            best_follow = None
            best_days   = rule["max_days"] + 1
            for fi, follow in follows:
                if fi in used_follows or ti == fi:
                    continue
                days = (follow["date_obj"] - trigger["date_obj"]).days
                if 0 < days <= rule["max_days"] and days < best_days:
                    best_follow = (fi, follow)
                    best_days   = days
            if best_follow:
                matched_pairs.append({
                    "trigger":      trigger,
                    "follow":       best_follow[1],
                    "days_between": best_days,
                    "trigger_idx":  ti,
                    "follow_idx":   best_follow[0],
                })
                used_follows.add(best_follow[0])

        if len(matched_pairs) < 2:
            continue

        all_txs  = []
        all_idxs = []
        for pair in matched_pairs:
            all_txs.extend([pair["trigger"], pair["follow"]])
            all_idxs.extend([pair["trigger_idx"], pair["follow_idx"]])

        amounts_trigger = [abs(p["trigger"]["betrag"]) for p in matched_pairs]
        amounts_follow  = [abs(p["follow"]["betrag"])  for p in matched_pairs]
        all_amounts     = [abs(t["betrag"]) for t in all_txs]
        dates           = sorted(t["date_obj"] for t in all_txs)
        avg_delay       = round(mean([p["days_between"] for p in matched_pairs]), 1)

        last_trigger_date = max(p["trigger"]["date_obj"] for p in matched_pairs)
        next_expected     = adjust_to_banking_day(last_trigger_date + timedelta(days=round(avg_delay)))

        pattern = {
            "pattern_type":            "SEQUENTIAL",
            "sequence_name":           rule["name"],
            "sequence_description":    rule["description"],
            "gegenpartei":             f"{matched_pairs[0]['trigger']['gegenpartei']} -> {matched_pairs[0]['follow']['gegenpartei']}",
            "iban":                    "-",
            "category":                f"{rule['trigger_category']} -> {rule['follow_category']}",
            "amount_avg":              round(mean(all_amounts), 2),
            "amount_std":              round(stdev(all_amounts), 2) if len(all_amounts) >= 2 else 0.0,
            "amount_min":              round(min(all_amounts), 2),
            "amount_max":              round(max(all_amounts), 2),
            "amount_sum":              round(sum(all_amounts), 2),
            "amount_trend_per_period": compute_amount_trend(all_amounts),
            "first_seen":              min(dates).isoformat(),
            "last_seen":               max(dates).isoformat(),
            "next_expected_date":      next_expected.isoformat(),
            "seq_pair_count":          len(matched_pairs),
            "seq_avg_delay_days":      avg_delay,
            "seq_trigger_avg_amount":  round(mean(amounts_trigger), 2),
            "seq_follow_avg_amount":   round(mean(amounts_follow), 2),
            "seq_trigger_category":    rule["trigger_category"],
            "seq_follow_category":     rule["follow_category"],
            "seq_consistency_score":   round(min(1.0, len(matched_pairs) / max(len(triggers), 1)), 2),
            "seq_amount_ratio":        round(mean(amounts_follow) / mean(amounts_trigger), 4) if mean(amounts_trigger) > 0 else None,
            "anomalies":               detect_amount_anomalies(all_amounts),
            "_idxs":                   all_idxs,         # intern
            "_transactions":           all_txs,          # intern
        }
        patterns.append(pattern)
    return patterns


# ─────────────────────────────────────────────
# 9. HAUPT-ANALYSE
# ─────────────────────────────────────────────

MIN_DATE_CONF: dict[str, float] = {
    "CUSTOM":  0.70,
    "DEFAULT": 0.40,
}
MIN_PATTERN_DATE_CONF:     float = 0.75
MIN_PATTERN_FORECAST_CONF: float = 0.75


def _run_analysis(transactions: list[dict]) -> tuple[list, list, list, list]:
    """
    Kernanalyse: gibt (recurring, seasonal, sequential, no_pattern) zurück.
    transactions müssen bereits date_obj enthalten.
    """
    used_ids: set[int] = set()

    patterns_recurring:  list[dict] = []
    patterns_seasonal:   list[dict] = []
    patterns_sequential: list[dict] = []

    # A) RECURRING / SEASONAL
    groups: defaultdict[tuple, list] = defaultdict(list)
    for i, t in enumerate(transactions):
        groups[group_key(t)].append((i, t))

    for key, indexed_txs in groups.items():
        available = [(i, t) for i, t in indexed_txs if i not in used_ids]
        if len(available) < 2:
            continue

        idxs  = [i for i, _ in available]
        txs   = [t for _, t in available]
        dates = [t["date_obj"] for t in txs]

        interval_label, interval_days, date_conf = detect_interval(dates)
        if not interval_label:
            continue

        min_conf = MIN_DATE_CONF.get(interval_label, MIN_DATE_CONF["DEFAULT"])
        if date_conf < min_conf:
            continue

        seasonal = is_seasonal(dates, interval_label)
        pattern  = build_recurring(txs, seasonal=seasonal)
        if not pattern:
            continue

        if (pattern["recurrence_date_confidence"] < MIN_PATTERN_DATE_CONF
                or pattern["recurrence_confidence"] < MIN_PATTERN_FORECAST_CONF):
            continue

        for i in idxs:
            used_ids.add(i)

        if seasonal:
            patterns_seasonal.append(pattern)
        else:
            patterns_recurring.append(pattern)

    # B) SEQUENTIAL
    seq_patterns = find_sequential_patterns(transactions, used_ids)
    for p in seq_patterns:
        for i in p["_idxs"]:
            used_ids.add(i)
    patterns_sequential = seq_patterns

    # C) OHNE MUSTER
    no_pattern = [transactions[i] for i in range(len(transactions)) if i not in used_ids]

    return patterns_recurring, patterns_seasonal, patterns_sequential, no_pattern


# ─────────────────────────────────────────────
# 10. FIRESTORE-INTEGRATION  [P2]
# ─────────────────────────────────────────────

def _make_doc_id(raw: str) -> str:
    """
    Erstellt einen sicheren, deterministischen Firestore-Dokument-ID aus einem
    beliebigen String. Ersetzt alle ungültigen Zeichen durch '_' und kürzt auf 200 Zeichen.
    """
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)
    return safe[:200]


def _clean_transaction(t: dict) -> dict:
    """
    Entfernt interne Felder (date_obj, _*) aus einer Transaktion
    bevor sie in Firestore geschrieben wird.
    """
    return {
        k: v for k, v in t.items()
        if not k.startswith("_") and k != "date_obj"
    }


def _clean_pattern(p: dict) -> tuple[dict, list[dict]]:
    """
    Trennt ein Pattern-Dict in:
      - pattern_doc  : Metadaten-Felder (ohne interne _* Felder und ohne Transaktionen)
      - transactions : bereinigte Transaktionsliste

    Returns:
        (pattern_doc, transactions_list)
    """
    transactions = [_clean_transaction(t) for t in p.get("_transactions", [])]
    pattern_doc  = {
        k: v for k, v in p.items()
        if not k.startswith("_")
    }
    return pattern_doc, transactions


def _init_firestore():
    """
    Initialisiert Firebase Admin SDK mit dem Service-Account-JSON.
    Gibt den Firestore-Client zurück.
    Wirft RuntimeError wenn die Datei fehlt.
    """
    import firebase_admin
    from firebase_admin import credentials, firestore

    SERVICE_ACCOUNT_FILE = "bank-417a7-firebase-adminsdk-fbsvc-60ba2be615.json"

    # Verhindert Doppelinitialisierung bei wiederholten Aufrufen
    if not firebase_admin._apps:
        import os
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            raise RuntimeError(
                f"Service-Account-Datei nicht gefunden: {SERVICE_ACCOUNT_FILE}\n"
                "GitHub Action: Secrets → FIREBASE_SERVICE_ACCOUNT muss gesetzt sein."
            )
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)

    return firestore.client()


def _save_to_firestore(
    patterns_recurring:  list[dict],
    patterns_seasonal:   list[dict],
    patterns_sequential: list[dict],
    no_pattern:          list[dict],
) -> dict[str, int]:
    """
    Speichert alle Ergebnisse in Firestore:

    patterns_db:
      Dokument-ID: {PATTERN_TYPE}_{gegenpartei}_{first_seen}
      Felder:      alle Pattern-Metadaten
      Unterfeld:   "transactions" (Array der zugehörigen Transaktionen)

    distributions_db:
      Dokument-ID: {datum}_{gegenpartei}_{betrag}
      Felder:      alle Transaktionsfelder

    Alle Schreibvorgänge nutzen set() → idempotent, kein Duplizieren bei
    wiederholten Pipeline-Durchläufen.

    Returns:
        {"total_patterns": int, "distributions": int}
    """
    db = _init_firestore()
    patterns_ref     = db.collection("patterns_db")
    distributions_ref = db.collection("distributions_db")

    total_patterns = 0

    for p in patterns_recurring + patterns_seasonal + patterns_sequential:
        pattern_doc, transactions = _clean_pattern(p)

        # Deterministischer Dokument-ID  [P3]
        raw_id = f"{p['pattern_type']}_{p['gegenpartei']}_{p['first_seen']}"
        doc_id = _make_doc_id(raw_id)

        # Pattern-Metadaten + Transaktionen als Array im selben Dokument
        pattern_doc["transactions"] = transactions

        patterns_ref.document(doc_id).set(pattern_doc)
        total_patterns += 1

    distributions = 0
    for t in no_pattern:
        clean = _clean_transaction(t)

        raw_id = f"{clean['datum']}_{clean['gegenpartei']}_{clean['betrag']}"
        doc_id = _make_doc_id(raw_id)

        distributions_ref.document(doc_id).set(clean)
        distributions += 1

    return {"total_patterns": total_patterns, "distributions": distributions}


# ─────────────────────────────────────────────
# 11. PUBLIC API  (für pipeline.py)  [P1]
# ─────────────────────────────────────────────

def detect_patterns(categorized: list[dict]) -> dict[str, int]:
    """
    Haupteinstiegspunkt für die Pipeline.

    Parameters:
        categorized : In-Memory-Liste aus categorize.categorize()
                      Jedes Dict enthält mindestens: datum, betrag, gegenpartei,
                      verwendungszweck, iban, category_level1

    Ablauf:
      1. date_obj hinzufügen und chronologisch sortieren
      2. Muster erkennen (RECURRING, SEASONAL, SEQUENTIAL)
      3. Patterns → patterns_db in Firestore (ein Dokument pro Muster)
      4. Ohne Muster → distributions_db in Firestore (ein Dokument pro Transaktion)

    Returns:
        {"total_patterns": int, "distributions": int}
        → kompatibel mit pipeline.py Ausgabe-Format
    """
    transactions = _prepare_transactions(categorized)
    rec, sea, seq, no_pat = _run_analysis(transactions)

    print(f"    RECURRING  : {len(rec)}")
    print(f"    SEASONAL   : {len(sea)}")
    print(f"    SEQUENTIAL : {len(seq)}")
    print(f"    Ohne Muster: {len(no_pat)}")
    print(f"    → Firestore speichern...")

    fs_result = _save_to_firestore(rec, sea, seq, no_pat)
    return fs_result


# ─────────────────────────────────────────────
# 12. KONSOLENAUSGABE (nur CLI)
# ─────────────────────────────────────────────

SEP      = "-" * 70
SEP_THIN = "." * 70


def fmt_amount(a: float) -> str:
    return f"CHF {a:>10,.2f}"


def fmt_trend(slope: float | None) -> str:
    if slope is None:
        return "-"
    sign = "+" if slope > 0 else ("-" if slope < 0 else "=")
    return f"{sign} {abs(slope):,.4f} CHF/Periode"


def print_pattern_header(p: dict, idx: int, total: int):
    pt = p["pattern_type"]
    color_map = {"RECURRING": "\033[94m", "SEASONAL": "\033[95m", "SEQUENTIAL": "\033[93m"}
    color = color_map.get(pt, "")
    print(f"\n{SEP}")
    print(f"  {color}[{pt}]\033[0m  #{idx}/{total}  {p.get('sequence_name', p['gegenpartei'])}")
    print(SEP)


def print_common_fields(p: dict):
    print(f"  Gegenpartei    : {p['gegenpartei']}")
    print(f"  IBAN           : {p['iban']}")
    print(f"  Kategorie      : {p['category']}")
    print(f"  Zeitraum       : {p['first_seen']}  ->  {p['last_seen']}")
    print(f"  Naechstes Dat. : \033[92m{p['next_expected_date']}\033[0m  (CH Bankarbeitstag, ZH)")
    print(f"  {SEP_THIN}")
    print(f"  Betrag Oe      : {fmt_amount(p['amount_avg'])}")
    print(f"  Betrag Sigma   : {fmt_amount(p.get('amount_std', 0.0))}")
    print(f"  Betrag Min     : {fmt_amount(p['amount_min'])}")
    print(f"  Betrag Max     : {fmt_amount(p['amount_max'])}")
    print(f"  Betrag Summe   : {fmt_amount(p['amount_sum'])}")
    if p.get("amount_trend_per_period") is not None:
        print(f"  Betrag Trend   : {fmt_trend(p['amount_trend_per_period'])}")


def print_recurring_fields(p: dict):
    print(f"  {SEP_THIN}")
    print(f"  Intervall      : {p['recurrence_interval']}")
    print(f"  Datum-Konf.    : {p['recurrence_date_confidence']*100:.0f}%  (Erkennung)")
    print(f"  Forecast-Konf. : {p['recurrence_confidence']*100:.0f}%  (Datum + Betrag + IBAN)")
    print(f"  Stichproben    : {p['recurrence_sample_size']}")
    print(f"  Tag im Monat   : {p['recurrence_day_of_month']}")
    print(f"  Wochentag      : {p['recurrence_day_of_week']}")
    print(f"  Luecken vorh.  : {'JA' if p['recurrence_has_gaps'] else 'Nein'}")
    if p.get("seasonal_months"):
        print(f"  Saisonmonate   : {', '.join(p['seasonal_months'])}")
    if p.get("seasonal_years"):
        print(f"  Saisonstueck   : {p['seasonal_years']}")


def print_sequential_fields(p: dict):
    print(f"  {SEP_THIN}")
    print(f"  Oe Verzoegerung  : {p['seq_avg_delay_days']} Tage")
    print(f"  Paare erkannt    : {p['seq_pair_count']}")
    print(f"  Trigger-Kat.     : {p['seq_trigger_category']}")
    print(f"  Folge-Kat.       : {p['seq_follow_category']}")
    print(f"  Trigger-Oe Betr. : {fmt_amount(p['seq_trigger_avg_amount'])}")
    print(f"  Folge-Oe Betrag  : {fmt_amount(p['seq_follow_avg_amount'])}")
    if p["seq_amount_ratio"] is not None:
        print(f"  Betrag-Verh.     : {p['seq_amount_ratio']:.4f}  (Folge/Trigger)")
    print(f"  Konsistenz       : {p['seq_consistency_score']*100:.0f}%")
    print(f"  Beschreibung     : {p['sequence_description']}")


def print_anomalies(p: dict):
    anomalies = p.get("anomalies", [])
    if not isinstance(anomalies, list) or not anomalies or not isinstance(anomalies[0], dict):
        return
    print(f"  {SEP_THIN}")
    print(f"  ANOMALIEN ({len(anomalies)}):")
    for a in anomalies:
        print(f"     {a['date']}  {fmt_amount(abs(a['amount']))}  [{a['type']}]  {a['description'][:40]}")


def print_no_pattern(txs: list[dict]):
    print(f"\n{'=' * 70}")
    print(f"  TRANSAKTIONEN OHNE MUSTER  ({len(txs)})")
    print(f"{'=' * 70}")
    for t in txs:
        sign = "+" if t["betrag"] > 0 else ""
        print(f"  {t['datum']}  {sign}{t['betrag']:>10,.2f} CHF  {t['gegenpartei'][:30]:<30}  {t['category_level1']}")


def print_summary(rec, sea, seq, no_pat):
    total = len(rec) + len(sea) + len(seq)
    tx_in = (sum(p["recurrence_sample_size"] for p in rec + sea)
             + sum(p["seq_pair_count"] * 2 for p in seq))
    print(f"\n{'=' * 70}")
    print(f"  ZUSAMMENFASSUNG")
    print(f"{'=' * 70}")
    print(f"  RECURRING  Muster : {len(rec)}")
    print(f"  SEASONAL   Muster : {len(sea)}")
    print(f"  SEQUENTIAL Muster : {len(seq)}")
    print(f"  Gesamt Muster     : {total}")
    print(f"  Tx in Mustern     : {tx_in}")
    print(f"  Tx ohne Muster    : {len(no_pat)}")


# ─────────────────────────────────────────────
# 13. MAIN (CLI)
# ─────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tink_categorized.json"

    print(f"\n{'=' * 70}")
    print(f"  PATTERN DETECTION  -  {path}")
    print(f"{'=' * 70}")

    transactions = load_transactions(path)
    rec, sea, seq, no_pat = _run_analysis(transactions)

    if rec:
        print(f"\n\033[94m>>>  RECURRING PATTERNS  ({len(rec)})\033[0m")
        for i, p in enumerate(rec, 1):
            print_pattern_header(p, i, len(rec))
            print_common_fields(p)
            print_recurring_fields(p)
            print_anomalies(p)

    if sea:
        print(f"\n\033[95m>>>  SEASONAL PATTERNS  ({len(sea)})\033[0m")
        for i, p in enumerate(sea, 1):
            print_pattern_header(p, i, len(sea))
            print_common_fields(p)
            print_recurring_fields(p)
            print_anomalies(p)

    if seq:
        print(f"\n\033[93m>>>  SEQUENTIAL PATTERNS  ({len(seq)})\033[0m")
        for i, p in enumerate(seq, 1):
            print_pattern_header(p, i, len(seq))
            print_common_fields(p)
            print_sequential_fields(p)

    print_no_pattern(no_pat)
    print_summary(rec, sea, seq, no_pat)

    print(f"\n{'=' * 70}")
    print(f"  Firestore speichern...")
    fs = _save_to_firestore(rec, sea, seq, no_pat)
    print(f"  ✅ patterns_db     : {fs['total_patterns']} Dokumente")
    print(f"  ✅ distributions_db: {fs['distributions']} Dokumente")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
