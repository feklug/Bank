"""
detect_patterns.py
==================
Erkennt Muster (RECURRING, SEASONAL, SEQUENTIAL) in Banktransaktionen.
Integriert österreichischen Feiertagskalender (bundesweit) für next_expected_date.

Ausgabe:
  - Konsole: strukturierte Ausgabe (wie bisher, vollständig)
  - JSON:    data/tink_patterns.json  (für Folge-Skripte)

Konfiguration via Env-Variablen:
  INPUT_FILE   Pfad zur kategorisierten JSON  (default: tink_categorized.json)
  OUTPUT_FILE  Pfad für Pattern-Output JSON   (default: tink_patterns.json)
"""

import json
import os
import pathlib
import sys
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict, Counter
from statistics import mean, stdev, linear_regression
import re

# ─────────────────────────────────────────────
# KONFIGURATION  (Env-Variablen überschreiben Defaults)
# ─────────────────────────────────────────────

_ROOT       = pathlib.Path(__file__).parent        # training/
_DATA       = _ROOT.parent / "data"                # data/

INPUT_FILE  = os.environ.get("INPUT_FILE",  str(_DATA / "tink_categorized.json"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", str(_DATA / "tink_patterns.json"))


# ─────────────────────────────────────────────
# 1. ÖSTERREICHISCHER FEIERTAGSKALENDER
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


def get_austrian_holidays(year: int) -> set[date]:
    """Gibt alle gesetzlichen Feiertage in Österreich (bundesweit) zurück."""
    easter = _easter(year)
    return {
        date(year,  1,  1),
        date(year,  1,  6),
        easter + timedelta(days=1),
        date(year,  5,  1),
        easter + timedelta(days=39),
        easter + timedelta(days=50),
        easter + timedelta(days=60),
        date(year,  8, 15),
        date(year, 10, 26),
        date(year, 11,  1),
        date(year, 12,  8),
        date(year, 12, 25),
        date(year, 12, 26),
    }


_holidays_cache: dict[int, set[date]] = {}


def next_banking_day(d: date) -> date:
    """Gibt den nächsten österreichischen Bankarbeitstag zurück."""
    current = d
    while True:
        year = current.year
        if year not in _holidays_cache:
            _holidays_cache[year] = get_austrian_holidays(year)
        if current.weekday() < 5 and current not in _holidays_cache[year]:
            return current
        current += timedelta(days=1)


def adjust_to_banking_day(d: date) -> date:
    return next_banking_day(d)


# ─────────────────────────────────────────────
# 2. NÄCHSTEN MONATSTAG BERECHNEN
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


def compute_typical_time(transactions: list[dict]) -> str | None:
    seconds_list = []
    for t in transactions:
        datum = t.get("datum", "")
        if "T" in datum:
            try:
                dt = datetime.fromisoformat(datum.replace("Z", "+00:00"))
                seconds_list.append(dt.hour * 3600 + dt.minute * 60 + dt.second)
            except ValueError:
                pass
    if not seconds_list:
        return None
    avg_sec = round(mean(seconds_list))
    h = avg_sec // 3600
    m = (avg_sec % 3600) // 60
    s = avg_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


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

def parse_datum(datum_str: str) -> tuple[date, str]:
    s = datum_str.strip()
    if "T" in s:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            display = dt.strftime("%Y-%m-%d %H:%M:%S")
            return dt.date(), display
        except ValueError:
            pass
    d = date.fromisoformat(s[:10])
    return d, s[:10]


def _clean_iban(raw) -> str | None:
    """Gibt None zurück wenn IBAN leer/null, sonst getrimmten String."""
    if raw is None or str(raw).strip().lower() in ("null", "none", ""):
        return None
    return str(raw).strip()


def load_transactions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for t in data:
        d, display = parse_datum(t["datum"])
        t["date_obj"]      = d
        t["datum_display"] = display
        t["iban"]          = _clean_iban(t.get("iban"))   # 'null/AT48...' → 'AT48...'
    return sorted(data, key=lambda x: x["date_obj"])


def group_key(t: dict) -> tuple:
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

    if best_label:
        in_tol     = sum(1 for g in gaps if abs(g - best_interval) <= best_tolerance)
        confidence = in_tol / len(gaps)
        return best_label, best_interval, round(confidence, 2)

    MULTI_BASES = [
        (30,  "MONTHLY",    5),
        (7,   "WEEKLY",     2),
        (90,  "QUARTERLY", 10),
        (365, "ANNUAL",    30),
    ]
    for base_days, label, base_tol in MULTI_BASES:
        ok = 0
        for g in gaps:
            n = round(g / base_days)
            if n < 1:
                continue
            if abs(g - n * base_days) <= base_tol * n:
                ok += 1
        conf = ok / len(gaps)
        if conf >= 0.60:
            return label, base_days, round(conf, 2)

    best_interval  = round(avg_gap)
    best_tolerance = max(5, round(avg_gap * 0.18))
    in_tol         = sum(1 for g in gaps if abs(g - best_interval) <= best_tolerance)
    confidence     = in_tol / len(gaps)
    return "CUSTOM", best_interval, round(confidence, 2)


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
    ibans     = [(i, t.get("iban") or "") for i, t in enumerate(transactions)]
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
# 6. KOMBINIERTE KONFIDENZ
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

    ibans     = [t.get("iban") or "" for t in transactions]
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
    typical_time  = compute_typical_time(transactions)
    next_expected_str = (
        f"{next_expected.isoformat()} {typical_time}" if next_expected and typical_time
        else next_expected.isoformat() if next_expected
        else "-"
    )

    amount_anomaly_idxs = set(detect_amount_anomalies(amounts))
    date_anomaly_idxs   = set(detect_date_anomalies(dates, interval_days or 30))
    iban_anomaly_idxs   = set(detect_iban_anomalies(transactions))
    duplicate_idxs      = set(detect_duplicates(transactions))

    all_anomaly_idxs = amount_anomaly_idxs | date_anomaly_idxs | iban_anomaly_idxs | duplicate_idxs
    anomalies = []
    for idx in sorted(all_anomaly_idxs):
        t = transactions[idx]
        anomaly_type = []
        if idx in duplicate_idxs:
            anomaly_type.append("DUPLICATE")
        if idx in iban_anomaly_idxs:
            anomaly_type.append("IBAN")
        if idx in amount_anomaly_idxs:
            anomaly_type.append("AMOUNT")
        if idx in date_anomaly_idxs:
            anomaly_type.append("DATE")
        anomalies.append({
            "date":        t["datum_display"],
            "amount":      t["betrag"],
            "type":        "+".join(anomaly_type),
            "description": t["verwendungszweck"],
        })

    has_gaps = False
    if interval_days and len(dates) > 1:
        gaps     = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        has_gaps = any(g > interval_days * 1.8 for g in gaps)

    amount_std   = round(stdev(amounts), 2) if len(amounts) >= 2 else 0.0
    amount_trend = compute_amount_trend(amounts)

    sample         = transactions[0]
    ibans_in_group = [t.get("iban") or "" for t in transactions]
    non_empty      = [ib for ib in ibans_in_group if ib]
    dominant_iban  = non_empty[0] if non_empty else "-"
    if non_empty and len(set(non_empty)) > 1:
        dominant_iban = Counter(non_empty).most_common(1)[0][0] + "  WECHSEL"

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
        "first_seen":                 transactions[min(range(len(dates)), key=lambda i: dates[i])]["datum_display"],
        "last_seen":                  transactions[max(range(len(dates)), key=lambda i: dates[i])]["datum_display"],
        "next_expected_date":         next_expected_str,
        "recurrence_interval":        f"{interval_label} (~{interval_days}d)" if interval_label else "UNBEKANNT",
        "recurrence_interval_days":   interval_days,
        "recurrence_date_confidence": round(date_conf, 2),
        "recurrence_confidence":      fcast_conf,
        "recurrence_day_of_month":    anchor_dom,
        "recurrence_day_of_week":     dow,
        "recurrence_has_gaps":        has_gaps,
        "recurrence_sample_size":     len(transactions),
        "anomalies":                  anomalies,
        "_transactions":              transactions,
    }

    if seasonal:
        months_seen            = sorted(set(d.month for d in dates))
        month_names            = ["Jan","Feb","Mar","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
        result["seasonal_months"] = [month_names[m - 1] for m in months_seen]
        result["seasonal_years"]  = sorted(set(d.year for d in dates))

    return result


# ─────────────────────────────────────────────
# 8. SEQUENTIAL PATTERN ERKENNUNG
# ─────────────────────────────────────────────

SEQUENTIAL_RULES = [
    {
        "name":             "Lohnzahlung -> Sozialversicherung",
        "trigger_category": "AUSGABEN - PERSONAL",
        "follow_category":  "AUSGABEN - SOZIALVERSICHERUNGEN",
        "max_days":         10,
        "description":      "Lohnzahlungen lösen typischerweise SV-Beiträge aus",
    },
    {
        "name":             "Quartalsabrechnung -> Steuerüberweisung",
        "trigger_keyword":  "Q",
        "trigger_category": "AUSGABEN - BETRIEBSKOSTEN",
        "follow_category":  "AUSGABEN - STEUERN & ABGABEN",
        "max_days":         15,
        "description":      "Quartalsabrechnung führt oft zu Steuerzahlungen",
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
        "trigger_category": "AUSGABEN - INVESTITIONEN",
        "follow_category":  "AUSGABEN - FINANZEN & BANKING",
        "max_days":         30,
        "description":      "Investitionen können Kreditkosten auslösen",
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
        typical_time      = compute_typical_time(all_txs)
        next_expected_str = (
            f"{next_expected.isoformat()} {typical_time}" if typical_time
            else next_expected.isoformat()
        )

        first_tx = min(all_txs, key=lambda t: t["date_obj"])
        last_tx  = max(all_txs, key=lambda t: t["date_obj"])

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
            "first_seen":              first_tx["datum_display"],
            "last_seen":               last_tx["datum_display"],
            "next_expected_date":      next_expected_str,
            "seq_pair_count":          len(matched_pairs),
            "seq_avg_delay_days":      avg_delay,
            "seq_trigger_avg_amount":  round(mean(amounts_trigger), 2),
            "seq_follow_avg_amount":   round(mean(amounts_follow), 2),
            "seq_trigger_category":    rule["trigger_category"],
            "seq_follow_category":     rule["follow_category"],
            "seq_consistency_score":   round(min(1.0, len(matched_pairs) / max(len(triggers), 1)), 2),
            "seq_amount_ratio":        round(mean(amounts_follow) / mean(amounts_trigger), 4) if mean(amounts_trigger) > 0 else None,
            "anomalies":               detect_amount_anomalies(all_amounts),
            "_idxs":                   all_idxs,
            "_transactions":           all_txs,
        }
        patterns.append(pattern)

    return patterns


# ─────────────────────────────────────────────
# 9. HAUPT-ANALYSE
# ─────────────────────────────────────────────

MIN_DATE_CONF: dict[str, float] = {
    "CUSTOM":  0.55,
    "DEFAULT": 0.30,
}
MIN_PATTERN_DATE_CONF:     float = 0.60
MIN_PATTERN_FORECAST_CONF: float = 0.60


def analyze(path: str):
    transactions = load_transactions(path)
    used_ids: set[int] = set()

    patterns_recurring:  list[dict] = []
    patterns_seasonal:   list[dict] = []
    patterns_sequential: list[dict] = []

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

    seq_patterns = find_sequential_patterns(transactions, used_ids)
    for p in seq_patterns:
        for i in p["_idxs"]:
            used_ids.add(i)
    patterns_sequential = seq_patterns

    no_pattern = [transactions[i] for i in range(len(transactions)) if i not in used_ids]

    return patterns_recurring, patterns_seasonal, patterns_sequential, no_pattern


# ─────────────────────────────────────────────
# 10. JSON EXPORT  (für Folge-Skripte)
# ─────────────────────────────────────────────

def _serialize(obj):
    """Konvertiert date-Objekte und andere nicht-JSON-serialisierbare Typen."""
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _clean_pattern(p: dict) -> dict:
    """Entfernt interne Felder (_transactions, _idxs, date_obj) vor dem Export."""
    clean = {k: v for k, v in p.items() if not k.startswith("_")}

    # Transaktionen ohne interne Felder exportieren
    if "_transactions" in p:
        clean["transactions"] = [
            {k2: v2 for k2, v2 in t.items() if k2 not in ("date_obj",)}
            for t in p["_transactions"]
        ]
    return clean


def save_output(
    patterns_recurring:  list[dict],
    patterns_seasonal:   list[dict],
    patterns_sequential: list[dict],
    no_pattern:          list[dict],
    path: str,
):
    """
    Speichert alle Pattern-Ergebnisse als JSON.
    Format ist kompatibel mit Folge-Skripten (z.B. forecast.py, report.py).

    Struktur:
    {
      "meta": { "generated_at": "...", "input_file": "..." },
      "summary": { "recurring": N, "seasonal": N, "sequential": N, ... },
      "recurring":  [ { pattern-Felder + transactions: [...] } ],
      "seasonal":   [ ... ],
      "sequential": [ ... ],
      "no_pattern": [ { datum, betrag, verwendungszweck, gegenpartei, ... } ]
    }
    """
    output = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "input_file":   path,
        },
        "summary": {
            "recurring":       len(patterns_recurring),
            "seasonal":        len(patterns_seasonal),
            "sequential":      len(patterns_sequential),
            "no_pattern":      len(no_pattern),
            "total_patterns":  len(patterns_recurring) + len(patterns_seasonal) + len(patterns_sequential),
        },
        "recurring":  [_clean_pattern(p) for p in patterns_recurring],
        "seasonal":   [_clean_pattern(p) for p in patterns_seasonal],
        "sequential": [_clean_pattern(p) for p in patterns_sequential],
        "no_pattern": [
            {k: v for k, v in t.items() if k not in ("date_obj",)}
            for t in no_pattern
        ],
    }

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=_serialize)

    print(f"\n✅  Gespeichert: {path}")
    print(f"    → Folge-Skript kann Daten laden mit:")
    print(f"      data = json.load(open('{path}'))")
    print(f"      recurring   = data['recurring']")
    print(f"      seasonal    = data['seasonal']")
    print(f"      sequential  = data['sequential']")
    print(f"      no_pattern  = data['no_pattern']")


# ─────────────────────────────────────────────
# 11. KONSOLEN-AUSGABE  (unverändert)
# ─────────────────────────────────────────────

SEP      = "-" * 70
SEP_THIN = "." * 70


def fmt_amount(a: float) -> str:
    return f"EUR {a:>10,.2f}"


def fmt_trend(slope: float | None) -> str:
    if slope is None:
        return "-"
    sign = "+" if slope > 0 else ("-" if slope < 0 else "=")
    return f"{sign} {abs(slope):,.4f} EUR/Periode"


def print_pattern_header(p: dict, idx: int, total: int):
    pt = p["pattern_type"]
    color_map = {
        "RECURRING":  "\033[94m",
        "SEASONAL":   "\033[95m",
        "SEQUENTIAL": "\033[93m",
    }
    reset = "\033[0m"
    color = color_map.get(pt, "")
    print(f"\n{SEP}")
    print(f"  {color}[{pt}]{reset}  #{idx}/{total}  {p.get('sequence_name', p['gegenpartei'])}")
    print(SEP)


def print_common_fields(p: dict):
    print(f"  Gegenpartei    : {p['gegenpartei']}")
    print(f"  IBAN           : {p['iban']}")
    print(f"  Kategorie      : {p['category']}")
    print(f"  Zeitraum       : {p['first_seen']}  ->  {p['last_seen']}")
    print(f"  Nächstes Dat.  : \033[92m{p['next_expected_date']}\033[0m  (AT Bankarbeitstag)")
    print(f"  {SEP_THIN}")
    print(f"  Betrag Ø       : {fmt_amount(p['amount_avg'])}")
    print(f"  Betrag Sigma   : {fmt_amount(p.get('amount_std', 0.0))}")
    print(f"  Betrag Min     : {fmt_amount(p['amount_min'])}")
    print(f"  Betrag Max     : {fmt_amount(p['amount_max'])}")
    print(f"  Betrag Summe   : {fmt_amount(p['amount_sum'])}")
    trend = p.get("amount_trend_per_period")
    if trend is not None:
        print(f"  Betrag Trend   : {fmt_trend(trend)}")


def print_recurring_fields(p: dict):
    print(f"  {SEP_THIN}")
    print(f"  Intervall      : {p['recurrence_interval']}")
    print(f"  Datum-Konf.    : {p['recurrence_date_confidence']*100:.0f}%  (Erkennung)")
    print(f"  Forecast-Konf. : {p['recurrence_confidence']*100:.0f}%  (Datum + Betrag + IBAN)")
    print(f"  Stichproben    : {p['recurrence_sample_size']}")
    print(f"  Tag im Monat   : {p['recurrence_day_of_month']}")
    print(f"  Wochentag      : {p['recurrence_day_of_week']}")
    print(f"  Lücken vorh.   : {'JA' if p['recurrence_has_gaps'] else 'Nein'}")
    if p.get("seasonal_months"):
        print(f"  Saisonmonate   : {', '.join(p['seasonal_months'])}")
    if p.get("seasonal_years"):
        print(f"  Saisonstück    : {p['seasonal_years']}")


def print_sequential_fields(p: dict):
    print(f"  {SEP_THIN}")
    print(f"  Ø Verzögerung    : {p['seq_avg_delay_days']} Tage")
    print(f"  Paare erkannt    : {p['seq_pair_count']}")
    print(f"  Trigger-Kat.     : {p['seq_trigger_category']}")
    print(f"  Folge-Kat.       : {p['seq_follow_category']}")
    print(f"  Trigger-Ø Betr.  : {fmt_amount(p['seq_trigger_avg_amount'])}")
    print(f"  Folge-Ø Betrag   : {fmt_amount(p['seq_follow_avg_amount'])}")
    if p["seq_amount_ratio"] is not None:
        print(f"  Betrag-Verh.     : {p['seq_amount_ratio']:.4f}  (Folge/Trigger)")
    print(f"  Konsistenz       : {p['seq_consistency_score']*100:.0f}%")
    print(f"  Beschreibung     : {p['sequence_description']}")


def print_anomalies(p: dict):
    anomalies = p.get("anomalies", [])
    if not isinstance(anomalies, list) or not anomalies:
        return
    if not isinstance(anomalies[0], dict):
        return
    print(f"  {SEP_THIN}")
    print(f"  ANOMALIEN ({len(anomalies)}):")
    for a in anomalies:
        print(f"     {a['date']:<22}  {fmt_amount(abs(a['amount']))}  [{a['type']}]  {a['description'][:40]}")


def print_no_pattern(txs: list[dict]):
    print(f"\n{'=' * 70}")
    print(f"  TRANSAKTIONEN OHNE MUSTER  ({len(txs)})")
    print(f"{'=' * 70}")
    for t in txs:
        sign = "+" if t["betrag"] > 0 else ""
        print(
            f"  {t['datum_display']:<22}  {sign}{t['betrag']:>10,.2f} EUR  "
            f"{t['gegenpartei'][:28]:<28}  {t['category_level1']}"
        )


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
# 12. MAIN
# ─────────────────────────────────────────────

def main():
    # CLI-Argument hat Vorrang vor Env-Variable (Rückwärtskompatibilität)
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE

    if not os.path.exists(path):
        print(f"\n❌  Input-Datei nicht gefunden: {path}")
        raise SystemExit(1)

    print(f"\n{'=' * 70}")
    print(f"  PATTERN DETECTION  -  {path}")
    print(f"  Output            ->  {OUTPUT_FILE}")
    print(f"{'=' * 70}")

    rec, sea, seq, no_pat = analyze(path)

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

    # ── JSON speichern (für Folge-Skripte) ───────────────────────
    save_output(rec, sea, seq, no_pat, OUTPUT_FILE)

    print(f"\n{'─'*70}")
    print(f"  Nächster Schritt: python3 scripts/forecast.py")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()
