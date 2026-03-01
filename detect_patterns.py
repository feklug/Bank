"""
detect_patterns.py
==================
Erkennt Muster (RECURRING, SEASONAL, SEQUENTIAL) in Banktransaktionen.
Speichert Ergebnisse direkt in Firestore:
  - patterns_db      -> jedes Pattern als eigenes Dokument (inkl. Transaktionen)
  - distributions_db -> jede Transaktion ohne Muster als eigenes Dokument

NEU: bookedDateTime wird durchgaengig mitgefuehrt und ausgewertet.
     avg_booking_hour / booking_hour_std pro Pattern fuer 120h-Forecasting.
"""

import json
import re
import sys
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from statistics import linear_regression, mean, stdev

# ─────────────────────────────────────────────
# 1. OESTERREICHISCHER FEIERTAGSKALENDER
# ─────────────────────────────────────────────

def _easter(year: int) -> date:
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
    day   = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def get_austrian_holidays(year: int) -> set[date]:
    easter = _easter(year)
    return {
        date(year, 1,  1),                       # Neujahr
        date(year, 1,  6),                       # Heilige Drei Koenige
        easter - timedelta(days=2),              # Karfreitag
        easter,                                  # Ostersonntag
        easter + timedelta(days=1),              # Ostermontag
        date(year, 5,  1),                       # Staatsfeiertag
        easter + timedelta(days=39),             # Christi Himmelfahrt
        easter + timedelta(days=49),             # Pfingstsonntag
        easter + timedelta(days=50),             # Pfingstmontag
        easter + timedelta(days=60),             # Fronleichnam
        date(year, 8, 15),                       # Mariae Himmelfahrt
        date(year, 10, 26),                      # Nationalfeiertag
        date(year, 11,  1),                      # Allerheiligen
        date(year, 12,  8),                      # Mariae Empfaengnis
        date(year, 12, 25),                      # Weihnachten
        date(year, 12, 26),                      # Stephanstag
    }


_holidays_cache: dict[int, set[date]] = {}


def next_banking_day(d: date) -> date:
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
# 2. NAECHSTEN MONATSTAG BERECHNEN
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

def _parse_booked_dt(raw: str) -> datetime | None:
    """Parst bookedDateTime ISO 8601 -> datetime. Gibt None bei leerem/fehlendem Wert."""
    if not raw:
        return None
    try:
        # Unterstuetzt: "2024-01-05T08:23:41Z" und "2024-01-05T08:23:41+00:00"
        s = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        # Timezone-Info entfernen fuer einfachere Verarbeitung
        return dt.replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def _prepare_transactions(data: list[dict]) -> list[dict]:
    """
    Fuegt date_obj (date) und datetime_obj (datetime | None) hinzu.
    Sortiert chronologisch nach bookedDateTime, Fallback auf datum.
    Veraendert die Originaldaten nicht.
    """
    result = []
    for t in data:
        entry              = dict(t)
        entry["date_obj"]  = date.fromisoformat(t["datum"])
        entry["datetime_obj"] = _parse_booked_dt(t.get("bookedDateTime", ""))
        result.append(entry)

    # Sortierung: bookedDateTime wenn vorhanden, sonst datum
    return sorted(result, key=lambda x: (
        x["datetime_obj"] if x["datetime_obj"] else
        datetime(x["date_obj"].year, x["date_obj"].month, x["date_obj"].day)
    ))


def load_transactions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _prepare_transactions(data)


def group_key(t: dict) -> tuple:
    iban = t.get("iban") or ""
    cat  = t["category_level1"]
    if iban:
        return ("__IBAN__", iban, cat)
    return (t["gegenpartei"], cat, "")


# ─────────────────────────────────────────────
# 4. UHRZEIT-ANALYSE (neu)
# ─────────────────────────────────────────────

def compute_booking_time_stats(transactions: list[dict]) -> dict:
    """
    Berechnet Buchungszeit-Statistiken aus bookedDateTime.
    Gibt avg_booking_hour, booking_hour_std und next_expected_time zurueck.
    Relevant fuer 120h-Liquiditaets-Forecasting.

    Returns dict mit:
        avg_booking_hour      : float | None  (z.B. 6.5 = 06:30 Uhr)
        booking_hour_std      : float | None  (Standardabweichung der Stunde)
        booking_minute_avg    : int | None    (Durchschnittsminute)
        next_expected_time    : str | None    (z.B. "06:30")
        time_data_available   : bool
    """
    hours   = []
    minutes = []

    for t in transactions:
        dt = t.get("datetime_obj")
        if dt:
            hours.append(dt.hour + dt.minute / 60.0)  # z.B. 6.5 fuer 06:30
            minutes.append(dt.minute)

    if not hours:
        return {
            "avg_booking_hour":   None,
            "booking_hour_std":   None,
            "booking_minute_avg": None,
            "next_expected_time": None,
            "time_data_available": False,
        }

    avg_h = mean(hours)
    std_h = round(stdev(hours), 2) if len(hours) >= 2 else 0.0
    h_int = int(avg_h)
    m_int = round((avg_h - h_int) * 60)
    if m_int >= 60:
        h_int += 1
        m_int  = 0

    return {
        "avg_booking_hour":   round(avg_h, 2),
        "booking_hour_std":   std_h,
        "booking_minute_avg": round(mean(minutes)) if minutes else None,
        "next_expected_time": f"{h_int:02d}:{m_int:02d}",
        "time_data_available": True,
    }


# ─────────────────────────────────────────────
# 5. INTERVALL-ERKENNUNG
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
    confidence   = in_tolerance / len(gaps)
    return best_label, best_interval, round(confidence, 2)


def is_seasonal(dates: list[date], interval_label: str | None) -> bool:
    if len(dates) < 2:
        return False
    if interval_label in ("MONTHLY", "WEEKLY", "BIWEEKLY"):
        return False
    month_set = set(d.month for d in dates)
    return len(month_set) <= max(2, len(dates) // 2)


# ─────────────────────────────────────────────
# 6. ANOMALIE-ERKENNUNG
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
# 7. KOMBINIERTE KONFIDENZ
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
    conf  = 0.6 * date_conf + 0.4 * amount_conf
    ibans = [t.get("iban") or "" for t in transactions]
    non_empty = [ib for ib in ibans if ib]
    if non_empty and len(set(non_empty)) == 1:
        conf = min(1.0, conf + 0.15)
    if len(amounts) >= 3 and sd == 0.0:
        conf = min(1.0, conf + 0.10)
    return round(conf, 2)


# ─────────────────────────────────────────────
# 8. RECURRING PATTERN BUILDER
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
    dow_names  = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"]
    dow        = dow_names[dow_num] if dow_num is not None else None

    fcast_conf    = forecast_confidence(date_conf, amounts, transactions)
    next_expected = compute_next_expected(dates, interval_label, interval_days, anchor_dom)

    # Uhrzeit-Statistiken
    time_stats = compute_booking_time_stats(transactions)

    # next_expected_datetime: Datum + erwartete Uhrzeit kombinieren
    next_expected_datetime = None
    if next_expected and time_stats["next_expected_time"]:
        next_expected_datetime = f"{next_expected.isoformat()}T{time_stats['next_expected_time']}:00"

    amount_anomaly_idxs = set(detect_amount_anomalies(amounts))
    date_anomaly_idxs   = set(detect_date_anomalies(dates, interval_days or 30))
    iban_anomaly_idxs   = set(detect_iban_anomalies(transactions))
    duplicate_idxs      = set(detect_duplicates(transactions))
    all_anomaly_idxs    = amount_anomaly_idxs | date_anomaly_idxs | iban_anomaly_idxs | duplicate_idxs

    anomalies = []
    for idx in sorted(all_anomaly_idxs):
        t            = transactions[idx]
        anomaly_type = []
        if idx in duplicate_idxs:      anomaly_type.append("DUPLICATE")
        if idx in iban_anomaly_idxs:   anomaly_type.append("IBAN")
        if idx in amount_anomaly_idxs: anomaly_type.append("AMOUNT")
        if idx in date_anomaly_idxs:   anomaly_type.append("DATE")
        anomalies.append({
            "date":          t["datum"],
            "bookedDateTime": t.get("bookedDateTime", ""),
            "amount":        t["betrag"],
            "type":          "+".join(anomaly_type),
            "description":   t["verwendungszweck"],
        })

    has_gaps = False
    if interval_days and len(dates) > 1:
        gaps     = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        has_gaps = any(g > interval_days * 1.8 for g in gaps)

    amount_std   = round(stdev(amounts), 2) if len(amounts) >= 2 else 0.0
    amount_trend = compute_amount_trend(amounts)

    sample        = transactions[0]
    ibans_in_grp  = [t.get("iban") or "" for t in transactions]
    non_empty_ib  = [ib for ib in ibans_in_grp if ib]
    dominant_iban = non_empty_ib[0] if non_empty_ib else "-"
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
        "next_expected_datetime":     next_expected_datetime or "-",   # NEU: Datum + Uhrzeit
        "recurrence_interval":        f"{interval_label} (~{interval_days}d)" if interval_label else "UNBEKANNT",
        "recurrence_interval_days":   interval_days,
        "recurrence_date_confidence": round(date_conf, 2),
        "recurrence_confidence":      fcast_conf,
        "recurrence_day_of_month":    anchor_dom,
        "recurrence_day_of_week":     dow,
        "recurrence_has_gaps":        has_gaps,
        "recurrence_sample_size":     len(transactions),
        # Uhrzeit-Felder (NEU)
        "avg_booking_hour":           time_stats["avg_booking_hour"],
        "booking_hour_std":           time_stats["booking_hour_std"],
        "booking_minute_avg":         time_stats["booking_minute_avg"],
        "next_expected_time":         time_stats["next_expected_time"],
        "time_data_available":        time_stats["time_data_available"],
        "anomalies":                  anomalies,
        "_transactions":              transactions,
    }

    if seasonal:
        months_seen          = sorted(set(d.month for d in dates))
        month_names          = ["Jan","Feb","Mar","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
        result["seasonal_months"] = [month_names[m - 1] for m in months_seen]
        result["seasonal_years"]  = sorted(set(d.year for d in dates))

    return result


# ─────────────────────────────────────────────
# 9. SEQUENTIAL PATTERN ERKENNUNG
# ─────────────────────────────────────────────

SEQUENTIAL_RULES = [
    {
        "name":             "Lohnzahlung -> Sozialversicherung",
        "trigger_category": "AUSGABEN - PERSONAL",
        "follow_category":  "AUSGABEN - SOZIALVERSICHERUNGEN",
        "max_days":         10,
        "description":      "Lohnzahlungen loesen typischerweise SV-Beitraege aus",
    },
    {
        "name":             "Quartalsabrechnung -> Steuerueberweisung",
        "trigger_keyword":  "Q",
        "trigger_category": "AUSGABEN - BETRIEBSKOSTEN",
        "follow_category":  "AUSGABEN - STEUERN & ABGABEN",
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
        "trigger_category": "AUSGABEN - INVESTITIONEN",
        "follow_category":  "AUSGABEN - FINANZEN & BANKING",
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

        # Uhrzeit-Statistiken fuer Trigger und Follow getrennt
        time_stats_trigger = compute_booking_time_stats([p["trigger"] for p in matched_pairs])
        time_stats_follow  = compute_booking_time_stats([p["follow"]  for p in matched_pairs])

        last_trigger_date = max(p["trigger"]["date_obj"] for p in matched_pairs)
        next_expected     = adjust_to_banking_day(last_trigger_date + timedelta(days=round(avg_delay)))

        # next_expected_datetime mit Follow-Uhrzeit
        next_expected_datetime = None
        if time_stats_follow["next_expected_time"]:
            next_expected_datetime = f"{next_expected.isoformat()}T{time_stats_follow['next_expected_time']}:00"

        pattern = {
            "pattern_type":               "SEQUENTIAL",
            "sequence_name":              rule["name"],
            "sequence_description":       rule["description"],
            "gegenpartei":                f"{matched_pairs[0]['trigger']['gegenpartei']} -> {matched_pairs[0]['follow']['gegenpartei']}",
            "iban":                       "-",
            "category":                   f"{rule['trigger_category']} -> {rule['follow_category']}",
            "amount_avg":                 round(mean(all_amounts), 2),
            "amount_std":                 round(stdev(all_amounts), 2) if len(all_amounts) >= 2 else 0.0,
            "amount_min":                 round(min(all_amounts), 2),
            "amount_max":                 round(max(all_amounts), 2),
            "amount_sum":                 round(sum(all_amounts), 2),
            "amount_trend_per_period":    compute_amount_trend(all_amounts),
            "first_seen":                 min(dates).isoformat(),
            "last_seen":                  max(dates).isoformat(),
            "next_expected_date":         next_expected.isoformat(),
            "next_expected_datetime":     next_expected_datetime or "-",   # NEU
            "seq_pair_count":             len(matched_pairs),
            "seq_avg_delay_days":         avg_delay,
            "seq_trigger_avg_amount":     round(mean(amounts_trigger), 2),
            "seq_follow_avg_amount":      round(mean(amounts_follow), 2),
            "seq_trigger_category":       rule["trigger_category"],
            "seq_follow_category":        rule["follow_category"],
            "seq_consistency_score":      round(min(1.0, len(matched_pairs) / max(len(triggers), 1)), 2),
            "seq_amount_ratio":           round(mean(amounts_follow) / mean(amounts_trigger), 4) if mean(amounts_trigger) > 0 else None,
            # Uhrzeit-Felder (NEU)
            "trigger_avg_booking_hour":   time_stats_trigger["avg_booking_hour"],
            "trigger_next_expected_time": time_stats_trigger["next_expected_time"],
            "follow_avg_booking_hour":    time_stats_follow["avg_booking_hour"],
            "follow_next_expected_time":  time_stats_follow["next_expected_time"],
            "time_data_available":        time_stats_follow["time_data_available"],
            "anomalies":                  detect_amount_anomalies(all_amounts),
            "_idxs":                      all_idxs,
            "_transactions":              all_txs,
        }
        patterns.append(pattern)
    return patterns


# ─────────────────────────────────────────────
# 10. HAUPT-ANALYSE
# ─────────────────────────────────────────────

MIN_DATE_CONF: dict[str, float] = {
    "CUSTOM":  0.70,
    "DEFAULT": 0.40,
}
MIN_PATTERN_DATE_CONF:     float = 0.75
MIN_PATTERN_FORECAST_CONF: float = 0.75


def _run_analysis(transactions: list[dict]) -> tuple[list, list, list, list]:
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
# 11. FIRESTORE-INTEGRATION
# ─────────────────────────────────────────────

def _make_doc_id(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)
    return safe[:200]


def _clean_transaction(t: dict) -> dict:
    """
    Entfernt interne Felder (date_obj, datetime_obj, _*) fuer Firestore.
    bookedDateTime bleibt erhalten.
    """
    return {
        k: v for k, v in t.items()
        if not k.startswith("_") and k not in ("date_obj", "datetime_obj")
    }


def _clean_pattern(p: dict) -> tuple[dict, list[dict]]:
    transactions = [_clean_transaction(t) for t in p.get("_transactions", [])]
    pattern_doc  = {
        k: v for k, v in p.items()
        if not k.startswith("_")
    }
    return pattern_doc, transactions


def _init_firestore():
    import firebase_admin
    from firebase_admin import credentials, firestore

    SERVICE_ACCOUNT_FILE = "bank-417a7-firebase-adminsdk-fbsvc-60ba2be615.json"

    if not firebase_admin._apps:
        import os
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            raise RuntimeError(
                f"Service-Account-Datei nicht gefunden: {SERVICE_ACCOUNT_FILE}\n"
                "GitHub Action: Secrets -> FIREBASE_SERVICE_ACCOUNT muss gesetzt sein."
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
    Speichert alle Ergebnisse in Firestore.

    patterns_db:
      ID:     {PATTERN_TYPE}_{gegenpartei}_{first_seen}
      Felder: alle Pattern-Metadaten inkl. Uhrzeit-Felder
      Feld:   "transactions" (Array mit bookedDateTime je Tx)

    distributions_db:
      ID:     {datum}_{bookedDateTime}_{gegenpartei}_{betrag}
      Felder: alle Transaktionsfelder inkl. bookedDateTime

    Alle Schreibvorgaenge: set() -> idempotent.
    """
    db = _init_firestore()
    patterns_ref      = db.collection("patterns_db")
    distributions_ref = db.collection("distributions_db")

    total_patterns = 0

    for p in patterns_recurring + patterns_seasonal + patterns_sequential:
        pattern_doc, transactions = _clean_pattern(p)

        raw_id = f"{p['pattern_type']}_{p['gegenpartei']}_{p['first_seen']}"
        doc_id = _make_doc_id(raw_id)

        pattern_doc["transactions"] = transactions
        patterns_ref.document(doc_id).set(pattern_doc)
        total_patterns += 1

    distributions = 0
    for t in no_pattern:
        clean = _clean_transaction(t)

        # bookedDateTime in der Doc-ID fuer Eindeutigkeit auf Sekunden-Ebene
        bdt    = clean.get("bookedDateTime", "")
        raw_id = f"{clean['datum']}_{bdt}_{clean['gegenpartei']}_{clean['betrag']}"
        doc_id = _make_doc_id(raw_id)

        distributions_ref.document(doc_id).set(clean)
        distributions += 1

    return {"total_patterns": total_patterns, "distributions": distributions}


# ─────────────────────────────────────────────
# 12. PUBLIC API  (fuer pipeline.py)
# ─────────────────────────────────────────────

def detect_patterns(categorized: list[dict]) -> dict[str, int]:
    """
    Haupteinstiegspunkt fuer die Pipeline.

    Parameters:
        categorized : In-Memory-Liste aus categorize.categorize()
                      Pflichtfelder: datum, bookedDateTime, betrag,
                      gegenpartei, verwendungszweck, iban, category_level1

    Ablauf:
      1. date_obj + datetime_obj hinzufuegen, chronologisch sortieren
      2. Muster erkennen (RECURRING, SEASONAL, SEQUENTIAL)
      3. Patterns -> patterns_db in Firestore
      4. Ohne Muster -> distributions_db in Firestore

    Returns:
        {"total_patterns": int, "distributions": int}
    """
    transactions = _prepare_transactions(categorized)
    rec, sea, seq, no_pat = _run_analysis(transactions)

    print(f"    RECURRING  : {len(rec)}")
    print(f"    SEASONAL   : {len(sea)}")
    print(f"    SEQUENTIAL : {len(seq)}")
    print(f"    Ohne Muster: {len(no_pat)}")
    print(f"    -> Firestore speichern...")

    fs_result = _save_to_firestore(rec, sea, seq, no_pat)
    return fs_result


# ─────────────────────────────────────────────
# 13. KONSOLENAUSGABE (nur CLI)
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
    pt        = p["pattern_type"]
    color_map = {"RECURRING": "\033[94m", "SEASONAL": "\033[95m", "SEQUENTIAL": "\033[93m"}
    color     = color_map.get(pt, "")
    print(f"\n{SEP}")
    print(f"  {color}[{pt}]\033[0m  #{idx}/{total}  {p.get('sequence_name', p['gegenpartei'])}")
    print(SEP)


def print_common_fields(p: dict):
    print(f"  Gegenpartei    : {p['gegenpartei']}")
    print(f"  IBAN           : {p['iban']}")
    print(f"  Kategorie      : {p['category']}")
    print(f"  Zeitraum       : {p['first_seen']}  ->  {p['last_seen']}")
    print(f"  Naechstes Dat. : \033[92m{p['next_expected_date']}\033[0m")
    # Uhrzeit-Info
    if p.get("time_data_available"):
        print(f"  Naechste Zeit  : \033[92m{p.get('next_expected_time','?')}\033[0m  "
              f"(Oe Buchungsstunde: {p.get('avg_booking_hour','?'):.1f}, "
              f"Sigma: {p.get('booking_hour_std','?')}h)")
        print(f"  Naechster TS   : \033[92m{p.get('next_expected_datetime','-')}\033[0m")
    else:
        print(f"  Uhrzeit        : keine bookedDateTime-Daten verfuegbar")
    print(f"  {SEP_THIN}")
    print(f"  Betrag Oe      : {fmt_amount(p['amount_avg'])}")
    print(f"  Betrag Sigma   : {fmt_amount(p.get('amount_std', 0.0))}")
    print(f"  Betrag Min/Max : {fmt_amount(p['amount_min'])}  /  {fmt_amount(p['amount_max'])}")
    print(f"  Betrag Summe   : {fmt_amount(p['amount_sum'])}")
    if p.get("amount_trend_per_period") is not None:
        print(f"  Betrag Trend   : {fmt_trend(p['amount_trend_per_period'])}")


def print_recurring_fields(p: dict):
    print(f"  {SEP_THIN}")
    print(f"  Intervall      : {p['recurrence_interval']}")
    print(f"  Datum-Konf.    : {p['recurrence_date_confidence']*100:.0f}%")
    print(f"  Forecast-Konf. : {p['recurrence_confidence']*100:.0f}%")
    print(f"  Stichproben    : {p['recurrence_sample_size']}")
    print(f"  Tag im Monat   : {p['recurrence_day_of_month']}")
    print(f"  Wochentag      : {p['recurrence_day_of_week']}")
    print(f"  Luecken vorh.  : {'JA' if p['recurrence_has_gaps'] else 'Nein'}")
    if p.get("seasonal_months"):
        print(f"  Saisonmonate   : {', '.join(p['seasonal_months'])}")


def print_sequential_fields(p: dict):
    print(f"  {SEP_THIN}")
    print(f"  Oe Verzoegerung  : {p['seq_avg_delay_days']} Tage")
    print(f"  Paare erkannt    : {p['seq_pair_count']}")
    print(f"  Trigger-Kat.     : {p['seq_trigger_category']}")
    print(f"  Folge-Kat.       : {p['seq_follow_category']}")
    print(f"  Trigger-Oe Betr. : {fmt_amount(p['seq_trigger_avg_amount'])}")
    print(f"  Folge-Oe Betrag  : {fmt_amount(p['seq_follow_avg_amount'])}")
    if p.get("time_data_available"):
        print(f"  Trigger-Uhrzeit  : {p.get('trigger_next_expected_time','?')}  "
              f"(Oe: {p.get('trigger_avg_booking_hour','?')}h)")
        print(f"  Folge-Uhrzeit    : {p.get('follow_next_expected_time','?')}  "
              f"(Oe: {p.get('follow_avg_booking_hour','?')}h)")
    if p.get("seq_amount_ratio") is not None:
        print(f"  Betrag-Verh.     : {p['seq_amount_ratio']:.4f}")
    print(f"  Konsistenz       : {p['seq_consistency_score']*100:.0f}%")
    print(f"  Beschreibung     : {p['sequence_description']}")


def print_anomalies(p: dict):
    anomalies = p.get("anomalies", [])
    if not isinstance(anomalies, list) or not anomalies or not isinstance(anomalies[0], dict):
        return
    print(f"  {SEP_THIN}")
    print(f"  ANOMALIEN ({len(anomalies)}):")
    for a in anomalies:
        bdt = a.get("bookedDateTime", "")
        bdt_str = f"  [{bdt}]" if bdt else ""
        print(f"     {a['date']}{bdt_str}  {fmt_amount(abs(a['amount']))}  [{a['type']}]  {a['description'][:40]}")


def print_no_pattern(txs: list[dict]):
    print(f"\n{'=' * 70}")
    print(f"  TRANSAKTIONEN OHNE MUSTER  ({len(txs)})")
    print(f"{'=' * 70}")
    for t in txs:
        sign = "+" if t["betrag"] > 0 else ""
        bdt  = t.get("bookedDateTime", "")
        time_str = f"  {bdt[11:16]}" if len(bdt) >= 16 else "       "
        print(f"  {t['datum']}{time_str}  {sign}{t['betrag']:>10,.2f} EUR  "
              f"{t['gegenpartei'][:28]:<28}  {t['category_level1']}")


def print_summary(rec, sea, seq, no_pat):
    total  = len(rec) + len(sea) + len(seq)
    tx_in  = (sum(p["recurrence_sample_size"] for p in rec + sea)
              + sum(p["seq_pair_count"] * 2 for p in seq))
    with_time = sum(1 for p in rec + sea + seq if p.get("time_data_available"))
    print(f"\n{'=' * 70}")
    print(f"  ZUSAMMENFASSUNG")
    print(f"{'=' * 70}")
    print(f"  RECURRING  Muster : {len(rec)}")
    print(f"  SEASONAL   Muster : {len(sea)}")
    print(f"  SEQUENTIAL Muster : {len(seq)}")
    print(f"  Gesamt Muster     : {total}")
    print(f"  Tx in Mustern     : {tx_in}")
    print(f"  Tx ohne Muster    : {len(no_pat)}")
    print(f"  Mit Uhrzeit       : {with_time}/{total} Muster haben bookedDateTime-Daten")


# ─────────────────────────────────────────────
# 14. MAIN (CLI)
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
    print(f"  patterns_db     : {fs['total_patterns']} Dokumente")
    print(f"  distributions_db: {fs['distributions']} Dokumente")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
