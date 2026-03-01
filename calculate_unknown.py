"""
calculate_unknown.py
====================
Berechnet Wahrscheinlichkeitsverteilungen für Transaktionen ohne Muster
(Quelle: Firestore-Sammlung distributions_db) und speichert die Ergebnisse
in der Sammlung forecast_distribution für den Frontend-Zugriff.

Statistisches Modell:
  - Poisson-Prozess  : Schätzt die monatliche Auftretenswahrscheinlichkeit
                       (λ = Ø Transaktionen/Monat; P(≥1) = 1 - e^(-λ))
  - Normalverteilung : Schätzt μ (Mittelwert) und σ (Streuung) der Beträge
                       inkl. 90%- und 95%-Konfidenzintervall

  NEU – Präzisere Vorhersage via bookedDateTime:
  - Buchungsstunde   : avg_booking_hour / booking_hour_std / next_expected_time
                       Ermöglicht Tages-genaue Forecast-Zeitstempel
  - Wochentag-Analyse: Dominanter Wochentag (z.B. "Montag 62%")
                       → next_likely_date wählt nächsten solchen Wochentag
  - Monatstag-Analyse: Dominante Monatshälfte (1-15 vs. 16-31)
                       + Durchschnittlicher Monatstag
  - next_likely_date : Kombination aus Wochentag-Tendenz + Buchungszeit
                       → präzisester einzelner Vorhersagewert pro Gruppe

Zeitfenster:
  Referenzdatum = letztes Transaktionsdatum in distributions_db
  (NICHT today – damit funktioniert das Modul auch mit historischen Demo-Daten)
  Cutoff        = Referenzdatum - LOOKBACK_MONTHS

PUBLIC API (für pipeline.py):
    from calculate_unknown import calculate_unknown
    result = calculate_unknown()
    # result = {"groups_written": int, "transactions_analyzed": int}

CLI (Standalone):
    python calculate_unknown.py
"""

import math
import re
import sys
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta
from statistics import mean, stdev

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

LOOKBACK_MONTHS     = 6      # Analysefenster in Monaten
MIN_SAMPLE_SINGLE   = 2      # Mindest-Transaktionen für Kategorie+Gegenpartei-Gruppe
MIN_SAMPLE_CATEGORY = 1      # Mindest-Transaktionen für reine Kategorie-Gruppe
FORECAST_HORIZON    = 6      # Vorausschau in Monaten
CI_90_Z             = 1.645  # Z-Wert für 90%-Konfidenzintervall
CI_95_Z             = 1.960  # Z-Wert für 95%-Konfidenzintervall

SOURCE_COLLECTION   = "distributions_db"
TARGET_COLLECTION   = "forecast_distribution"

DOW_NAMES = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


# ─────────────────────────────────────────────
# FIRESTORE SETUP
# ─────────────────────────────────────────────

def _init_firestore():
    """Initialisiert Firebase Admin SDK. Idempotent bei Mehrfachaufruf."""
    import os
    import firebase_admin
    from firebase_admin import credentials, firestore

    SERVICE_ACCOUNT_FILE = "bank-417a7-firebase-adminsdk-fbsvc-60ba2be615.json"

    if not firebase_admin._apps:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            raise RuntimeError(
                f"Service-Account-Datei nicht gefunden: {SERVICE_ACCOUNT_FILE}\n"
                "GitHub Action: Secrets → FIREBASE_SERVICE_ACCOUNT muss gesetzt sein."
            )
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)

    return firestore.client()


# ─────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────

def _parse_booked_dt(raw: str) -> datetime | None:
    """Parst bookedDateTime ISO 8601 → datetime (timezone-naiv). None bei leerem Wert."""
    if not raw:
        return None
    try:
        s  = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def _load_all_distributions(db) -> list[dict]:
    """
    Lädt ALLE Transaktionen aus distributions_db ohne Datumsfilter.
    Ergänzt _date_obj (date) und _datetime_obj (datetime | None) aus
    bookedDateTime für die Zeitanalyse.
    """
    docs         = db.collection(SOURCE_COLLECTION).stream()
    transactions = []
    for doc in docs:
        data = doc.to_dict()
        try:
            data["_date_obj"] = date.fromisoformat(data["datum"])
        except (KeyError, ValueError):
            continue
        # bookedDateTime parsen – None wenn nicht vorhanden oder ungültig
        data["_datetime_obj"] = _parse_booked_dt(data.get("bookedDateTime", ""))
        transactions.append(data)
    return transactions


def _determine_window(transactions: list[dict]) -> tuple[date, date, int]:
    """
    Leitet das Analysefenster aus den Daten ab.
    Referenzdatum = letztes Transaktionsdatum (nicht today).
    """
    if not transactions:
        today        = date.today()
        cutoff_month = today.month - LOOKBACK_MONTHS
        cutoff_year  = today.year
        while cutoff_month <= 0:
            cutoff_month += 12
            cutoff_year  -= 1
        return date(cutoff_year, cutoff_month, 1), today, LOOKBACK_MONTHS

    reference_date = max(t["_date_obj"] for t in transactions)
    cutoff_month   = reference_date.month - LOOKBACK_MONTHS
    cutoff_year    = reference_date.year
    while cutoff_month <= 0:
        cutoff_month += 12
        cutoff_year  -= 1
    cutoff_date = date(cutoff_year, cutoff_month, 1)

    return cutoff_date, reference_date, LOOKBACK_MONTHS


def _filter_by_window(transactions: list[dict], cutoff_date: date, reference_date: date) -> list[dict]:
    """Filtert Transaktionen auf das Analysefenster [cutoff_date, reference_date]."""
    return [
        t for t in transactions
        if cutoff_date <= t["_date_obj"] <= reference_date
    ]


# ─────────────────────────────────────────────
# GRUPPIERUNG
# ─────────────────────────────────────────────

def _build_groups(transactions: list[dict]) -> dict[str, list[dict]]:
    """
    Erstellt zwei Gruppenebenen:
    1. Primär  : category_level1 + gegenpartei  (spezifisch)
    2. Fallback: nur category_level1             (aggregiert)
    """
    primary: defaultdict[str, list] = defaultdict(list)
    for t in transactions:
        cat = t.get("category_level1", "UNBEKANNT")
        geg = t.get("gegenpartei", "").strip() or "Unbekannt"
        primary[f"{cat} :: {geg}"].append(t)

    category: defaultdict[str, list] = defaultdict(list)
    final: dict[str, list] = {}

    for key, txs in primary.items():
        if len(txs) >= MIN_SAMPLE_SINGLE:
            final[key] = txs
        else:
            cat = key.split(" :: ")[0]
            category[cat].extend(txs)

    for cat, txs in category.items():
        if len(txs) >= MIN_SAMPLE_CATEGORY and cat not in final:
            final[cat] = txs

    return final


# ─────────────────────────────────────────────
# UHRZEIT-ANALYSE  (NEU)
# ─────────────────────────────────────────────

def _booking_time_stats(transactions: list[dict]) -> dict:
    """
    Berechnet Buchungszeit-Statistiken aus bookedDateTime (_datetime_obj).

    Returns:
        avg_booking_hour      : float | None   (z.B. 8.5 = 08:30)
        booking_hour_std      : float | None
        next_expected_time    : str   | None   (z.B. "08:30")
        time_data_available   : bool
        time_data_count       : int            (Transaktionen mit Zeitdaten)
    """
    hours   = []
    minutes = []

    for t in transactions:
        dt = t.get("_datetime_obj")
        if dt:
            hours.append(dt.hour + dt.minute / 60.0)
            minutes.append(dt.minute)

    if not hours:
        return {
            "avg_booking_hour":   None,
            "booking_hour_std":   None,
            "next_expected_time": None,
            "time_data_available": False,
            "time_data_count":    0,
        }

    avg_h = mean(hours)
    std_h = round(stdev(hours), 2) if len(hours) >= 2 else 0.0
    h_int = int(avg_h)
    m_int = round((avg_h - h_int) * 60)
    if m_int >= 60:
        h_int += 1
        m_int  = 0

    return {
        "avg_booking_hour":    round(avg_h, 2),
        "booking_hour_std":    std_h,
        "next_expected_time":  f"{h_int:02d}:{m_int:02d}",
        "time_data_available": True,
        "time_data_count":     len(hours),
    }


# ─────────────────────────────────────────────
# WOCHENTAG- & MONATSTAG-ANALYSE  (NEU)
# ─────────────────────────────────────────────

def _weekday_stats(transactions: list[dict]) -> dict:
    """
    Analysiert an welchen Wochentagen diese Gruppe typischerweise bucht.

    Returns:
        dominant_weekday       : str   (z.B. "Montag")
        dominant_weekday_idx   : int   (0=Mo … 6=So)
        dominant_weekday_pct   : float (Anteil 0–100)
        weekday_distribution   : dict  {"Montag": 3, "Dienstag": 1, ...}
        weekday_is_reliable    : bool  (dominant_weekday_pct >= 50%)
    """
    counts: Counter = Counter()
    for t in transactions:
        counts[t["_date_obj"].weekday()] += 1

    if not counts:
        return {
            "dominant_weekday":     None,
            "dominant_weekday_idx": None,
            "dominant_weekday_pct": None,
            "weekday_distribution": {},
            "weekday_is_reliable":  False,
        }

    total     = sum(counts.values())
    dom_idx   = counts.most_common(1)[0][0]
    dom_count = counts[dom_idx]
    dom_pct   = round(dom_count / total * 100, 1)

    return {
        "dominant_weekday":     DOW_NAMES[dom_idx],
        "dominant_weekday_idx": dom_idx,
        "dominant_weekday_pct": dom_pct,
        "weekday_distribution": {DOW_NAMES[i]: counts[i] for i in range(7) if counts[i] > 0},
        "weekday_is_reliable":  dom_pct >= 50.0,
    }


def _dom_stats(transactions: list[dict]) -> dict:
    """
    Analysiert an welchem Monatstag (day-of-month) diese Gruppe typischerweise bucht.

    Returns:
        avg_day_of_month    : float  (Durchschnittlicher Tag, z.B. 14.3)
        dom_half            : str    ("erste Hälfte" | "zweite Hälfte")
        dom_first_half_pct  : float  (Anteil 1.–15. in %)
    """
    days = [t["_date_obj"].day for t in transactions]
    if not days:
        return {"avg_day_of_month": None, "dom_half": None, "dom_first_half_pct": None}

    avg_dom        = round(mean(days), 1)
    first_half     = sum(1 for d in days if d <= 15)
    first_half_pct = round(first_half / len(days) * 100, 1)

    return {
        "avg_day_of_month":   avg_dom,
        "dom_half":           "erste Hälfte" if first_half_pct >= 50 else "zweite Hälfte",
        "dom_first_half_pct": first_half_pct,
    }


def _next_likely_date(
    reference_date: date,
    weekday_stats:  dict,
    dom_stats:      dict,
    time_stats:     dict,
) -> str | None:
    """
    Berechnet das wahrscheinlichste nächste Auftrittsdatum.

    Logik:
      1. Wochentag zuverlässig (>= 50%) → nächster solcher Wochentag ab reference_date
      2. Kein zuverlässiger Wochentag    → reference_date + 30 Tage (monatliche Erwartung),
         verschoben auf avg_day_of_month in diesem Monat
      3. Uhrzeit anhängen wenn time_data_available

    Gibt ISO-String zurück: "2024-03-05" oder "2024-03-05T08:30:00"
    """
    dom_idx = weekday_stats.get("dominant_weekday_idx")

    if weekday_stats.get("weekday_is_reliable") and dom_idx is not None:
        # Nächster passender Wochentag (frühestens +1 Tag)
        candidate = reference_date + timedelta(days=1)
        for _ in range(7):
            if candidate.weekday() == dom_idx:
                break
            candidate += timedelta(days=1)
        base_date = candidate
    else:
        # Kein klarer Wochentag → in etwa 1 Monat, gerundet auf avg_day_of_month
        avg_dom = dom_stats.get("avg_day_of_month")
        if avg_dom:
            target_day = int(round(avg_dom))
            m = reference_date.month + 1 if reference_date.month < 12 else 1
            y = reference_date.year if reference_date.month < 12 else reference_date.year + 1
            import calendar
            target_day = min(target_day, calendar.monthrange(y, m)[1])
            base_date  = date(y, m, target_day)
        else:
            base_date = reference_date + timedelta(days=30)

    # Auf Werktag verschieben (Sa/So → Mo)
    while base_date.weekday() >= 5:
        base_date += timedelta(days=1)

    date_str = base_date.isoformat()

    # Uhrzeit anhängen wenn verfügbar
    if time_stats.get("time_data_available") and time_stats.get("next_expected_time"):
        return f"{date_str}T{time_stats['next_expected_time']}:00"

    return date_str


# ─────────────────────────────────────────────
# STATISTIK PRO GRUPPE
# ─────────────────────────────────────────────

def _month_label(d: date) -> str:
    return d.strftime("%Y-%m")


def _iter_months(start: date, end: date):
    """Generator: liefert den ersten Tag jedes Monats von start bis end."""
    cursor = start.replace(day=1)
    while cursor <= end:
        yield cursor
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def _compute_monthly_breakdown(
    transactions: list[dict],
    cutoff_date:  date,
    reference_date: date,
) -> dict[str, dict]:
    """
    Monatliche Aufschlüsselung im Analysefenster.
    Alle Monate werden initialisiert – auch solche ohne Transaktionen.
    """
    breakdown: dict[str, dict] = {
        _month_label(m): {"count": 0, "total": 0.0, "amounts": []}
        for m in _iter_months(cutoff_date, reference_date)
    }

    for t in transactions:
        label = _month_label(t["_date_obj"])
        if label in breakdown:
            amt = abs(t.get("betrag", 0.0))
            breakdown[label]["count"]  += 1
            breakdown[label]["total"]   = round(breakdown[label]["total"] + amt, 2)
            breakdown[label]["amounts"].append(round(amt, 2))

    return breakdown


def _poisson_stats(n_transactions: int, n_months: int) -> dict:
    """Poisson-Parameter aus beobachteter Häufigkeit."""
    lm    = round(n_transactions / n_months, 4)
    p_any = round(1.0 - math.exp(-lm), 4)
    return {
        "lambda_per_month":    lm,
        "probability_any":     p_any,
        "probability_any_pct": round(p_any * 100, 1),
    }


def _amount_stats(amounts: list[float]) -> dict:
    """Lage- und Streuungsmasse + 90%/95%-Konfidenzintervalle."""
    if not amounts:
        return {}

    mu = round(mean(amounts), 2)
    sd = round(stdev(amounts), 2) if len(amounts) >= 2 else 0.0

    return {
        "amount_mean":       mu,
        "amount_std":        sd,
        "amount_min":        round(min(amounts), 2),
        "amount_max":        round(max(amounts), 2),
        "amount_median":     round(sorted(amounts)[len(amounts) // 2], 2),
        "amount_ci_90_low":  round(max(0.0, mu - CI_90_Z * sd), 2),
        "amount_ci_90_high": round(mu + CI_90_Z * sd, 2),
        "amount_ci_95_low":  round(max(0.0, mu - CI_95_Z * sd), 2),
        "amount_ci_95_high": round(mu + CI_95_Z * sd, 2),
    }


def _build_forecast_months(
    reference_date:   date,
    horizon:          int,
    lambda_per_month: float,
    amount_mean:      float,
    amount_std:       float,
    expected_time:    str | None,
) -> list[dict]:
    """
    Vorausschau der nächsten `horizon` Monate.
    Enthält next_expected_time pro Monat wenn verfügbar.
    """
    forecast = []
    cursor   = reference_date

    for _ in range(horizon):
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

        p_any    = round(1.0 - math.exp(-lambda_per_month), 4)
        expected = round(lambda_per_month * amount_mean, 2)
        ci_low   = round(max(0.0, lambda_per_month * max(0.0, amount_mean - CI_90_Z * amount_std)), 2)
        ci_high  = round(lambda_per_month * (amount_mean + CI_90_Z * amount_std), 2)

        entry = {
            "month":           _month_label(cursor),
            "probability":     p_any,
            "probability_pct": round(p_any * 100, 1),
            "expected_amount": expected,
            "ci_90_low":       ci_low,
            "ci_90_high":      ci_high,
        }
        if expected_time:
            entry["expected_time"] = expected_time   # NEU: Uhrzeit pro Forecast-Monat

        forecast.append(entry)

    return forecast


# ─────────────────────────────────────────────
# DOKUMENT PRO GRUPPE ZUSAMMENBAUEN
# ─────────────────────────────────────────────

def _build_distribution_doc(
    group_key:      str,
    transactions:   list[dict],
    cutoff_date:    date,
    reference_date: date,
    n_months:       int,
) -> dict:
    amounts   = [abs(t.get("betrag", 0.0)) for t in transactions]
    poisson   = _poisson_stats(len(transactions), n_months)
    amt_stats = _amount_stats(amounts)
    monthly   = _compute_monthly_breakdown(transactions, cutoff_date, reference_date)

    # NEU: Zeit-, Wochentag- und Monatstag-Analyse
    time_stats    = _booking_time_stats(transactions)
    weekday       = _weekday_stats(transactions)
    dom           = _dom_stats(transactions)
    next_likely   = _next_likely_date(reference_date, weekday, dom, time_stats)

    if " :: " in group_key:
        category, counterparty = group_key.split(" :: ", 1)
        group_type = "category_counterparty"
    else:
        category     = group_key
        counterparty = None
        group_type   = "category_only"

    amount_expected_monthly = round(
        poisson["lambda_per_month"] * amt_stats.get("amount_mean", 0.0), 2
    )

    forecast = _build_forecast_months(
        reference_date   = reference_date,
        horizon          = FORECAST_HORIZON,
        lambda_per_month = poisson["lambda_per_month"],
        amount_mean      = amt_stats.get("amount_mean", 0.0),
        amount_std       = amt_stats.get("amount_std", 0.0),
        expected_time    = time_stats.get("next_expected_time"),   # NEU
    )

    return {
        # ── Identifikation ──────────────────────────────────────────
        "group_key":               group_key,
        "group_type":              group_type,
        "category":                category,
        "counterparty":            counterparty,
        # ── Metadaten ───────────────────────────────────────────────
        "sample_size":             len(transactions),
        "lookback_months":         n_months,
        "analysis_from":           cutoff_date.isoformat(),
        "analysis_to":             reference_date.isoformat(),
        "calculated_at":           datetime.utcnow().isoformat() + "Z",
        # ── Poisson: Häufigkeit ──────────────────────────────────────
        **poisson,
        # ── Normalverteilung: Beträge ────────────────────────────────
        **amt_stats,
        "amount_expected_monthly": amount_expected_monthly,
        # ── NEU: Uhrzeit-Analyse ─────────────────────────────────────
        "avg_booking_hour":        time_stats["avg_booking_hour"],
        "booking_hour_std":        time_stats["booking_hour_std"],
        "next_expected_time":      time_stats["next_expected_time"],
        "time_data_available":     time_stats["time_data_available"],
        "time_data_count":         time_stats["time_data_count"],
        # ── NEU: Wochentag-Tendenz ────────────────────────────────────
        "dominant_weekday":        weekday["dominant_weekday"],
        "dominant_weekday_pct":    weekday["dominant_weekday_pct"],
        "weekday_is_reliable":     weekday["weekday_is_reliable"],
        "weekday_distribution":    weekday["weekday_distribution"],
        # ── NEU: Monatstag-Tendenz ────────────────────────────────────
        "avg_day_of_month":        dom["avg_day_of_month"],
        "dom_half":                dom["dom_half"],
        "dom_first_half_pct":      dom["dom_first_half_pct"],
        # ── NEU: Bester Einzelvorhersagewert ─────────────────────────
        # Kombiniert Wochentag-Tendenz + avg_day_of_month + Uhrzeit.
        # Frontend nutzt dieses Feld analog zu next_expected_date in pattern_db.
        "next_likely_date":        next_likely,
        # ── Monatliche Aufschlüsselung ───────────────────────────────
        "monthly_breakdown":       monthly,
        # ── Vorausschau nächste 6 Monate ─────────────────────────────
        "forecast_months":         forecast,
    }


# ─────────────────────────────────────────────
# FIRESTORE SCHREIBEN
# ─────────────────────────────────────────────

def _safe_doc_id(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)
    return safe[:200]


def _write_to_firestore(db, documents: list[dict]) -> int:
    ref     = db.collection(TARGET_COLLECTION)
    written = 0
    for doc in documents:
        ref.document(_safe_doc_id(doc["group_key"])).set(doc)
        written += 1
    return written


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def calculate_unknown() -> dict:
    """
    Haupteinstiegspunkt für pipeline.py.

    Ablauf:
      1. Alle Dokumente aus distributions_db laden (inkl. bookedDateTime)
      2. Analysefenster aus den Daten ableiten (letztes Datum - 6 Monate)
      3. Transaktionen auf das Fenster filtern
      4. Nach Kategorie (+Gegenpartei) gruppieren
      5. Pro Gruppe: Poisson + Normalverteilung + Zeit/Wochentag/DOM + Forecast berechnen
      6. Ergebnisse in forecast_distribution schreiben

    Returns:
        {"groups_written": int, "transactions_analyzed": int}
    """
    db = _init_firestore()

    all_transactions = _load_all_distributions(db)

    if not all_transactions:
        print("    ⚠️  Keine Transaktionen in distributions_db – nichts zu berechnen.")
        return {"groups_written": 0, "transactions_analyzed": 0}

    cutoff_date, reference_date, n_months = _determine_window(all_transactions)
    print(f"    Zeitfenster  : {cutoff_date} → {reference_date}  ({n_months} Monate)")
    print(f"    Referenz     : letztes Transaktionsdatum in den Daten")

    transactions = _filter_by_window(all_transactions, cutoff_date, reference_date)

    with_time = sum(1 for t in transactions if t.get("_datetime_obj"))
    print(f"    Transaktionen: {len(transactions)} im Fenster  "
          f"({len(all_transactions)} total in distributions_db)")
    print(f"    Mit Uhrzeit  : {with_time}/{len(transactions)} haben bookedDateTime")

    if not transactions:
        print("    ⚠️  Keine Transaktionen im Zeitfenster – nichts zu berechnen.")
        return {"groups_written": 0, "transactions_analyzed": 0}

    groups    = _build_groups(transactions)
    print(f"    Gruppen      : {len(groups)}")

    documents = [
        _build_distribution_doc(
            group_key      = gkey,
            transactions   = txs,
            cutoff_date    = cutoff_date,
            reference_date = reference_date,
            n_months       = n_months,
        )
        for gkey, txs in groups.items()
    ]

    written = _write_to_firestore(db, documents)
    print(f"    Geschrieben  : {written} Dokumente → '{TARGET_COLLECTION}'")

    return {
        "groups_written":        written,
        "transactions_analyzed": len(transactions),
    }


# ─────────────────────────────────────────────
# KONSOLENAUSGABE (nur CLI)
# ─────────────────────────────────────────────

def _print_group_summary(doc: dict):
    SEP = "." * 65
    print(f"\n  {'─'*65}")
    print(f"  {doc['group_key']}")
    print(f"  {SEP}")
    print(f"  Typ            : {doc['group_type']}")
    print(f"  Stichproben    : {doc['sample_size']}  "
          f"({doc['analysis_from']} → {doc['analysis_to']})")
    print(f"  λ/Monat        : {doc['lambda_per_month']}  "
          f"→ P(≥1 Tx/Monat) = {doc['probability_any_pct']}%")
    print(f"  Betrag Ø       : CHF {doc.get('amount_mean', 0):>10,.2f}  "
          f"(σ = {doc.get('amount_std', 0):,.2f})")
    print(f"  90%-CI         : CHF {doc.get('amount_ci_90_low', 0):>10,.2f}  –  "
          f"CHF {doc.get('amount_ci_90_high', 0):,.2f}")
    print(f"  Erw. Monatslast: CHF {doc.get('amount_expected_monthly', 0):>10,.2f}")

    # NEU: Uhrzeit & Wochentag
    if doc.get("time_data_available"):
        print(f"  Buchungszeit   : Ø {doc['avg_booking_hour']:.1f}h  "
              f"→ {doc['next_expected_time']}  (σ={doc['booking_hour_std']}h, "
              f"n={doc['time_data_count']})")
    else:
        print(f"  Buchungszeit   : keine bookedDateTime-Daten")

    dow = doc.get("dominant_weekday")
    if dow:
        reliable = "✓ zuverlässig" if doc.get("weekday_is_reliable") else "schwach"
        print(f"  Wochentag      : {dow}  ({doc['dominant_weekday_pct']}%)  [{reliable}]")

    dom_half = doc.get("dom_half")
    if dom_half:
        print(f"  Monatstag Ø    : {doc['avg_day_of_month']}  ({dom_half}, "
              f"1.–15.: {doc['dom_first_half_pct']}%)")

    next_ld = doc.get("next_likely_date")
    if next_ld:
        time_flag = "  🕐" if "T" in next_ld else ""
        print(f"  Nächstm. Datum : \033[92m{next_ld}\033[0m{time_flag}")

    forecasts = doc.get("forecast_months", [])[:3]
    if forecasts:
        print(f"  Vorausschau    :", end="")
        for fm in forecasts:
            t_str = f" {fm['expected_time']}" if fm.get("expected_time") else ""
            print(f"  {fm['month']}{t_str} ({fm['probability_pct']}% / CHF {fm['expected_amount']:,.0f})", end="")
        print(" ...")


def _print_summary_table(documents: list[dict]):
    SEP = "=" * 80
    print(f"\n{SEP}")
    print(f"  FORECAST DISTRIBUTION – ÜBERSICHT  ({len(documents)} Gruppen)")
    print(f"{SEP}")
    print(f"  {'Gruppe':<38}  {'P(≥1)':<7}  {'Ø Betrag':>12}  {'Nächstm. Datum':<22}  {'Uhrzeit':<8}")
    print(f"  {'─'*38}  {'─'*7}  {'─'*12}  {'─'*22}  {'─'*8}")
    for doc in sorted(documents, key=lambda d: d.get("amount_expected_monthly", 0), reverse=True):
        next_ld   = doc.get("next_likely_date", "-") or "-"
        time_part = doc.get("next_expected_time", "-") or "-"
        print(
            f"  {doc['group_key'][:38]:<38}  "
            f"{doc['probability_any_pct']:>5.1f}%  "
            f"CHF {doc.get('amount_mean', 0):>8,.0f}  "
            f"{next_ld:<22}  "
            f"{time_part:<8}"
        )
    print(f"{SEP}")


def main():
    print(f"\n{'='*70}")
    print(f"  CALCULATE UNKNOWN – Wahrscheinlichkeitsverteilung")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    db = _init_firestore()

    print(f"  Lade alle Daten aus '{SOURCE_COLLECTION}'...")
    all_transactions = _load_all_distributions(db)
    print(f"  ✅ {len(all_transactions)} Transaktionen total")

    if not all_transactions:
        print("\n  ⚠️  Keine Transaktionen gefunden – Abbruch.")
        sys.exit(0)

    cutoff_date, reference_date, n_months = _determine_window(all_transactions)
    transactions = _filter_by_window(all_transactions, cutoff_date, reference_date)

    with_time = sum(1 for t in transactions if t.get("_datetime_obj"))
    print(f"  Zeitfenster  : {cutoff_date} → {reference_date}  ({n_months} Monate)")
    print(f"  Im Fenster   : {len(transactions)} Transaktionen")
    print(f"  Mit Uhrzeit  : {with_time}/{len(transactions)} haben bookedDateTime\n")

    if not transactions:
        print("  ⚠️  Keine Transaktionen im Fenster – Abbruch.")
        sys.exit(0)

    groups    = _build_groups(transactions)
    documents = [
        _build_distribution_doc(
            group_key      = gkey,
            transactions   = txs,
            cutoff_date    = cutoff_date,
            reference_date = reference_date,
            n_months       = n_months,
        )
        for gkey, txs in groups.items()
    ]

    for doc in documents:
        _print_group_summary(doc)

    _print_summary_table(documents)

    written = _write_to_firestore(db, documents)

    print(f"\n{'='*70}")
    print(f"  ✅ {written} Dokumente → Firestore '{TARGET_COLLECTION}'")
    print(f"  ✅ {len(transactions)} Transaktionen analysiert")
    print(f"  Abgeschlossen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
