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
from collections import defaultdict
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

def _load_all_distributions(db) -> list[dict]:
    """
    Lädt ALLE Transaktionen aus distributions_db ohne Datumsfilter.
    Der Zeitfenster-Filter wird erst nach dem Laden angewendet,
    damit wir zuerst das Referenzdatum aus den Daten ableiten können.
    """
    docs = db.collection(SOURCE_COLLECTION).stream()
    transactions = []
    for doc in docs:
        data = doc.to_dict()
        try:
            data["_date_obj"] = date.fromisoformat(data["datum"])
        except (KeyError, ValueError):
            continue  # Dokument ohne gültiges Datum überspringen
        transactions.append(data)
    return transactions


def _determine_window(transactions: list[dict]) -> tuple[date, date, int]:
    """
    Leitet das Analysefenster aus den Daten ab.

    Referenzdatum = letztes Transaktionsdatum in den Daten
    Cutoff        = Referenzdatum - LOOKBACK_MONTHS

    Damit funktioniert das Modul korrekt mit:
      - Live-Daten    (Referenz ≈ today)
      - Demo-Daten    (Referenz = letztes Datum im Datensatz)
      - Backfills     (Referenz = Ende des importierten Zeitraums)

    Returns:
        (cutoff_date, reference_date, n_months)
    """
    if not transactions:
        today = date.today()
        cutoff_month = today.month - LOOKBACK_MONTHS
        cutoff_year  = today.year
        while cutoff_month <= 0:
            cutoff_month += 12
            cutoff_year  -= 1
        return date(cutoff_year, cutoff_month, 1), today, LOOKBACK_MONTHS

    # Letztes Transaktionsdatum als Referenz
    reference_date = max(t["_date_obj"] for t in transactions)

    # Cutoff = LOOKBACK_MONTHS vor dem Referenzdatum
    cutoff_month = reference_date.month - LOOKBACK_MONTHS
    cutoff_year  = reference_date.year
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
                 Schlüssel: "AUSGABEN – PERSONAL :: Muster AG"
    2. Fallback: nur category_level1             (aggregiert)
                 Schlüssel: "AUSGABEN – PERSONAL"

    Primärgruppe wird verwendet wenn sie >= MIN_SAMPLE_SINGLE Einträge hat,
    sonst wird die Transaktion der Kategorie-Gruppe zugeschlagen.
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
    cutoff_date: date,
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
            breakdown[label]["total"]  = round(breakdown[label]["total"] + amt, 2)
            breakdown[label]["amounts"].append(round(amt, 2))

    return breakdown


def _poisson_stats(n_transactions: int, n_months: int) -> dict:
    """
    Poisson-Parameter aus beobachteter Häufigkeit.
    λ = n / n_months | P(≥1) = 1 - e^(-λ)
    """
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
    reference_date:    date,
    horizon:           int,
    lambda_per_month:  float,
    amount_mean:       float,
    amount_std:        float,
) -> list[dict]:
    """
    Vorausschau der nächsten `horizon` Monate ab reference_date.
    Stationäres Modell: λ und μ gelten als konstant pro Monat.
    """
    forecast = []
    cursor = reference_date

    for _ in range(horizon):
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

        p_any    = round(1.0 - math.exp(-lambda_per_month), 4)
        expected = round(lambda_per_month * amount_mean, 2)
        ci_low   = round(max(0.0, lambda_per_month * max(0.0, amount_mean - CI_90_Z * amount_std)), 2)
        ci_high  = round(lambda_per_month * (amount_mean + CI_90_Z * amount_std), 2)

        forecast.append({
            "month":           _month_label(cursor),
            "probability":     p_any,
            "probability_pct": round(p_any * 100, 1),
            "expected_amount": expected,
            "ci_90_low":       ci_low,
            "ci_90_high":      ci_high,
        })

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
      1. Alle Dokumente aus distributions_db laden
      2. Analysefenster aus den Daten ableiten (letztes Datum - 6 Monate)
      3. Transaktionen auf das Fenster filtern
      4. Nach Kategorie (+Gegenpartei) gruppieren
      5. Pro Gruppe: Poisson + Normalverteilung + Forecast berechnen
      6. Ergebnisse in forecast_distribution schreiben

    Returns:
        {"groups_written": int, "transactions_analyzed": int}
    """
    db = _init_firestore()

    # 1. Alle Transaktionen laden (kein Datumsfilter)
    all_transactions = _load_all_distributions(db)

    if not all_transactions:
        print("    ⚠️  Keine Transaktionen in distributions_db – nichts zu berechnen.")
        return {"groups_written": 0, "transactions_analyzed": 0}

    # 2. Fenster aus den Daten ableiten
    cutoff_date, reference_date, n_months = _determine_window(all_transactions)
    print(f"    Zeitfenster  : {cutoff_date} → {reference_date}  ({n_months} Monate)")
    print(f"    Referenz     : letztes Transaktionsdatum in den Daten")

    # 3. Auf Fenster filtern
    transactions = _filter_by_window(all_transactions, cutoff_date, reference_date)
    print(f"    Transaktionen: {len(transactions)} im Fenster  "
          f"({len(all_transactions)} total in distributions_db)")

    if not transactions:
        print("    ⚠️  Keine Transaktionen im Zeitfenster – nichts zu berechnen.")
        return {"groups_written": 0, "transactions_analyzed": 0}

    # 4. Gruppieren
    groups = _build_groups(transactions)
    print(f"    Gruppen      : {len(groups)}")

    # 5. Dokumente bauen
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

    # 6. Firestore schreiben
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
    forecasts = doc.get("forecast_months", [])[:3]
    if forecasts:
        print(f"  Vorausschau    :", end="")
        for fm in forecasts:
            print(f"  {fm['month']} ({fm['probability_pct']}% / CHF {fm['expected_amount']:,.0f})", end="")
        print(" ...")


def _print_summary_table(documents: list[dict]):
    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  FORECAST DISTRIBUTION – ÜBERSICHT  ({len(documents)} Gruppen)")
    print(f"{SEP}")
    print(f"  {'Gruppe':<42}  {'P(≥1)':<7}  {'Ø Betrag':>12}  {'Erw./Mo':>12}")
    print(f"  {'─'*42}  {'─'*7}  {'─'*12}  {'─'*12}")
    for doc in sorted(documents, key=lambda d: d.get("amount_expected_monthly", 0), reverse=True):
        print(
            f"  {doc['group_key'][:42]:<42}  "
            f"{doc['probability_any_pct']:>5.1f}%  "
            f"CHF {doc.get('amount_mean', 0):>8,.0f}  "
            f"CHF {doc.get('amount_expected_monthly', 0):>8,.0f}"
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

    print(f"  Zeitfenster  : {cutoff_date} → {reference_date}  ({n_months} Monate)")
    print(f"  Im Fenster   : {len(transactions)} Transaktionen\n")

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
