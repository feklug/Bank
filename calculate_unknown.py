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

Zeitfenster: letzte 6 Monate ab heute (rollend)

PUBLIC API (für pipeline.py):
    from calculate_unknown import calculate_unknown
    result = calculate_unknown()
    # result = {"groups_written": int, "transactions_analyzed": int}

CLI (Standalone):
    python calculate_unknown.py
"""

import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from statistics import mean, stdev

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

LOOKBACK_MONTHS        = 6      # Analysefenster in Monaten
MIN_SAMPLE_SINGLE      = 2      # Mindest-Transaktionen für Kategorie+Gegenpartei-Gruppe
MIN_SAMPLE_CATEGORY    = 1      # Mindest-Transaktionen für reine Kategorie-Gruppe
FORECAST_HORIZON       = 6      # Vorausschau in Monaten
CI_90_Z                = 1.645  # Z-Wert für 90%-Konfidenzintervall
CI_95_Z                = 1.960  # Z-Wert für 95%-Konfidenzintervall

SOURCE_COLLECTION      = "distributions_db"
TARGET_COLLECTION      = "forecast_distribution"


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

def _load_distributions(db, cutoff_date: date) -> list[dict]:
    """
    Lädt alle Transaktionen aus distributions_db,
    die auf oder nach cutoff_date liegen.

    Gibt eine Liste von Dicts zurück, jedes mit mindestens:
      datum, betrag, gegenpartei, category_level1
    """
    ref  = db.collection(SOURCE_COLLECTION)
    docs = ref.stream()

    transactions = []
    for doc in docs:
        data = doc.to_dict()
        try:
            tx_date = date.fromisoformat(data["datum"])
        except (KeyError, ValueError):
            continue  # Dokument ohne gültiges Datum überspringen

        if tx_date >= cutoff_date:
            data["_date_obj"] = tx_date
            transactions.append(data)

    return transactions


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

    Rückgabe: Dict {group_key: [Transaktionen]}
    """
    # Zuerst alle Transaktionen nach Primärschlüssel sammeln
    primary: defaultdict[str, list] = defaultdict(list)
    for t in transactions:
        cat  = t.get("category_level1", "UNBEKANNT")
        geg  = t.get("gegenpartei", "").strip() or "Unbekannt"
        key  = f"{cat} :: {geg}"
        primary[key].append(t)

    # Kategorie-Fallback aufbauen
    category: defaultdict[str, list] = defaultdict(list)
    final: dict[str, list] = {}

    for key, txs in primary.items():
        if len(txs) >= MIN_SAMPLE_SINGLE:
            final[key] = txs
        else:
            # Zu wenige Daten: nur Kategorie-Level verwenden
            cat = key.split(" :: ")[0]
            category[cat].extend(txs)

    for cat, txs in category.items():
        if len(txs) >= MIN_SAMPLE_CATEGORY:
            # Kategorie-Gruppe zusammenführen – Duplikate vermeiden
            if cat not in final:
                final[cat] = txs
            else:
                # Bereits als Primärgruppe drin – nicht doppelt zählen
                pass

    return final


# ─────────────────────────────────────────────
# STATISTIK PRO GRUPPE
# ─────────────────────────────────────────────

def _month_label(d: date) -> str:
    """Gibt 'YYYY-MM' zurück."""
    return d.strftime("%Y-%m")


def _compute_monthly_breakdown(
    transactions: list[dict],
    cutoff_date: date,
    today: date,
) -> dict[str, dict]:
    """
    Erstellt eine monatliche Aufschlüsselung der Transaktionen im Analysefenster.

    Rückgabe: {"2024-07": {"count": 2, "total": 1450.0, "amounts": [800.0, 650.0]}, ...}
    """
    # Alle Monate im Fenster initialisieren (auch Monate ohne Transaktionen)
    breakdown: dict[str, dict] = {}
    cursor = cutoff_date.replace(day=1)
    while cursor <= today:
        label = _month_label(cursor)
        breakdown[label] = {"count": 0, "total": 0.0, "amounts": []}
        # Nächsten Monat berechnen
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    for t in transactions:
        label = _month_label(t["_date_obj"])
        if label in breakdown:
            amt = abs(t.get("betrag", 0.0))
            breakdown[label]["count"]  += 1
            breakdown[label]["total"]  += amt
            breakdown[label]["amounts"].append(amt)

    # amounts-Liste für JSON-Serialisierung runden
    for v in breakdown.values():
        v["total"]   = round(v["total"], 2)
        v["amounts"] = [round(a, 2) for a in v["amounts"]]

    return breakdown


def _poisson_stats(transactions: list[dict], n_months: int) -> dict:
    """
    Berechnet Poisson-Parameter aus der beobachteten Häufigkeit.

    λ (lambda_per_month) = Anzahl Transaktionen / Anzahl Monate im Fenster
    P(≥1)                = 1 - e^(-λ)
    E[N pro Monat]       = λ  (Erwartungswert der Poisson-Verteilung)
    """
    n  = len(transactions)
    lm = round(n / n_months, 4)  # lambda per month
    p_any = round(1.0 - math.exp(-lm), 4)  # P(mindestens 1 Transaktion im Monat)
    return {
        "lambda_per_month": lm,
        "probability_any":  p_any,            # 0.0 – 1.0
        "probability_any_pct": round(p_any * 100, 1),  # als % für Frontend
    }


def _amount_stats(amounts: list[float]) -> dict:
    """
    Berechnet Lage- und Streuungsmasse sowie Konfidenzintervalle.

    Bei n=1: σ=0, CI = Punktschätzung.
    """
    if not amounts:
        return {}

    mu = round(mean(amounts), 2)

    if len(amounts) >= 2:
        sd = round(stdev(amounts), 2)
    else:
        sd = 0.0

    ci90_low  = round(max(0.0, mu - CI_90_Z * sd), 2)
    ci90_high = round(mu + CI_90_Z * sd, 2)
    ci95_low  = round(max(0.0, mu - CI_95_Z * sd), 2)
    ci95_high = round(mu + CI_95_Z * sd, 2)

    return {
        "amount_mean":      mu,
        "amount_std":       sd,
        "amount_min":       round(min(amounts), 2),
        "amount_max":       round(max(amounts), 2),
        "amount_median":    round(sorted(amounts)[len(amounts) // 2], 2),
        "amount_ci_90_low":  ci90_low,
        "amount_ci_90_high": ci90_high,
        "amount_ci_95_low":  ci95_low,
        "amount_ci_95_high": ci95_high,
    }


def _build_forecast_months(
    today: date,
    horizon: int,
    lambda_per_month: float,
    amount_mean: float,
    amount_std: float,
) -> list[dict]:
    """
    Erstellt eine Liste von monatlichen Vorhersagen für die nächsten `horizon` Monate.

    Pro Monat:
      - month          : "YYYY-MM"
      - probability    : P(≥1 Transaktion) – identisch für jeden Monat (stationäres Modell)
      - expected_amount: λ × μ  (Erwartungswert der Gesamtbelastung)
      - ci_90_low/high : Unsicherheitsbereich (Betrag × CI-Faktoren)
    """
    forecast = []
    cursor = today

    for _ in range(horizon):
        # Nächsten Monatsanfang berechnen
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

        expected = round(lambda_per_month * amount_mean, 2)
        p_any    = round(1.0 - math.exp(-lambda_per_month), 4)

        # CI auf Monatsebene: kombiniert Häufigkeits- und Betragsstreuung
        # Vereinfachte Näherung: Breite des Betrags-CI × λ
        ci_low  = round(max(0.0, lambda_per_month * max(0.0, amount_mean - CI_90_Z * amount_std)), 2)
        ci_high = round(lambda_per_month * (amount_mean + CI_90_Z * amount_std), 2)

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
    group_key:    str,
    transactions: list[dict],
    cutoff_date:  date,
    today:        date,
    n_months:     int,
) -> dict:
    """
    Erstellt das vollständige Firestore-Dokument für eine Gruppe.
    """
    amounts = [abs(t.get("betrag", 0.0)) for t in transactions]

    poisson = _poisson_stats(transactions, n_months)
    amounts_stats = _amount_stats(amounts)
    monthly = _compute_monthly_breakdown(transactions, cutoff_date, today)

    # Kategorie und Gegenpartei aus dem Schlüssel ableiten
    if " :: " in group_key:
        parts    = group_key.split(" :: ", 1)
        category = parts[0]
        counterparty = parts[1]
        group_type = "category_counterparty"
    else:
        category     = group_key
        counterparty = None
        group_type   = "category_only"

    # Monatliche Gesamtbelastung schätzen: λ × μ
    amount_expected_monthly = round(
        poisson["lambda_per_month"] * amounts_stats.get("amount_mean", 0.0), 2
    )

    forecast = _build_forecast_months(
        today          = today,
        horizon        = FORECAST_HORIZON,
        lambda_per_month = poisson["lambda_per_month"],
        amount_mean    = amounts_stats.get("amount_mean", 0.0),
        amount_std     = amounts_stats.get("amount_std", 0.0),
    )

    doc = {
        # ── Identifikation ──────────────────────────────────────────
        "group_key":              group_key,
        "group_type":             group_type,  # "category_counterparty" | "category_only"
        "category":               category,
        "counterparty":           counterparty,   # None wenn reine Kategorie-Gruppe

        # ── Metadaten ───────────────────────────────────────────────
        "sample_size":            len(transactions),
        "lookback_months":        n_months,
        "analysis_from":          cutoff_date.isoformat(),
        "analysis_to":            today.isoformat(),
        "calculated_at":          datetime.utcnow().isoformat() + "Z",

        # ── Poisson: Häufigkeit ──────────────────────────────────────
        **poisson,

        # ── Normalverteilung: Beträge ────────────────────────────────
        **amounts_stats,
        "amount_expected_monthly": amount_expected_monthly,

        # ── Monatliche Aufschlüsselung (für Balkendiagramm) ──────────
        "monthly_breakdown":      monthly,

        # ── Vorausschau nächste 6 Monate ─────────────────────────────
        "forecast_months":        forecast,
    }

    return doc


# ─────────────────────────────────────────────
# FIRESTORE SCHREIBEN
# ─────────────────────────────────────────────

def _safe_doc_id(raw: str) -> str:
    """
    Erstellt einen sicheren Firestore-Dokument-ID aus einem beliebigen String.
    Erlaubt: a-z A-Z 0-9 _ -  |  Ersetzt alles andere durch _
    Max. 200 Zeichen.
    """
    import re
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)
    return safe[:200]


def _write_to_firestore(db, documents: list[dict]) -> int:
    """
    Schreibt alle Dokumente in TARGET_COLLECTION.
    Nutzt set() → idempotent, bestehende Dokumente werden überschrieben.
    Gibt die Anzahl geschriebener Dokumente zurück.
    """
    ref = db.collection(TARGET_COLLECTION)
    written = 0

    for doc in documents:
        doc_id = _safe_doc_id(doc["group_key"])
        ref.document(doc_id).set(doc)
        written += 1

    return written


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def calculate_unknown() -> dict:
    """
    Haupteinstiegspunkt – berechnet Wahrscheinlichkeitsverteilungen
    für alle Transaktionen ohne Muster und speichert sie in Firestore.

    Ablauf:
      1. distributions_db lesen (letzte LOOKBACK_MONTHS Monate)
      2. Transaktionen nach Kategorie (+Gegenpartei) gruppieren
      3. Pro Gruppe: Poisson-Parameter + Normalverteilung berechnen
      4. Vorausschau der nächsten FORECAST_HORIZON Monate erstellen
      5. Ergebnisse in forecast_distribution schreiben

    Returns:
        {"groups_written": int, "transactions_analyzed": int}
    """
    db    = _init_firestore()
    today = date.today()

    # Analysefenster: letzter Tag von vor LOOKBACK_MONTHS Monaten
    cutoff_month = today.month - LOOKBACK_MONTHS
    cutoff_year  = today.year
    while cutoff_month <= 0:
        cutoff_month += 12
        cutoff_year  -= 1
    cutoff_date = date(cutoff_year, cutoff_month, 1)

    # Tatsächliche Anzahl voller Monate im Fenster
    n_months = LOOKBACK_MONTHS

    print(f"    Zeitfenster  : {cutoff_date} → {today}  ({n_months} Monate)")

    # 1. Daten laden
    transactions = _load_distributions(db, cutoff_date)
    print(f"    Transaktionen: {len(transactions)} aus '{SOURCE_COLLECTION}'")

    if not transactions:
        print("    ⚠️  Keine Transaktionen gefunden – nichts zu berechnen.")
        return {"groups_written": 0, "transactions_analyzed": 0}

    # 2. Gruppieren
    groups = _build_groups(transactions)
    print(f"    Gruppen      : {len(groups)}")

    # 3. + 4. Dokumente pro Gruppe bauen
    documents = []
    for gkey, txs in groups.items():
        doc = _build_distribution_doc(
            group_key    = gkey,
            transactions = txs,
            cutoff_date  = cutoff_date,
            today        = today,
            n_months     = n_months,
        )
        documents.append(doc)

    # 5. In Firestore speichern
    written = _write_to_firestore(db, documents)
    print(f"    Geschrieben  : {written} Dokumente → '{TARGET_COLLECTION}'")

    return {
        "groups_written":       written,
        "transactions_analyzed": len(transactions),
    }


# ─────────────────────────────────────────────
# KONSOLENAUSGABE (nur CLI)
# ─────────────────────────────────────────────

def _print_group_summary(doc: dict):
    """Gibt eine Zusammenfassung eines Gruppen-Dokuments auf der Konsole aus."""
    SEP = "." * 65
    print(f"\n  {'─'*65}")
    print(f"  {doc['group_key']}")
    print(f"  {SEP}")
    print(f"  Typ            : {doc['group_type']}")
    print(f"  Stichproben    : {doc['sample_size']}  (letzte {doc['lookback_months']} Monate)")
    print(f"  λ/Monat        : {doc['lambda_per_month']}  "
          f"→ P(≥1 Tx/Monat) = {doc['probability_any_pct']}%")
    print(f"  Betrag Ø       : CHF {doc.get('amount_mean', 0):>10,.2f}  "
          f"(σ = {doc.get('amount_std', 0):,.2f})")
    print(f"  90%-CI         : CHF {doc.get('amount_ci_90_low', 0):>10,.2f}  –  "
          f"CHF {doc.get('amount_ci_90_high', 0):,.2f}")
    print(f"  Erw. Monatslast: CHF {doc.get('amount_expected_monthly', 0):>10,.2f}")
    print(f"  Vorausschau    :", end="")
    for fm in doc.get("forecast_months", [])[:3]:
        print(f"  {fm['month']} ({fm['probability_pct']}% / CHF {fm['expected_amount']:,.0f})", end="")
    print(" ...")


def _print_summary_table(documents: list[dict]):
    """Gibt eine sortierte Übersichtstabelle aus."""
    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  FORECAST DISTRIBUTION – ÜBERSICHT  ({len(documents)} Gruppen)")
    print(f"{SEP}")
    print(f"  {'Gruppe':<40}  {'P(≥1)':<7}  {'Ø Betrag':>10}  {'Erw./Mo':>10}")
    print(f"  {'─'*40}  {'─'*7}  {'─'*10}  {'─'*10}")

    # Sortiert nach erwarteter Monatsbelastung (absteigend)
    for doc in sorted(documents, key=lambda d: d.get("amount_expected_monthly", 0), reverse=True):
        gkey  = doc["group_key"][:40]
        p_pct = f"{doc['probability_any_pct']}%"
        mu    = f"CHF {doc.get('amount_mean', 0):>8,.0f}"
        exp   = f"CHF {doc.get('amount_expected_monthly', 0):>8,.0f}"
        print(f"  {gkey:<40}  {p_pct:<7}  {mu:>10}  {exp:>10}")
    print(f"{SEP}")


def main():
    print(f"\n{'='*65}")
    print(f"  CALCULATE UNKNOWN – Wahrscheinlichkeitsverteilung")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    db    = _init_firestore()
    today = date.today()

    cutoff_month = today.month - LOOKBACK_MONTHS
    cutoff_year  = today.year
    while cutoff_month <= 0:
        cutoff_month += 12
        cutoff_year  -= 1
    cutoff_date = date(cutoff_year, cutoff_month, 1)

    print(f"  Lade Daten aus '{SOURCE_COLLECTION}'...")
    print(f"  Zeitfenster: {cutoff_date} → {today}\n")

    transactions = _load_distributions(db, cutoff_date)
    print(f"  ✅ {len(transactions)} Transaktionen geladen")

    if not transactions:
        print("\n  ⚠️  Keine Transaktionen im Zeitfenster – Abbruch.")
        sys.exit(0)

    groups    = _build_groups(transactions)
    documents = []
    for gkey, txs in groups.items():
        doc = _build_distribution_doc(
            group_key    = gkey,
            transactions = txs,
            cutoff_date  = cutoff_date,
            today        = today,
            n_months     = LOOKBACK_MONTHS,
        )
        documents.append(doc)
        _print_group_summary(doc)

    _print_summary_table(documents)

    written = _write_to_firestore(db, documents)

    print(f"\n{'='*65}")
    print(f"  ✅ {written} Dokumente → Firestore '{TARGET_COLLECTION}'")
    print(f"  ✅ {len(transactions)} Transaktionen analysiert")
    print(f"  Abgeschlossen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
