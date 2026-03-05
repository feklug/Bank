"""
simulate/track_performance.py
==============================
Trackt die Forecast-Performance: Ist-Kontostand vs. Worst-Case-Forecast
zu 5 EOD-Checkpoints (End-of-Day T+1 bis T+5).

Ablauf:
  1. Referenzpunkt T = Datum der aktuellen TX (aus simulate_state.json)
  2. Ist-Kontostand bei jedem EOD(T+n):
       Balance(T+n) = INITIAL_BALANCE + Σ TX in simulate.json mit datum ≤ T+n
  3. Forecast worst case bei EOD(T+n):
       a) patterns_db:         alle Patterns mit next_expected_date ≤ T+n
                               Ausgaben:  amount_avg + 1.645 × amount_std
                               Einnahmen: amount_avg − 1.645 × amount_std
       b) forecast_distribution: kumulierter ci_90_low(T+1 bis T+n)
       Forecast(T+n) = Balance(T) + Σ Pattern-Worst-Case + Σ ci_90_low
  4. Abweichung = Ist(T+n) − Forecast-Worst-Case(T+n)
       Positiv  → besser als worst case
       Negativ  → schlechter als worst case (Forecast hat Risiko unterschätzt)
  5. Speichern in Firestore: track_performance_db

Konfiguration:
  INITIAL_BALANCE  Kontostand nach 0 simulate.json TX (GitHub Secret / Env-Variable)
  INPUT_FILE       Pfad zu simulate.json (default: data/simulate.json)
  GOOGLE_APPLICATION_CREDENTIALS

Importierbar für simulate/pipeline.py:
  from track_performance import track_performance
  result = track_performance(db, current_tx_date)
"""

import json
import math
import os
import pathlib
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

_ROOT = pathlib.Path(__file__).parent
_DATA = _ROOT.parent / "data"

INPUT_FILE      = os.environ.get("INPUT_FILE", str(_DATA / "simulate.json"))
STATE_FILE      = str(_DATA / "simulate_state.json")
INITIAL_BALANCE = float(os.environ.get("INITIAL_BALANCE", "0.0"))

COLLECTION_PATTERNS      = "patterns_db"
COLLECTION_FORECAST      = "forecast_distribution"
COLLECTION_PERFORMANCE   = "track_performance_db"

CI_90_Z = 1.645
CHECKPOINTS = [1, 2, 3, 4, 5]   # EOD T+1 bis T+5


# ─────────────────────────────────────────────
# DATUM-HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def _parse_date(s) -> Optional[date]:
    if not s:
        return None
    s = str(s).strip().split("T")[0].split(" ")[0]
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────
# 1. IST-KONTOSTAND AUS SIMULATE.JSON
# ─────────────────────────────────────────────

def _load_simulate_transactions() -> list[dict]:
    """Lädt alle Transaktionen aus simulate.json chronologisch sortiert."""
    path = pathlib.Path(INPUT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"simulate.json nicht gefunden: {INPUT_FILE}")
    with open(path, encoding="utf-8") as f:
        txs = json.load(f)
    result = []
    for tx in txs:
        d = _parse_date(tx.get("datum", ""))
        if d:
            result.append({**tx, "_date": d})
    result.sort(key=lambda t: t["_date"])
    return result


def _compute_balance_at(
    transactions: list[dict],
    up_to_date: date,
    initial_balance: float,
) -> float:
    """
    Berechnet den Kontostand am EOD eines Datums.
    Balance = INITIAL_BALANCE + Σ betrag aller TX mit datum ≤ up_to_date
    """
    total = sum(
        float(tx.get("betrag", 0))
        for tx in transactions
        if tx["_date"] <= up_to_date
    )
    return round(initial_balance + total, 2)


# ─────────────────────────────────────────────
# 2. FORECAST WORST CASE
# ─────────────────────────────────────────────

def _patterns_worst_case(
    db,
    reference_date: date,
    checkpoint_date: date,
) -> float:
    """
    Summiert den Worst-Case-Beitrag aller Patterns die zwischen
    reference_date (exklusiv) und checkpoint_date (inklusiv) erwartet werden.

    Ausgaben (betrag < 0 oder amount_avg bei OUTFLOW):
        Worst case = −(amount_avg + 1.645 × amount_std)
    Einnahmen (INFLOW):
        Worst case = +(amount_avg − 1.645 × amount_std)  [min 0]
    """
    pattern_docs = db.collection(COLLECTION_PATTERNS).stream()
    total = 0.0

    for doc in pattern_docs:
        p = doc.to_dict()

        # Inaktive Patterns ignorieren
        if p.get("status") == "INACTIVE":
            continue

        next_exp = _parse_date(p.get("next_expected_date", ""))
        if not next_exp:
            continue
        if not (reference_date < next_exp <= checkpoint_date):
            continue

        avg = float(p.get("amount_avg", 0.0))
        std = float(p.get("amount_std", 0.0))
        wc  = avg + CI_90_Z * std   # Worst-case Betrag (immer positiv)

        # Richtung aus Kategorie oder erstem TX-Eintrag ableiten
        category  = p.get("category", p.get("category_level1", ""))
        is_income = category.startswith("EINNAHMEN")

        if is_income:
            # Einnahme worst case: weniger rein als erwartet (min 0)
            total += max(0.0, avg - CI_90_Z * std)
        else:
            # Ausgabe worst case: mehr raus als erwartet
            total -= wc

    return round(total, 2)


def _forecast_distribution_worst_case(
    db,
    reference_date: date,
    checkpoint_date: date,
) -> float:
    """
    Summiert ci_90_low aus forecast_distribution für alle Tage
    zwischen reference_date (exklusiv) und checkpoint_date (inklusiv).

    ci_90_low ist bereits signed (negativ für OUTFLOW, positiv für INFLOW).
    """
    forecast_docs = db.collection(COLLECTION_FORECAST).stream()
    total = 0.0

    for doc in forecast_docs:
        fd = doc.to_dict()
        daily = fd.get("daily_forecast", [])
        direction = fd.get("direction", "OUTFLOW")

        for entry in daily:
            d = _parse_date(entry.get("date", ""))
            if not d:
                continue
            if not (reference_date < d <= checkpoint_date):
                continue

            ci_low = float(entry.get("ci_90_low", 0.0))

            # ci_90_low ist immer positiv in der DB — Vorzeichen aus direction
            if direction == "OUTFLOW":
                total -= ci_low
            elif direction == "INFLOW":
                total += ci_low
            # MIXED: ci_90_low ignorieren (zu unsicher)

    return round(total, 2)


def _compute_forecast_worst_case(
    db,
    balance_at_t: float,
    reference_date: date,
    checkpoint_date: date,
) -> float:
    """
    Gesamter Worst-Case-Forecast für EOD(checkpoint_date):
    Balance(T) + Pattern-Worst-Case + forecast_distribution-Worst-Case
    """
    pattern_wc  = _patterns_worst_case(db, reference_date, checkpoint_date)
    forecast_wc = _forecast_distribution_worst_case(db, reference_date, checkpoint_date)
    return round(balance_at_t + pattern_wc + forecast_wc, 2)


# ─────────────────────────────────────────────
# 3. FIRESTORE SCHREIBEN
# ─────────────────────────────────────────────

def _make_doc_id(reference_date: date, tx_index: int) -> str:
    return f"{reference_date.isoformat()}_{tx_index:05d}"


def _write_performance(db, doc: dict):
    doc_id = _make_doc_id(
        _parse_date(doc["reference_date"]),
        doc.get("tx_index", 0),
    )
    db.collection(COLLECTION_PERFORMANCE).document(doc_id).set(doc)
    return doc_id


# ─────────────────────────────────────────────
# HAUPT-FUNKTION  (importierbar)
# ─────────────────────────────────────────────

def track_performance(
    db,
    reference_date: date,
    tx_index: int,
    initial_balance: Optional[float] = None,
) -> dict:
    """
    Berechnet und speichert die Forecast-Performance für die aktuelle TX.

    Parameter:
        db               : Firestore-Client
        reference_date   : Datum der aktuellen TX (= Referenzpunkt T)
        tx_index         : Index der TX in simulate.json (für Doc-ID)
        initial_balance  : Überschreibt INITIAL_BALANCE Env-Variable

    Rückgabe:
        {
          "doc_id":      str,
          "checkpoints": [ { "label", "date", "ist", "forecast_wc", "delta" } ],
        }
    """
    balance = initial_balance if initial_balance is not None else INITIAL_BALANCE

    # ── Transaktionen laden ───────────────────────────────────────
    transactions  = _load_simulate_transactions()

    # ── Balance bei T (Referenzpunkt) ────────────────────────────
    balance_at_t  = _compute_balance_at(transactions, reference_date, balance)

    # ── 5 Checkpoints berechnen ───────────────────────────────────
    checkpoints = []
    for n in CHECKPOINTS:
        checkpoint_date = reference_date + timedelta(days=n)
        label           = f"EOD T+{n}"

        ist_value       = _compute_balance_at(transactions, checkpoint_date, balance)
        forecast_wc     = _compute_forecast_worst_case(
            db, balance_at_t, reference_date, checkpoint_date
        )
        delta           = round(ist_value - forecast_wc, 2)

        checkpoints.append({
            "label":        label,
            "date":         checkpoint_date.isoformat(),
            "ist":          ist_value,
            "forecast_wc":  forecast_wc,
            "delta":        delta,
            "above_wc":     delta >= 0,   # True = besser als worst case
        })

    # ── Zusammenfassung ───────────────────────────────────────────
    deltas     = [cp["delta"] for cp in checkpoints]
    worst_cp   = min(checkpoints, key=lambda cp: cp["delta"])
    all_above  = all(cp["above_wc"] for cp in checkpoints)

    doc = {
        "reference_date":   reference_date.isoformat(),
        "tx_index":         tx_index,
        "balance_at_t":     balance_at_t,
        "initial_balance":  balance,
        "calculated_at":    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoints":      checkpoints,
        "worst_delta":      round(min(deltas), 2),
        "worst_delta_at":   worst_cp["label"],
        "all_above_wc":     all_above,
        "mean_delta":       round(sum(deltas) / len(deltas), 2),
    }

    doc_id = _write_performance(db, doc)

    return {
        "doc_id":      doc_id,
        "checkpoints": checkpoints,
        "all_above_wc": all_above,
        "worst_delta":  doc["worst_delta"],
    }


# ─────────────────────────────────────────────
# DIREKTAUFRUF  (für Tests)
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

    # State laden für reference_date und tx_index
    state_path = pathlib.Path(STATE_FILE)
    if not state_path.exists():
        print("❌  simulate_state.json nicht gefunden. Pipeline zuerst ausführen.")
        raise SystemExit(1)

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    tx_index = max(0, state.get("next_index", 1) - 1)   # letzter verarbeiteter Index

    # Datum der letzten TX aus simulate.json holen
    transactions = _load_simulate_transactions()
    if tx_index >= len(transactions):
        print(f"❌  TX-Index {tx_index} außerhalb simulate.json ({len(transactions)} TX).")
        raise SystemExit(1)

    reference_date = transactions[tx_index]["_date"]
    init_bal       = INITIAL_BALANCE

    print(f"\n{'='*65}")
    print(f"  TRACK PERFORMANCE")
    print(f"  Referenz     : {reference_date.isoformat()}  (TX #{tx_index})")
    print(f"  Startkontostand : EUR {init_bal:,.2f}")
    print(f"{'='*65}\n")

    result = track_performance(db, reference_date, tx_index, init_bal)

    print(f"  {'Checkpoint':<12}  {'Datum':<12}  {'Ist':>12}  {'Forecast WC':>12}  {'Δ':>10}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")
    for cp in result["checkpoints"]:
        flag = "✅" if cp["above_wc"] else "❌"
        print(
            f"  {cp['label']:<12}  {cp['date']:<12}  "
            f"EUR {cp['ist']:>8,.2f}  "
            f"EUR {cp['forecast_wc']:>8,.2f}  "
            f"{flag} {cp['delta']:>+8,.2f}"
        )

    print(f"\n  Schlechtester Punkt : {result['worst_delta_at']}  "
          f"Δ {result['worst_delta']:>+,.2f} EUR")
    print(f"  Alle über Worst Case: {'✅ Ja' if result['all_above_wc'] else '❌ Nein'}")
    print(f"\n  → Firestore: {COLLECTION_PERFORMANCE}/{result['doc_id']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
