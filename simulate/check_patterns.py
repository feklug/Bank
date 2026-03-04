"""
simulate/check_patterns.py
===========================
Prüft ob Patterns in patterns_db noch aktiv sind.

Für jedes Pattern wird geprüft ob next_expected_date eingehalten wurde.
Dabei gelten dieselben Toleranzwerte wie in is_there_a_pattern.py.

Status-Übergänge:
  ACTIVE   → next_expected_date liegt in der Zukunft
             ODER ist höchstens 1× Toleranz überfällig
             (Zahlung könnte noch kommen)

  OVERDUE  → 1× bis 2× Toleranz überfällig
             Pattern wird geflaggt, aber nicht deaktiviert.
             Nächster Match reaktiviert es automatisch.

  INACTIVE → mehr als 2× Toleranz überfällig
             Pattern wird als inaktiv markiert.
             is_there_a_pattern.py ignoriert INACTIVE-Patterns.

Beispiel (MONTHLY, Toleranz ±5 Tage):
  next_expected_date = 2025-02-01
  Heute              = 2025-02-05  → noch ACTIVE  (4 Tage ≤ 5)
  Heute              = 2025-02-08  → OVERDUE       (7 Tage, 5–10)
  Heute              = 2025-02-12  → INACTIVE      (11 Tage > 10)

Importierbar für simulate/pipeline.py:
  from check_patterns import check_patterns
  result = check_patterns(db)
  # result = {"active": int, "overdue": int, "inactive": int, "skipped": int}
"""

import os
import pathlib
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# ─────────────────────────────────────────────
# KONSTANTEN  (identisch mit is_there_a_pattern.py)
# ─────────────────────────────────────────────

COLLECTION_PATTERNS = "patterns_db"

DATE_TOLERANCE_BY_INTERVAL: dict[str, int] = {
    "WEEKLY":      5,
    "BIWEEKLY":    5,
    "MONTHLY":     5,
    "BIMONTHLY":   5,
    "QUARTERLY":   5,
    "SEMIANNUAL":  5,
    "ANNUAL":      5,
    "CUSTOM":      5,
}
DATE_TOLERANCE_FALLBACK = 5


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    if not s or s == "-":
        return None
    s = str(s).strip().split(" ")[0]   # "2025-02-01 09:14:00" → "2025-02-01"
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _date_tolerance(interval_label: Optional[str]) -> int:
    """Gibt die Datums-Toleranz in Tagen zurück."""
    if not interval_label:
        return DATE_TOLERANCE_FALLBACK

    label = str(interval_label).split(" ")[0].upper()   # "MONTHLY (~30d)" → "MONTHLY"

    if label in DATE_TOLERANCE_BY_INTERVAL:
        return DATE_TOLERANCE_BY_INTERVAL[label]

    return DATE_TOLERANCE_FALLBACK


def _classify(
    next_expected: date,
    today: date,
    tolerance: int,
) -> str:
    """
    Klassifiziert ein Pattern basierend auf wie weit next_expected_date
    in der Vergangenheit liegt.

      days_overdue ≤ tolerance      → ACTIVE   (noch innerhalb Toleranz)
      tolerance < days_overdue ≤ 2× → OVERDUE  (überfällig, aber vielleicht Verzug)
      days_overdue > 2× tolerance   → INACTIVE (Pattern scheint inaktiv)
    """
    days_overdue = (today - next_expected).days

    if days_overdue <= tolerance:
        return "ACTIVE"
    elif days_overdue <= tolerance * 2:
        return "OVERDUE"
    else:
        return "INACTIVE"


# ─────────────────────────────────────────────
# HAUPT-FUNKTION  (importierbar)
# ─────────────────────────────────────────────

def check_patterns(db, today: Optional[date] = None) -> dict:
    """
    Prüft alle Patterns in patterns_db und aktualisiert ihren Status.

    Parameter:
        db    : Firestore-Client
        today : Referenzdatum (default: date.today())

    Rückgabe:
        {
          "active":   int,   # unverändert aktive Patterns
          "overdue":  int,   # neu als OVERDUE markiert
          "inactive": int,   # neu als INACTIVE markiert
          "skipped":  int,   # kein next_expected_date → übersprungen
        }
    """
    if today is None:
        today = date.today()

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    pattern_docs = list(db.collection(COLLECTION_PATTERNS).stream())

    counts = {"active": 0, "overdue": 0, "inactive": 0, "skipped": 0}

    for doc in pattern_docs:
        pattern = doc.to_dict()

        # Patterns die bereits manuell deaktiviert wurden → nicht anfassen
        if pattern.get("status") == "DISABLED":
            counts["skipped"] += 1
            continue

        next_exp_str  = pattern.get("next_expected_date", "") or ""
        next_exp_date = _parse_date(next_exp_str)

        # Kein next_expected_date → kann nicht bewertet werden
        if not next_exp_date:
            counts["skipped"] += 1
            continue

        # Pattern liegt noch in der Zukunft → immer ACTIVE, nichts tun
        if next_exp_date > today:
            counts["active"] += 1
            # Falls es vorher OVERDUE/INACTIVE war und inzwischen ein neues
            # next_expected_date gesetzt wurde → zurück auf ACTIVE
            if pattern.get("status") in ("OVERDUE", "INACTIVE"):
                doc.reference.update({
                    "status":         "ACTIVE",
                    "status_checked": now_iso,
                })
            continue

        interval_label = pattern.get("recurrence_interval", "")
        interval_days  = pattern.get("recurrence_interval_days")
        tolerance      = _date_tolerance(interval_label)

        new_status     = _classify(next_exp_date, today, tolerance)
        current_status = pattern.get("status", "ACTIVE")

        counts[new_status.lower()] += 1

        # Nur schreiben wenn sich der Status geändert hat
        if new_status != current_status:
            update = {
                "status":         new_status,
                "status_checked": now_iso,
            }
            if new_status == "INACTIVE":
                days_overdue = (today - next_exp_date).days
                update["inactive_since"]  = now_iso
                update["days_overdue"]    = days_overdue
            elif new_status == "OVERDUE":
                days_overdue = (today - next_exp_date).days
                update["days_overdue"]    = days_overdue

            doc.reference.update(update)

    return counts


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

    today = date.today()
    print(f"\n{'='*60}")
    print(f"  CHECK PATTERNS  –  {today.isoformat()}")
    print(f"{'='*60}\n")

    result = check_patterns(db, today)

    total = sum(result.values())
    print(f"  Patterns geprüft : {total}")
    print(f"  ✅  ACTIVE        : {result['active']}")
    print(f"  ⚠️   OVERDUE       : {result['overdue']}")
    print(f"  ❌  INACTIVE      : {result['inactive']}")
    print(f"  –   Übersprungen  : {result['skipped']}")
    print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    main()
