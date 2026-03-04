"""
simulate/pipeline.py
====================
Verarbeitet pro Run GENAU EINE Transaktion aus data/simulate.json.

State wird in data/simulate_state.json gespeichert:
  { "next_index": 42, "total": 312, "last_run": "2025-03-03T21:00:00Z" }

Ablauf pro Run:
  1. State laden → TX[next_index] aus simulate.json holen
  2. categorize_simulation.py  → TX kategorisieren
  3. is_there_a_pattern.py     → Match? patterns_db / distributions_db
  4. detect_patterns2.py       → Neue Muster in distributions_db?
  5. calculate_unknown.py      → forecast_distribution aktualisieren
  6. State speichern: next_index + 1

Manuell:    workflow_dispatch
Zukünftig:  schedule cron (alle 5 Minuten)

Konfiguration via Env-Variablen:
  INPUT_FILE                      (default: data/simulate.json)
  SILENT_MODE                     (default: true)
  ANTHROPIC_API_KEY               für Step 1
  GOOGLE_APPLICATION_CREDENTIALS  für Steps 2, 3, 4
"""

import io
import json
import os
import sys
import time
import pathlib
import importlib.util
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# PFADE
# ─────────────────────────────────────────────

ROOT       = pathlib.Path(__file__).parent       # simulate/
DATA       = ROOT.parent / "data"                # data/

INPUT_FILE  = os.environ.get("INPUT_FILE", str(DATA / "simulate.json"))
STATE_FILE  = str(DATA / "simulate_state.json")
SILENT_MODE = os.environ.get("SILENT_MODE", "true").lower() != "false"

DATA.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# STATE  (welche TX kommt als nächstes)
# ─────────────────────────────────────────────

def _load_state(total: int) -> dict:
    """Lädt State aus simulate_state.json. Erstellt neuen State wenn nicht vorhanden."""
    state_path = pathlib.Path(STATE_FILE)
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            # total aktualisieren falls simulate.json geändert wurde
            state["total"] = total
            return state
        except (json.JSONDecodeError, OSError):
            pass
    return {"next_index": 0, "total": total, "last_run": None}


def _save_state(state: dict):
    """Speichert aktualisierten State."""
    state["last_run"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

SEP      = "=" * 70
SEP_THIN = "─" * 70


def _header(step: int, title: str):
    print(f"\n{SEP}")
    print(f"  STEP {step}/5  ·  {title}")
    print(SEP)


def _ok(step: int, title: str, elapsed: float, extra: str = ""):
    msg = f"  ✅  Step {step}: {title}  ({elapsed:.1f}s)"
    if extra:
        msg += f"  |  {extra}"
    print(msg)
    print(SEP_THIN)


def _fail(step: int, title: str, error: Exception):
    print(f"\n{SEP}")
    print(f"  ❌  FEHLER in Step {step}: {title}")
    print(f"  {type(error).__name__}: {error}")
    print(SEP)
    sys.exit(1)


def _check_env(var: str, step_name: str):
    if not os.environ.get(var):
        print(f"\n❌  {var} nicht gesetzt (benötigt für {step_name})")
        sys.exit(1)


def _load_module(name: str):
    """Lädt Modul aus simulate/ — Fallback auf training/ für calculate_unknown."""
    path = ROOT / f"{name}.py"
    if not path.exists():
        path = ROOT.parent / "training" / f"{name}.py"
    if not path.exists():
        print(f"\n❌  Skript nicht gefunden: {name}.py")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_silent(fn, *args, **kwargs):
    if SILENT_MODE:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            return fn(*args, **kwargs)
    return fn(*args, **kwargs)


def _init_firestore():
    import firebase_admin
    from firebase_admin import credentials, firestore as fs
    if not firebase_admin._apps:
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not creds_path or not pathlib.Path(creds_path).exists():
            raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS fehlt: {creds_path}")
        firebase_admin.initialize_app(credentials.Certificate(creds_path))
    return fs.client()


# ─────────────────────────────────────────────
# STEPS
# ─────────────────────────────────────────────

def step5_check_patterns(db) -> dict:
    """Prüft ob bestehende Patterns noch aktiv sind."""
    _header(5, "Pattern-Status prüfen")
    t0 = time.time()

    result = _run_silent(
        _load_module("check_patterns").check_patterns, db
    )

    parts = []
    if result.get("overdue"):
        parts.append(f"⚠️  {result['overdue']} OVERDUE")
    if result.get("inactive"):
        parts.append(f"❌  {result['inactive']} INACTIVE")
    if not parts:
        parts.append(f"alle {result.get('active', 0)} aktiv")

    _ok(5, "Pattern-Status", time.time() - t0, extra="  ".join(parts))
    return result


def step1_categorize(tx: dict) -> dict:
    """Kategorisiert die eine TX."""
    _check_env("ANTHROPIC_API_KEY", "categorize_simulation.py")
    _header(1, "Kategorisierung")
    t0 = time.time()

    result = _run_silent(_load_module("categorize_simulation").categorize_one, tx)

    cat  = result.get("category_level1", "?")
    conf = result.get("confidence", 0.0)
    _ok(1, "Kategorisierung", time.time() - t0,
        extra=f"{cat}  conf: {conf:.2f}")
    return result


def step2_pattern_check(db, tx: dict) -> bool:
    """Prüft ob TX zu einem bekannten Pattern passt."""
    _header(2, "Pattern-Check")
    t0 = time.time()

    matched, doc_id = _run_silent(
        _load_module("is_there_a_pattern").check_pattern, db, tx
    )

    if matched:
        _ok(2, "Pattern-Check", time.time() - t0,
            extra=f"✅ Match → patterns_db/{doc_id}")
    else:
        _ok(2, "Pattern-Check", time.time() - t0,
            extra="➡️  Kein Match → distributions_db")
    return matched


def step3_detect_patterns(db) -> dict:
    """Sucht neue Muster in distributions_db."""
    _header(3, "Neue Muster erkennen")
    t0 = time.time()

    result = _run_silent(
        _load_module("detect_patterns2").detect_new_patterns, db
    )

    n_new  = result.get("new_patterns", 0)
    n_moved = result.get("tx_moved", 0)
    extra = (
        f"{n_new} neues Muster  |  {n_moved} TX verschoben"
        if n_new > 0 else "keine neuen Muster"
    )
    _ok(3, "Neue Muster", time.time() - t0, extra=extra)
    return result


def step4_forecast(db) -> dict:
    """Aktualisiert den Forecast."""
    _header(4, "Forecast aktualisieren")
    t0 = time.time()

    result = _run_silent(_load_module("calculate_unknown").calculate_unknown)

    extra = (
        f"{result.get('groups_written','?')} Gruppen  |  "
        f"{result.get('transactions_analyzed','?')} TX"
    ) if result else ""
    _ok(4, "Forecast", time.time() - t0, extra=extra)
    return result or {}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    pipeline_start = time.time()

    # ── simulate.json laden ───────────────────────────────────────
    if not pathlib.Path(INPUT_FILE).exists():
        print(f"\n❌  simulate.json nicht gefunden: {INPUT_FILE}")
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        all_transactions = json.load(f)

    # Chronologisch sortieren
    def _datum(t):
        return (
            t.get("dates", {}).get("bookedDateTime")
            or t.get("dates", {}).get("booked")
            or t.get("datum", "")
        )
    all_transactions.sort(key=_datum)
    total = len(all_transactions)

    # ── State laden ───────────────────────────────────────────────
    state = _load_state(total)
    idx   = state["next_index"]

    print(f"\n{SEP}")
    print(f"  SIMULATE PIPELINE  {'[SILENT]' if SILENT_MODE else '[VERBOSE]'}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TX {idx + 1}/{total}  (Index {idx})")
    print(SEP)

    # ── Simulation abgeschlossen? ─────────────────────────────────
    if idx >= total:
        print(f"\n  🏁  Alle {total} Transaktionen wurden bereits simuliert.")
        print(f"      State zurücksetzen: data/simulate_state.json löschen.\n")
        sys.exit(0)

    # ── Aktuelle TX holen ─────────────────────────────────────────
    tx = all_transactions[idx]
    tx_datum = _datum(tx)[:10] if _datum(tx) else "?"

    print(f"  Datum    : {tx_datum}")
    if not SILENT_MODE:
        print(f"  Betrag   : {tx.get('betrag', '?')} EUR")
        print(f"  Gegenp.  : {tx.get('gegenpartei', '?')[:40]}")

    # ── Firestore initialisieren ──────────────────────────────────
    _check_env("GOOGLE_APPLICATION_CREDENTIALS", "Firestore")
    try:
        db = _init_firestore()
    except Exception as e:
        print(f"\n❌  Firestore-Init fehlgeschlagen: {e}")
        sys.exit(1)

    # ── Pipeline ausführen ────────────────────────────────────────
    try:
        tx_cat = step1_categorize(tx)
        step2_pattern_check(db, tx_cat)
        step3_detect_patterns(db)
        step4_forecast(db)
        step5_check_patterns(db)
    except Exception as e:
        print(f"\n❌  Pipeline-Fehler: {e}")
        sys.exit(1)

    # ── State aktualisieren ───────────────────────────────────────
    state["next_index"] = idx + 1
    _save_state(state)

    total_elapsed = round(time.time() - pipeline_start, 1)

    remaining = total - (idx + 1)
    print(f"\n{SEP}")
    print(f"  ✅  TX {idx + 1}/{total} verarbeitet  ({total_elapsed:.1f}s)")
    print(f"  📊  Verbleibend: {remaining} TX")
    if remaining > 0:
        print(f"  ▶️   Nächste TX: Index {idx + 1}  ({all_transactions[idx + 1].get('dates', {}).get('bookedDateTime', '?')[:10] if idx + 1 < total else '–'})")
    else:
        print(f"  🏁  Simulation abgeschlossen!")
    print(SEP)


if __name__ == "__main__":
    main()
