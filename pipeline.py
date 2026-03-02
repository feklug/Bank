"""
pipeline.py
===========
Orchestriert den vollständigen Analyse-Workflow:

  1. categorize.py        → tink_demo_transactions.json → tink_categorized.json
  2. detect_patterns.py   → tink_categorized.json       → tink_patterns.json
  3. organisational.py    → tink_patterns.json          → Firestore (patterns_db, distributions_db)
  4. calculate_unknown.py → Firestore distributions_db  → Firestore forecast_distribution

Alle Skripte liegen im gleichen Verzeichnis wie pipeline.py (Repo-Root).

Konfiguration via Env-Variablen:
  INPUT_FILE                      Pfad zur Rohtransaktions-JSON  (default: tink_demo_transactions.json)
  OUTPUT_DIR                      Ausgabeverzeichnis             (default: . )
  ANTHROPIC_API_KEY               für Step 1
  GOOGLE_APPLICATION_CREDENTIALS  für Steps 3 + 4
"""

import os
import sys
import time
import pathlib
import importlib.util
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

ROOT       = pathlib.Path(__file__).parent          # Verzeichnis aller Skripte
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(ROOT))
INPUT_FILE = os.environ.get("INPUT_FILE", str(ROOT / "tink_demo_transactions.json"))

CATEGORIZED = str(pathlib.Path(OUTPUT_DIR) / "tink_categorized.json")
PATTERNS    = str(pathlib.Path(OUTPUT_DIR) / "tink_patterns.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

SEP      = "=" * 70
SEP_THIN = "─" * 70


def _header(step: int, title: str):
    print(f"\n{SEP}")
    print(f"  STEP {step}/4  ·  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)


def _ok(step: int, title: str, elapsed: float):
    print(f"\n  ✅  Step {step} abgeschlossen: {title}  ({elapsed:.1f}s)")
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
        print(f"    Lokal  : export {var}='...'")
        print(f"    GitHub : Settings → Secrets → {var}")
        sys.exit(1)


def _check_file(path: str, produced_by: str):
    if not os.path.exists(path):
        print(f"\n❌  Erwartete Datei nicht gefunden: {path}")
        print(f"    Wurde von {produced_by} nicht erzeugt.")
        sys.exit(1)
    size = os.path.getsize(path)
    print(f"  📄  {path}  ({size:,} Bytes)")


def _load(name: str):
    """Lädt ein Skript aus dem Root-Verzeichnis als Modul."""
    path = ROOT / f"{name}.py"
    if not path.exists():
        print(f"\n❌  Skript nicht gefunden: {path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────
# STEP 1 – KATEGORISIERUNG
# ─────────────────────────────────────────────

def run_categorize():
    _check_env("ANTHROPIC_API_KEY", "categorize.py")

    os.environ["INPUT_FILE"]  = INPUT_FILE
    os.environ["OUTPUT_FILE"] = CATEGORIZED

    _header(1, "Kategorisierung (categorize.py)")
    t0 = time.time()

    try:
        _load("categorize").main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"categorize.main() exited with code {e.code}") from e

    _check_file(CATEGORIZED, "categorize.py")
    _ok(1, "Kategorisierung", time.time() - t0)


# ─────────────────────────────────────────────
# STEP 2 – MUSTERERKENNUNG
# ─────────────────────────────────────────────

def run_detect_patterns():
    os.environ["INPUT_FILE"]  = CATEGORIZED
    os.environ["OUTPUT_FILE"] = PATTERNS

    _header(2, "Mustererkennung (detect_patterns.py)")
    t0 = time.time()

    try:
        _load("detect_patterns").main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"detect_patterns.main() exited with code {e.code}") from e

    _check_file(PATTERNS, "detect_patterns.py")
    _ok(2, "Mustererkennung", time.time() - t0)


# ─────────────────────────────────────────────
# STEP 3 – FIRESTORE EXPORT
# ─────────────────────────────────────────────

def run_organisational():
    _check_env("GOOGLE_APPLICATION_CREDENTIALS", "organisational.py")

    os.environ["INPUT_FILE"] = PATTERNS

    _header(3, "Firestore Export (organisational.py)")
    t0 = time.time()

    try:
        _load("organisational").main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"organisational.main() exited with code {e.code}") from e

    _ok(3, "Firestore Export", time.time() - t0)


# ─────────────────────────────────────────────
# STEP 4 – WAHRSCHEINLICHKEITSVERTEILUNG
# ─────────────────────────────────────────────

def run_calculate_unknown():
    _check_env("GOOGLE_APPLICATION_CREDENTIALS", "calculate_unknown.py")

    _header(4, "Liquiditätsprognose (calculate_unknown.py)")
    t0 = time.time()

    try:
        result = _load("calculate_unknown").calculate_unknown()
        print(f"\n  Gruppen geschrieben     : {result.get('groups_written', '?')}")
        print(f"  Transaktionen analysiert: {result.get('transactions_analyzed', '?')}")
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"calculate_unknown() exited with code {e.code}") from e

    _ok(4, "Liquiditätsprognose", time.time() - t0)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

STEPS = [
    (1, "Kategorisierung",     run_categorize),
    (2, "Mustererkennung",     run_detect_patterns),
    (3, "Firestore Export",    run_organisational),
    (4, "Liquiditätsprognose", run_calculate_unknown),
]


def main():
    pipeline_start = time.time()

    print(f"\n{SEP}")
    print(f"  PIPELINE START")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input     : {INPUT_FILE}")
    print(f"  Output    : {OUTPUT_DIR}/")
    print(f"  Skripte   : {ROOT}/")
    print(SEP)

    timings: list[tuple[str, float]] = []

    for step_num, step_name, step_fn in STEPS:
        t0 = time.time()
        try:
            step_fn()
        except Exception as e:
            _fail(step_num, step_name, e)
        timings.append((step_name, round(time.time() - t0, 1)))

    total = round(time.time() - pipeline_start, 1)

    print(f"\n{SEP}")
    print(f"  PIPELINE ABGESCHLOSSEN  ✅")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    for name, elapsed in timings:
        print(f"  ✅  {name:<28}  {elapsed:>6.1f}s")
    print(SEP_THIN)
    print(f"  {'Gesamt':<32}  {total:>6.1f}s")
    print(SEP)


if __name__ == "__main__":
    main()
