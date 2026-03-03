"""
training/pipeline.py
====================
Orchestriert den Trainings-Workflow:

  1. categorize.py       → data/training.json      → data/tink_categorized.json
  2. detect_patterns.py  → data/tink_categorized.json → data/tink_patterns.json
  3. organisational.py   → data/tink_patterns.json  → Firestore (patterns_db, distributions_db)
  4. calculate_unknown.py→ Firestore distributions_db → Firestore forecast_distribution

Alle Skripte liegen in training/, Daten in data/.

Konfiguration via Env-Variablen:
  INPUT_FILE                      (default: data/training.json)
  SILENT_MODE                     (default: true)
  ANTHROPIC_API_KEY               für Step 1
  GOOGLE_APPLICATION_CREDENTIALS  für Steps 3 + 4
"""

import io
import os
import sys
import time
import pathlib
import importlib.util
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# ─────────────────────────────────────────────
# PFADE
# ─────────────────────────────────────────────

ROOT        = pathlib.Path(__file__).parent          # training/
DATA        = ROOT.parent / "data"                   # data/

INPUT_FILE  = os.environ.get("INPUT_FILE",  str(DATA / "training.json"))
SILENT_MODE = os.environ.get("SILENT_MODE", "true").lower() != "false"

CATEGORIZED = str(DATA / "tink_categorized.json")
PATTERNS    = str(DATA / "tink_patterns.json")
TEMP_FILES  = [CATEGORIZED, PATTERNS]

DATA.mkdir(exist_ok=True)

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
    print(f"  ✅  Step {step} abgeschlossen: {title}  ({elapsed:.1f}s)")
    print(SEP_THIN)


def _fail(step: int, title: str, error: Exception):
    print(f"\n{SEP}")
    print(f"  ❌  FEHLER in Step {step}: {title}")
    print(f"  {type(error).__name__}: {error}")
    print(SEP)
    _cleanup()
    sys.exit(1)


def _check_env(var: str, step_name: str):
    if not os.environ.get(var):
        print(f"\n❌  {var} nicht gesetzt (benötigt für {step_name})")
        print(f"    Lokal  : export {var}='...'")
        print(f"    GitHub : Settings → Secrets → {var}")
        sys.exit(1)


def _check_file(path: str, produced_by: str):
    if not os.path.exists(path):
        raise RuntimeError(
            f"Erwartete Datei nicht gefunden: {pathlib.Path(path).name} "
            f"(sollte von {produced_by} erzeugt werden)"
        )


def _cleanup():
    """Löscht temporäre Zwischendateien mit Transaktionsdaten."""
    for path in TEMP_FILES:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _load(name: str):
    """Lädt ein Skript aus training/ als Modul."""
    path = ROOT / f"{name}.py"
    if not path.exists():
        print(f"\n❌  Skript nicht gefunden: {path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(fn, *args, **kwargs):
    """Führt fn() aus. Im SILENT_MODE wird stdout/stderr unterdrückt."""
    if SILENT_MODE:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            return fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


# ─────────────────────────────────────────────
# STEPS
# ─────────────────────────────────────────────

def run_categorize():
    _check_env("ANTHROPIC_API_KEY", "categorize.py")
    os.environ["INPUT_FILE"]  = INPUT_FILE
    os.environ["OUTPUT_FILE"] = CATEGORIZED

    _header(1, "Kategorisierung")
    t0 = time.time()
    try:
        _run(_load("categorize").main)
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"Exited with code {e.code}")
    _check_file(CATEGORIZED, "categorize.py")
    _ok(1, "Kategorisierung", time.time() - t0)


def run_detect_patterns():
    os.environ["INPUT_FILE"]  = CATEGORIZED
    os.environ["OUTPUT_FILE"] = PATTERNS

    _header(2, "Mustererkennung")
    t0 = time.time()
    try:
        _run(_load("detect_patterns").main)
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"Exited with code {e.code}")
    _check_file(PATTERNS, "detect_patterns.py")
    _ok(2, "Mustererkennung", time.time() - t0)


def run_organisational():
    _check_env("GOOGLE_APPLICATION_CREDENTIALS", "organisational.py")
    os.environ["INPUT_FILE"] = PATTERNS

    _header(3, "Firestore Export")
    t0 = time.time()
    try:
        _run(_load("organisational").main)
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"Exited with code {e.code}")
    _ok(3, "Firestore Export", time.time() - t0)


def run_calculate_unknown():
    _check_env("GOOGLE_APPLICATION_CREDENTIALS", "calculate_unknown.py")

    _header(4, "Liquiditätsprognose")
    t0 = time.time()
    try:
        result = _run(_load("calculate_unknown").calculate_unknown)
        if result:
            print(f"  Gruppen: {result.get('groups_written', '?')}  |  "
                  f"Transaktionen: {result.get('transactions_analyzed', '?')}")
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"Exited with code {e.code}")
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
    print(f"  TRAINING PIPELINE  {'[SILENT]' if SILENT_MODE else '[VERBOSE]'}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Data   : {DATA}/")
    print(SEP)

    timings: list[tuple[str, float]] = []

    try:
        for step_num, step_name, step_fn in STEPS:
            t0 = time.time()
            try:
                step_fn()
            except Exception as e:
                _fail(step_num, step_name, e)
            timings.append((step_name, round(time.time() - t0, 1)))

    finally:
        if SILENT_MODE:
            _cleanup()
            print(f"\n  🗑️   Zwischendateien gelöscht (Datenschutz)")

    total = round(time.time() - pipeline_start, 1)

    print(f"\n{SEP}")
    print(f"  TRAINING ABGESCHLOSSEN  ✅")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    for name, elapsed in timings:
        print(f"  ✅  {name:<28}  {elapsed:>6.1f}s")
    print(SEP_THIN)
    print(f"  {'Gesamt':<32}  {total:>6.1f}s")
    print(SEP)


if __name__ == "__main__":
    main()
