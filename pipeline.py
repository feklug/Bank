"""
pipeline.py
===========
Orchestriert den vollständigen Analyse-Workflow:

  1. categorize.py       → tink_demo_transactions.json → tink_categorized.json
  2. detect_patterns.py  → tink_categorized.json       → tink_patterns.json
  3. organisational.py   → tink_patterns.json          → Firestore (patterns_db, distributions_db)
  4. calculate_unknown.py→ Firestore distributions_db  → Firestore forecast_distribution

Konfiguration via Env-Variablen (alle optional, Defaults siehe unten):
  INPUT_FILE     Pfad zur Rohtransaktions-JSON  (default: data/tink_demo_transactions.json)
  OUTPUT_DIR     Ausgabeverzeichnis             (default: data/)
  ANTHROPIC_API_KEY             für Step 1
  GOOGLE_APPLICATION_CREDENTIALS für Steps 3 + 4
"""

import os
import sys
import time
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

INPUT_FILE   = os.environ.get("INPUT_FILE",  "data/tink_demo_transactions.json")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR",  "data")
CATEGORIZED  = os.path.join(OUTPUT_DIR, "tink_categorized.json")
PATTERNS     = os.path.join(OUTPUT_DIR, "tink_patterns.json")

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
    """Bricht ab wenn eine erforderliche Env-Variable fehlt."""
    if not os.environ.get(var):
        print(f"\n❌  {var} nicht gesetzt (benötigt für {step_name})")
        print(f"    Lokal  : export {var}='...'")
        print(f"    GitHub : Settings → Secrets → {var}")
        sys.exit(1)

def _check_file(path: str, produced_by: str):
    """Bricht ab wenn eine erwartete Output-Datei fehlt."""
    if not os.path.exists(path):
        print(f"\n❌  Erwartete Datei nicht gefunden: {path}")
        print(f"    Wurde von {produced_by} nicht erzeugt.")
        sys.exit(1)
    size = os.path.getsize(path)
    print(f"  📄  {path}  ({size:,} Bytes)")

# ─────────────────────────────────────────────
# STEP 1 – KATEGORISIERUNG
# ─────────────────────────────────────────────

def run_categorize():
    _check_env("ANTHROPIC_API_KEY", "categorize.py")

    # Env für das Submodul setzen
    os.environ["INPUT_FILE"]  = INPUT_FILE
    os.environ["OUTPUT_FILE"] = CATEGORIZED

    _header(1, "Kategorisierung (categorize.py)")

    # Import hier, damit Fehler klar dem Step zugeordnet werden können
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "categorize",
        pathlib.Path(__file__).parent / "categorize.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    try:
        mod.main()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"categorize.main() exited with code {e.code}") from e
    elapsed = time.time() - t0

    _check_file(CATEGORIZED, "categorize.py")
    _ok(1, "Kategorisierung", elapsed)


# ─────────────────────────────────────────────
# STEP 2 – MUSTERERKENNUNG
# ─────────────────────────────────────────────

def run_detect_patterns():
    os.environ["INPUT_FILE"]  = CATEGORIZED
    os.environ["OUTPUT_FILE"] = PATTERNS

    _header(2, "Mustererkennung (detect_patterns.py)")

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "detect_patterns",
        pathlib.Path(__file__).parent / "detect_patterns.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    try:
        mod.main()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"detect_patterns.main() exited with code {e.code}") from e
    elapsed = time.time() - t0

    _check_file(PATTERNS, "detect_patterns.py")
    _ok(2, "Mustererkennung", elapsed)


# ─────────────────────────────────────────────
# STEP 3 – FIRESTORE EXPORT (patterns + distributions)
# ─────────────────────────────────────────────

def run_organisational():
    _check_env("GOOGLE_APPLICATION_CREDENTIALS", "organisational.py")

    os.environ["INPUT_FILE"] = PATTERNS

    _header(3, "Firestore Export (organisational.py)")

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "organisational",
        pathlib.Path(__file__).parent / "organisational.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    try:
        mod.main()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"organisational.main() exited with code {e.code}") from e
    elapsed = time.time() - t0

    _ok(3, "Firestore Export", elapsed)


# ─────────────────────────────────────────────
# STEP 4 – WAHRSCHEINLICHKEITSVERTEILUNG
# ─────────────────────────────────────────────

def run_calculate_unknown():
    _check_env("GOOGLE_APPLICATION_CREDENTIALS", "calculate_unknown.py")

    _header(4, "Liquiditätsprognose (calculate_unknown.py)")

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "calculate_unknown",
        pathlib.Path(__file__).parent / "calculate_unknown.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    try:
        result = mod.calculate_unknown()
        print(f"\n  Gruppen geschrieben    : {result.get('groups_written', '?')}")
        print(f"  Transaktionen analysiert: {result.get('transactions_analyzed', '?')}")
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"calculate_unknown() exited with code {e.code}") from e
    elapsed = time.time() - t0

    _ok(4, "Liquiditätsprognose", elapsed)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

STEPS = [
    (1, "Kategorisierung",      run_categorize),
    (2, "Mustererkennung",      run_detect_patterns),
    (3, "Firestore Export",     run_organisational),
    (4, "Liquiditätsprognose",  run_calculate_unknown),
]

def main():
    pipeline_start = time.time()

    print(f"\n{SEP}")
    print(f"  PIPELINE START")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_DIR}/")
    print(SEP)

    timings: list[tuple[str, float]] = []

    for step_num, step_name, step_fn in STEPS:
        t0 = time.time()
        try:
            step_fn()
        except Exception as e:
            _fail(step_num, step_name, e)
        timings.append((step_name, round(time.time() - t0, 1)))

    # ── Abschlussbericht ─────────────────────────────────────────
    total = round(time.time() - pipeline_start, 1)

    print(f"\n{SEP}")
    print(f"  PIPELINE ABGESCHLOSSEN  ✅")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    for name, elapsed in timings:
        print(f"  {'✅':<4} {name:<28}  {elapsed:>6.1f}s")
    print(SEP_THIN)
    print(f"  {'Gesamt':<32}  {total:>6.1f}s")
    print(SEP)


if __name__ == "__main__":
    main()
