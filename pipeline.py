"""
pipeline.py – Modus A
=====================
Orchestriert die komplette Modus-A-Pipeline.
Alle Zwischenergebnisse bleiben im Memory – keine Dateien auf Disk.

Schritte:
  1. Daten laden        → tink_demo_transactions.json
  2. Kategorisieren     → categorize.py       (Claude API)
  3. Pattern Detection  → detect_patterns.py  (Statistik, kein KI)
  4. Organisieren       → organisational.py   (Firestore)

Verwendung:
  python pipeline.py
"""

import json
import os
import sys
from datetime import datetime


def main():
    print("=" * 70)
    print("  SWEEPY – MODUS A PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Schritt 1: Daten laden ────────────────────────────────────────────────
    INPUT_FILE = "tink_demo_transactions.json"
    print(f"\n[1/4] Daten laden aus {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"  ❌ Datei nicht gefunden: {INPUT_FILE}")
        print(f"     Bitte sicherstellen dass {INPUT_FILE} im Repo liegt.")
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        transactions = json.load(f)
    print(f"  ✅ {len(transactions)} Transaktionen geladen")

    # ── Schritt 2: Kategorisieren ─────────────────────────────────────────────
    print(f"\n[2/4] Kategorisierung via Claude API...")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ❌ ANTHROPIC_API_KEY nicht gesetzt")
        print("     GitHub: Settings → Secrets → ANTHROPIC_API_KEY")
        print("     Lokal:  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    from categorize import categorize
    categorized = categorize(transactions, api_key=api_key)
    print(f"  ✅ {len(categorized)} Transaktionen kategorisiert")

    # ── Schritt 3: Pattern Detection ──────────────────────────────────────────
    print(f"\n[3/4] Pattern Detection...")
    from detect_patterns import detect_patterns
    result = detect_patterns(categorized)
    print(f"  ✅ Pattern Detection abgeschlossen")

    # ── Schritt 4: Organisieren + Firestore ───────────────────────────────────
    print(f"\n[4/4] Firestore Speicherung...")
    from organisational import save_to_firestore
    fs_result = save_to_firestore(result)
    print(f"  ✅ Firestore: {fs_result['total_patterns']} Pattern-Dokumente | "
          f"{fs_result['distributions']} Distributions")

    # ── Abschluss ─────────────────────────────────────────────────────────────
    no_pat = sum(1 for t in result if not any([
        t["pattern"]["is_batch"],     t["pattern"]["is_recurring"],
        t["pattern"]["is_seasonal"],  t["pattern"]["is_sequential"],
        t["pattern"]["is_counter"],   t["pattern"]["is_anomaly"],
    ]))

    print("\n" + "=" * 70)
    print("  ✅ MODUS A ABGESCHLOSSEN")
    print(f"  Transaktionen total   : {len(result)}")
    print(f"  Mit Pattern           : {len(result) - no_pat}")
    print(f"  Ohne Pattern          : {no_pat}")
    print(f"  Firestore Patterns    : {fs_result['total_patterns']}")
    print(f"  Firestore Distrib.    : {fs_result['distributions']}")
    print(f"  Abgeschlossen         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
