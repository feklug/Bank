"""
pipeline.py – Modus A
=====================
Orchestriert die komplette Modus-A-Pipeline.
Alle Zwischenergebnisse bleiben im Memory – keine Dateien auf Disk.

Schritte:
  1. Daten laden           → tink_demo_transactions.json
  2. Kategorisieren        → categorize.py        (Claude API)
  3. Pattern Detection     → detect_patterns.py   (Statistik + Firestore)
  4. Verteilungsberechnung → calculate_unknown.py (Poisson + Firestore)

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

    # ── Schritt 3: Pattern Detection + Firestore ──────────────────────────────
    # detect_patterns() übernimmt beides:
    #   a) Muster erkennen  (RECURRING / SEASONAL / SEQUENTIAL)
    #   b) In Firestore speichern
    #      → patterns_db      : ein Dokument pro Muster (inkl. Transaktionen)
    #      → distributions_db : ein Dokument pro Transaktion ohne Muster
    print(f"\n[3/4] Pattern Detection + Firestore Speicherung...")
    from detect_patterns import detect_patterns
    fs_result = detect_patterns(categorized)
    print(f"  ✅ Firestore: {fs_result['total_patterns']} Pattern-Dokumente | "
          f"{fs_result['distributions']} Distributions")

    # ── Schritt 4: Wahrscheinlichkeitsverteilung ──────────────────────────────
    # calculate_unknown() liest distributions_db aus Firestore (Schritt 3 muss
    # zuerst abgeschlossen sein) und berechnet pro Gruppe:
    #   - Poisson-Parameter (λ, P(≥1 Tx/Monat))
    #   - Normalverteilung der Beträge (μ, σ, 90%- / 95%-CI)
    #   - Vorausschau der nächsten 6 Monate
    # Ergebnis → forecast_distribution (für Frontend-Zugriff)
    print(f"\n[4/4] Wahrscheinlichkeitsverteilung berechnen...")
    from calculate_unknown import calculate_unknown
    dist_result = calculate_unknown()
    print(f"  ✅ Firestore: {dist_result['groups_written']} Gruppen | "
          f"{dist_result['transactions_analyzed']} Transaktionen analysiert")

    # ── Abschluss ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ MODUS A ABGESCHLOSSEN")
    print(f"  Transaktionen total       : {len(categorized)}")
    print(f"  Firestore patterns_db     : {fs_result['total_patterns']}")
    print(f"  Firestore distributions_db: {fs_result['distributions']}")
    print(f"  Firestore forecast_dist.  : {dist_result['groups_written']} Gruppen")
    print(f"  Abgeschlossen             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
