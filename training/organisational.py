"""
organisational.py
=================
Liest die Analyse-Ergebnisse von detect_patterns.py (tink_patterns.json)
und speichert sie in Firestore.

Sammlungen:
  patterns_db      – Jedes erkannte Muster als eigenes Dokument (inkl. aller
                     zugehörigen Transaktionen als eingebettetes Array).
  distributions_db – Jede Transaktion OHNE Muster als eigenes Dokument.

Konfiguration via Env-Variablen:
  INPUT_FILE                    Pfad zur patterns JSON (default: tink_patterns.json)
  GOOGLE_APPLICATION_CREDENTIALS Pfad zur Firebase Service Account JSON

Aufruf lokal:
  export GOOGLE_APPLICATION_CREDENTIALS=/pfad/zu/serviceaccount.json
  python3 scripts/organisational.py

Aufruf GitHub Actions:
  Wird automatisch von organisational.yml konfiguriert.

Voraussetzungen:
  pip install google-cloud-firestore
"""

import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime
from typing import Any

from google.cloud import firestore

# ─────────────────────────────────────────────
# KONFIGURATION  (Env-Variablen überschreiben Defaults)
# ─────────────────────────────────────────────

_ROOT = pathlib.Path(__file__).parent        # training/
_DATA = _ROOT.parent / "data"                # data/

INPUT_FILE               = os.environ.get("INPUT_FILE",  str(_DATA / "tink_patterns.json"))
COLLECTION_PATTERNS      = "patterns_db"
COLLECTION_DISTRIBUTIONS = "distributions_db"


# ─────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────

def load_patterns(path: str) -> dict:
    """
    Lädt tink_patterns.json (Output von detect_patterns.py).

    Erwartet die Struktur:
    {
      "meta":       { ... },
      "summary":    { ... },
      "recurring":  [ { ...pattern, "transactions": [...] } ],
      "seasonal":   [ ... ],
      "sequential": [ ... ],
      "no_pattern": [ { ...transaktion } ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def _clean(obj: Any) -> Any:
    """
    Bereitet ein beliebiges Python-Objekt für Firestore vor:
      - None      → None  (Firestore speichert als null)
      - dict/list → rekursiv bereinigt
      - Schlüssel mit "_" am Anfang → werden übersprungen (interne Felder)
    """
    if isinstance(obj, dict):
        return {
            k: _clean(v)
            for k, v in obj.items()
            if not k.startswith("_")
        }
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    return obj


def _make_doc_id(seed: str) -> str:
    """Erzeugt eine stabile, eindeutige Dokument-ID aus einem Seed-String."""
    return hashlib.sha1(seed.encode()).hexdigest()[:20]


def _pattern_to_doc(pattern: dict, pattern_index: int) -> tuple[str, dict]:
    """
    Wandelt ein Pattern-Dict in (doc_id, firestore_doc) um.
    Die Transaktionen liegen bereits sauber als Array 'transactions' vor
    (so wie detect_patterns.py sie exportiert hat).
    """
    transactions = pattern.get("transactions", [])

    doc = {
        **_clean({k: v for k, v in pattern.items() if k != "transactions"}),
        "transactions":      transactions,
        "transaction_count": len(transactions),
        "stored_at":         datetime.utcnow().isoformat() + "Z",
    }

    first_datum = transactions[0]["datum"] if transactions else str(pattern_index)
    id_seed     = f"{pattern.get('pattern_type', '')}|{pattern.get('gegenpartei', '')}|{first_datum}"
    doc_id      = _make_doc_id(id_seed)

    return doc_id, doc


def _tx_to_standalone_doc(tx: dict) -> tuple[str, dict]:
    """Wandelt eine Transaktion ohne Muster in (doc_id, firestore_doc) um."""
    doc = {
        **_clean(tx),
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }
    id_seed = f"{tx.get('datum','')}|{tx.get('betrag','')}|{tx.get('gegenpartei','')}"
    doc_id  = _make_doc_id(id_seed)
    return doc_id, doc


# ─────────────────────────────────────────────
# FIRESTORE-OPERATIONEN
# ─────────────────────────────────────────────

def clear_collection(db: firestore.Client, collection: str) -> int:
    """
    Löscht ALLE Dokumente einer Collection vor dem Neu-Schreiben.
    Verhindert dass veraltete Dokumente aus früheren Runs auftauchen
    und historische Daten im Frontend verfälschen.
    Batched Deletes à 400 Ops.
    """
    col_ref = db.collection(collection)
    deleted = 0
    while True:
        docs = list(col_ref.limit(400).stream())
        if not docs:
            break
        batch = db.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()
        deleted += len(docs)
    print(f"  🗑️   {collection}: {deleted} alte Dokumente gelöscht")
    return deleted


def store_patterns(
    db: firestore.Client,
    patterns: list[dict],
    collection: str,
) -> int:
    """
    Schreibt alle Patterns in die angegebene Sammlung.
    set() mit merge=False → überschreibt bei gleichem doc_id.
    """
    col_ref = db.collection(collection)
    written = 0

    for i, pattern in enumerate(patterns):
        doc_id, doc = _pattern_to_doc(pattern, i)
        try:
            col_ref.document(doc_id).set(doc)
            pt    = pattern.get("pattern_type", "?")
            name  = pattern.get("sequence_name") or pattern.get("gegenpartei", "?")
            n_txs = len(pattern.get("transactions", []))
            print(f"    ✅  [{pt}]  {name[:45]:<45}  {n_txs:>3} Tx  →  {collection}/{doc_id}")
            written += 1
        except Exception as e:
            print(f"    ❌  Fehler bei Pattern #{i}: {e}")

    return written


def store_distributions(
    db: firestore.Client,
    transactions: list[dict],
    collection: str,
) -> int:
    """
    Schreibt jede Transaktion ohne Muster als eigenes Dokument.
    Nutzt Batched Writes (max. 400 Ops pro Batch) für Effizienz.
    """
    col_ref    = db.collection(collection)
    written    = 0
    BATCH_SIZE = 400

    for batch_start in range(0, len(transactions), BATCH_SIZE):
        batch = db.batch()
        chunk = transactions[batch_start: batch_start + BATCH_SIZE]

        for tx in chunk:
            doc_id, doc = _tx_to_standalone_doc(tx)
            batch.set(col_ref.document(doc_id), doc)

        try:
            batch.commit()
            written += len(chunk)
            end = min(batch_start + BATCH_SIZE, len(transactions))
            print(f"    ✅  Batch {batch_start + 1:>3}–{end:>3}  →  {collection}  ({len(chunk)} Dokumente)")
        except Exception as e:
            print(f"    ❌  Batch-Fehler ({batch_start}–{batch_start + BATCH_SIZE}): {e}")

    return written


# ─────────────────────────────────────────────
# HAUPT-PIPELINE
# ─────────────────────────────────────────────

def main():
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  ORGANISATIONAL  –  Firestore Export")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    print(f"  Input  : {INPUT_FILE}")

    # ── 1. Credentials prüfen ─────────────────────────────────────
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        print("\n❌  GOOGLE_APPLICATION_CREDENTIALS nicht gesetzt.")
        print("    Lokal  : export GOOGLE_APPLICATION_CREDENTIALS=/pfad/zu/serviceaccount.json")
        print("    GitHub : Secret FIREBASE_SERVICE_ACCOUNT wird automatisch gesetzt.")
        raise SystemExit(1)

    if not os.path.exists(creds_path):
        print(f"\n❌  Service Account Datei nicht gefunden: {creds_path}")
        raise SystemExit(1)

    # ── 2. Input laden ────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌  Input-Datei nicht gefunden: {INPUT_FILE}")
        print("    Zuerst detect_patterns.py ausführen (erzeugt tink_patterns.json).")
        raise SystemExit(1)

    data = load_patterns(INPUT_FILE)
    meta = data.get("meta", {})

    print(f"  Generiert am   : {meta.get('generated_at', '-')}")

    # Pattern-Listen aus JSON zusammenführen
    all_patterns = (
        data.get("recurring",  []) +
        data.get("seasonal",   []) +
        data.get("sequential", [])
    )
    no_pat = data.get("no_pattern", [])

    summary = data.get("summary", {})
    print(f"\n  RECURRING      : {summary.get('recurring',  0):>4} Muster")
    print(f"  SEASONAL       : {summary.get('seasonal',   0):>4} Muster")
    print(f"  SEQUENTIAL     : {summary.get('sequential', 0):>4} Muster")
    print(f"  Ohne Muster    : {summary.get('no_pattern', 0):>4} Transaktionen")

    # ── 3. Firestore-Client initialisieren ────────────────────────
    print(f"\n  Verbinde mit Firestore ...")
    try:
        db = firestore.Client()
        db.collection(COLLECTION_PATTERNS).limit(1).get()   # Verbindungstest
        print(f"  ✅  Verbindung OK\n")
    except Exception as e:
        print(f"\n❌  Firestore-Verbindung fehlgeschlagen: {e}")
        print("\n  Mögliche Ursachen:")
        print("    • Service Account hat keine Firestore-Berechtigung")
        print("    • Projekt-ID fehlt (im Service Account JSON enthalten?)")
        print("    • Firestore API im Google Cloud Projekt nicht aktiviert")
        raise SystemExit(1)

    # ── 4. Collections leeren → kein Datenmix aus alten Runs ─────
    print(f"{'─' * 70}")
    print("  Collections leeren ...")
    clear_collection(db, COLLECTION_PATTERNS)
    clear_collection(db, COLLECTION_DISTRIBUTIONS)

    # ── 5. patterns_db befüllen ───────────────────────────────────
    print(f"{'─' * 70}")
    print(f"  PATTERNS  →  {COLLECTION_PATTERNS}  ({len(all_patterns)} Dokumente)")
    print(f"{'─' * 70}")

    patterns_written = store_patterns(db, all_patterns, COLLECTION_PATTERNS)

    # ── 6. distributions_db befüllen ─────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  OHNE MUSTER  →  {COLLECTION_DISTRIBUTIONS}  ({len(no_pat)} Dokumente)")
    print(f"{'─' * 70}")

    dist_written = store_distributions(db, no_pat, COLLECTION_DISTRIBUTIONS)

    # ── 7. Abschlussbericht ───────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  ABSCHLUSS")
    print(SEP)
    print(f"  {COLLECTION_PATTERNS:<25} {patterns_written:>4} / {len(all_patterns):>4} Dokumente geschrieben")
    print(f"  {COLLECTION_DISTRIBUTIONS:<25} {dist_written:>4} / {len(no_pat):>4} Dokumente geschrieben")

    total_ok  = patterns_written + dist_written
    total_all = len(all_patterns) + len(no_pat)
    errors    = total_all - total_ok

    if errors == 0:
        print(f"\n  ✅  Alle {total_ok} Dokumente erfolgreich gespeichert.")
    else:
        print(f"\n  ⚠️   {errors} Fehler  |  {total_ok}/{total_all} erfolgreich")

    print(f"\n{'─' * 70}")
    print(f"  Firebase Console:")
    print(f"    https://console.firebase.google.com")
    print(f"{'─' * 70}\n")


if __name__ == "__main__":
    main()
