"""
organisational.py
=================
Organisiert Pattern-Ergebnisse aus detect_patterns.py und speichert sie in Firestore.

FIRESTORE-STRUKTUR:
  pattern_db/                        ← Sammlung: Patterns
    {pattern_id}/                    ← Dokument: ein einzigartiges Pattern
      pattern_type      : "recurring" | "batch" | "seasonal" | "sequential" | "counter" | "anomaly"
      pattern_key       : Gruppierschlüssel (Name|Kategorie oder IBAN|Kategorie)
      gegenpartei       : Gegenparteiname
      category          : Kategorie
      iban              : IBAN (wenn vorhanden, sonst null)
      transaction_count : Anzahl Transaktionen in diesem Pattern
      first_seen        : Frühestes Datum
      last_seen         : Spätestes Datum
      amount_sum        : Summe aller Beträge
      amount_avg        : Durchschnitt aller Beträge
      ...               : Alle relevanten Pattern-Felder (interval, confidence, etc.)
      created_at        : Firestore-Timestamp (Erstanlage)
      updated_at        : Firestore-Timestamp (letzte Aktualisierung)

      transactions/                  ← Untersammlung
        {txn_id}/                    ← Dokument: einzelne Transaktion
          datum, betrag, gegenpartei, verwendungszweck, category_level1, iban, ...

  distributions_db/                  ← Sammlung: Transaktionen ohne Pattern
    {txn_id}/                        ← Dokument: einzelne Transaktion
          datum, betrag, gegenpartei, verwendungszweck, category_level1, iban, ...

PATTERN-PRIORITÄT (eine Transaktion → genau ein Pattern-Dokument):
  1. recurring   → zuverlässigstes Pattern
  2. batch       → Tagesgruppe
  3. seasonal    → Jahresrhythmus
  4. sequential  → kausale Kette
  5. counter     → gegenläufig
  6. anomaly     → Ausreisser (is_anomaly=True)
  7. (keines)    → distributions_db

IDEMPOTENZ:
  Alle Schreibvorgänge sind idempotent:
  - Pattern-Dokumente: merge=True → bestehende Felder bleiben, neue werden ergänzt
  - Transaktions-Dokumente: feste ID aus Datum+Betrag+Gegenpartei → kein Duplikat

VERWENDUNG:
  Als Modul (von pipeline.py):
    from organisational import save_to_firestore
    save_to_firestore(result, service_account="bank-417a7-firebase-adminsdk-fbsvc-60ba2be615.json")

  Direkt (Standalone-Test):
    python organisational.py
    python organisational.py meine_kategorisierten_daten.json
"""

import json
import re
import hashlib
import sys
import os
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore


# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SERVICE_ACCOUNT_FILE = "bank-417a7-firebase-adminsdk-fbsvc-60ba2be615.json"

COLLECTION_PATTERNS      = "pattern_db"
COLLECTION_DISTRIBUTIONS = "distributions_db"

# Firestore Batch-Limit: max 500 Operationen pro Batch
FIRESTORE_BATCH_LIMIT = 400   # Sicherheitspuffer: 400 statt 500

# Pattern-Priorität (Index = Priorität, 0 = höchste)
PATTERN_PRIORITY = ["recurring", "batch", "seasonal", "sequential", "counter", "anomaly"]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    print(msg)

def _log_section(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")

def _log_subsection(title: str):
    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIRESTORE INITIALISIERUNG
# ═══════════════════════════════════════════════════════════════════════════════

def _init_firestore(service_account: str) -> firestore.Client:
    """
    Initialisiert Firebase Admin SDK und gibt Firestore-Client zurück.
    Idempotent: Prüft ob App bereits initialisiert ist.

    Args:
        service_account: Pfad zur Service-Account JSON-Datei

    Returns:
        Firestore Client

    Raises:
        FileNotFoundError: Service-Account-Datei nicht gefunden
        Exception: Firebase-Initialisierungsfehler
    """
    if not os.path.exists(service_account):
        raise FileNotFoundError(
            f"Service-Account nicht gefunden: {service_account}\n"
            f"Bitte sicherstellen dass die Datei im gleichen Ordner liegt."
        )

    try:
        # Prüfen ob App bereits initialisiert (z.B. von pipeline.py)
        app = firebase_admin.get_app()
        _log(f"  ℹ️  Firebase bereits initialisiert – verwende bestehende App")
    except ValueError:
        # Noch nicht initialisiert → neu anlegen
        cred = credentials.Certificate(service_account)
        app  = firebase_admin.initialize_app(cred)
        _log(f"  ✅ Firebase initialisiert mit: {service_account}")

    return firestore.client()


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _get_iban(txn: dict) -> Optional[str]:
    """Extrahiert IBAN aus Transaktion – prüft mehrere mögliche Feldnamen."""
    for field in ["iban", "iban_gegenpartei", "counterparty_iban", "empfaenger_iban"]:
        val = txn.get(field)
        if val and str(val).strip():
            return str(val).strip()
    return None


def _norm_key(name: str, category: str, iban: Optional[str] = None) -> str:
    """
    Erstellt Gruppierschlüssel für Transaktionen.
    IBAN hat Vorrang vor Name (zuverlässiger, Namensvarianten werden zusammengeführt).
    """
    if iban and iban.strip():
        iban_clean = re.sub(r"\s+", "", iban.strip().upper())
        if len(iban_clean) >= 15:
            return iban_clean + "|" + (category or "UNBEKANNT")

    generic = ["", "bank", "system", "intern", "unbekannt", "unbekannte gegenpartei",
               "eigene buchung", "intern transfer"]
    if not name or name.strip().lower() in generic:
        return "SYSTEM|" + (category or "UNBEKANNT")

    n = name.lower()
    for src, dst in [("ä","ae"),("ö","oe"),("ü","ue"),("ß","ss")]:
        n = n.replace(src, dst)
    n = re.sub(r"[^a-z0-9\s]", "", n)
    return "_".join(n.split()) + "|" + (category or "UNBEKANNT")


def _primary_pattern_type(p: dict) -> Optional[str]:
    """
    Bestimmt den primären Pattern-Typ einer Transaktion nach Priorität.
    Rückgabe: Typ-String oder None wenn kein Pattern.
    """
    for ptype in PATTERN_PRIORITY:
        flag = f"is_{ptype}"
        if p.get(flag):
            return ptype
    return None


def _pattern_group_key(txn: dict, ptype: str) -> str:
    """
    Erstellt den Gruppierschlüssel für ein Pattern-Dokument.
    Gleiche Gegenpartei + gleiche Kategorie + gleicher Typ → gleiches Dokument.

    Für batch: batch_id (bereits eindeutig)
    Für sequential: trigger_txn_id (Kette beginnt beim Trigger)
    Für alle anderen: norm_key + "_" + ptype
    """
    p = txn.get("pattern", {})

    if ptype == "batch":
        return f"batch|{p.get('batch_id', 'unknown')}"

    if ptype == "sequential":
        # Alle Folge-Transaktionen der gleichen Kette gruppieren
        trigger_id = p.get("sequential_trigger_txn_id", "unknown")
        cat_a      = p.get("sequential_trigger_category", "")
        cat_b      = p.get("sequential_follows_category", "")
        return f"sequential|trigger_{trigger_id}|{cat_a}→{cat_b}"

    if ptype == "counter":
        cat_trigger = p.get("counter_trigger_category", "")
        cat_result  = p.get("counter_result_category", "")
        return f"counter|{cat_trigger}→{cat_result}"

    if ptype == "anomaly":
        atype = p.get("anomaly_type", "unknown")
        key   = _norm_key(txn.get("gegenpartei",""), txn.get("category_level1",""), _get_iban(txn))
        return f"anomaly|{atype}|{key}"

    # recurring, seasonal → nach Gegenpartei+Kategorie+IBAN gruppieren
    key = _norm_key(txn.get("gegenpartei",""), txn.get("category_level1",""), _get_iban(txn))
    return f"{ptype}|{key}"


def _pattern_doc_id(group_key: str) -> str:
    """
    Generiert eine stabile, lesbare Firestore-Dokument-ID aus dem Gruppierschlüssel.

    Format: {slug}_{hash8}
    - slug: menschenlesbar, max 60 Zeichen
    - hash8: MD5-Prefix für Eindeutigkeit bei Kollisionen
    """
    # Slug: nur alphanumerisch + Unterstrich + Bindestrich
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", group_key)
    slug = re.sub(r"_+", "_", slug).strip("_")[:60]
    hash8 = hashlib.md5(group_key.encode()).hexdigest()[:8]
    return f"{slug}_{hash8}"


def _txn_doc_id(txn: dict) -> str:
    """
    Generiert eine stabile, eindeutige Dokument-ID für eine Transaktion.

    Format: {datum}_{betrag_cent}_{gegenpartei_slug}
    Idempotent: gleiche Transaktion → immer gleiche ID → keine Duplikate.
    """
    datum    = txn.get("datum", "0000-00-00").replace("-", "")
    betrag   = int(abs(txn.get("betrag", 0)) * 100)   # In Cent, kein Dezimalpunkt
    vorzeichen = "p" if txn.get("betrag", 0) >= 0 else "n"

    gp = txn.get("gegenpartei", "")
    gp_slug = re.sub(r"[^a-z0-9]", "_", gp.lower())
    gp_slug = re.sub(r"_+", "_", gp_slug).strip("_")[:25]

    raw = f"{datum}|{vorzeichen}{betrag}|{txn.get('verwendungszweck','')[:50]}"
    hash6 = hashlib.md5(raw.encode()).hexdigest()[:6]

    return f"{datum}_{vorzeichen}{betrag}_{gp_slug}_{hash6}"


def _extract_pattern_meta(txn: dict, ptype: str, group_txns: list) -> dict:
    """
    Extrahiert alle relevanten Pattern-Felder für das Firestore-Dokument.

    Berechnet zusätzlich aggregierte Felder aus allen Transaktionen der Gruppe:
    - transaction_count, first_seen, last_seen, amount_sum, amount_avg
    """
    p       = txn.get("pattern", {})
    amounts = [t.get("betrag", 0) for t in group_txns]
    dates   = sorted([t.get("datum", "") for t in group_txns])

    meta = {
        # ── Identifikation ───────────────────────────────────────────────────
        "pattern_type":      ptype,
        "pattern_key":       _norm_key(
            txn.get("gegenpartei",""),
            txn.get("category_level1",""),
            _get_iban(txn)
        ),
        "gegenpartei":       txn.get("gegenpartei",""),
        "category":          txn.get("category_level1",""),
        "iban":              _get_iban(txn),

        # ── Aggregierte Statistiken (aus allen Transaktionen der Gruppe) ─────
        "transaction_count": len(group_txns),
        "first_seen":        dates[0]  if dates else None,
        "last_seen":         dates[-1] if dates else None,
        "amount_sum":        round(sum(amounts), 2),
        "amount_avg":        round(sum(amounts) / len(amounts), 2) if amounts else 0.0,
        "amount_min":        round(min(amounts), 2) if amounts else 0.0,
        "amount_max":        round(max(amounts), 2) if amounts else 0.0,
    }

    # ── Pattern-spezifische Felder ────────────────────────────────────────────
    if ptype == "recurring":
        meta.update({
            "recurrence_interval":         p.get("recurrence_interval"),
            "recurrence_day_of_month":     p.get("recurrence_day_of_month"),
            "recurrence_day_of_week":      p.get("recurrence_day_of_week"),
            "recurrence_amount_avg":       p.get("recurrence_amount_avg"),
            "recurrence_amount_tolerance": p.get("recurrence_amount_tolerance"),
            "recurrence_confidence":       p.get("recurrence_confidence"),
            "recurrence_sample_size":      p.get("recurrence_sample_size"),
            "recurrence_has_gaps":         p.get("recurrence_has_gaps"),
            "next_expected_date":          p.get("next_expected_date"),
        })

    elif ptype == "batch":
        meta.update({
            "batch_id":           p.get("batch_id"),
            "batch_size":         p.get("batch_size"),
            "batch_total":        p.get("batch_total"),
            "batch_confidence":   p.get("batch_confidence"),
            "batch_anomaly_type": p.get("batch_anomaly_type"),
        })

    elif ptype == "seasonal":
        meta.update({
            "seasonal_months":         p.get("seasonal_months"),
            "seasonal_week_of_year":   p.get("seasonal_week_of_year"),
            "seasonal_amount_avg":     p.get("seasonal_amount_avg"),
            "seasonal_confidence":     p.get("seasonal_confidence"),
            "seasonal_years_observed": p.get("seasonal_years_observed"),
        })

    elif ptype == "sequential":
        meta.update({
            "sequential_trigger_txn_id":        p.get("sequential_trigger_txn_id"),
            "sequential_trigger_category":      p.get("sequential_trigger_category"),
            "sequential_follows_category":      p.get("sequential_follows_category"),
            "sequential_avg_delay_hours":       p.get("sequential_avg_delay_hours"),
            "sequential_delay_tolerance_hours": p.get("sequential_delay_tolerance_hours"),
            "sequential_amount_ratio":          p.get("sequential_amount_ratio"),
            "sequential_confidence":            p.get("sequential_confidence"),
            "sequential_observations":          p.get("sequential_observations"),
        })

    elif ptype == "counter":
        meta.update({
            "counter_trigger_category":      p.get("counter_trigger_category"),
            "counter_result_category":       p.get("counter_result_category"),
            "counter_avg_delay_hours":       p.get("counter_avg_delay_hours"),
            "counter_delay_tolerance_hours": p.get("counter_delay_tolerance_hours"),
            "counter_amount_ratio":          p.get("counter_amount_ratio"),
            "counter_confidence":            p.get("counter_confidence"),
            "counter_observations":          p.get("counter_observations"),
        })

    elif ptype == "anomaly":
        meta.update({
            "anomaly_type":          p.get("anomaly_type"),
            "anomaly_severity":      p.get("anomaly_severity"),
            "anomaly_reference_avg": p.get("anomaly_reference_avg"),
            "anomaly_score":         p.get("anomaly_score"),
            "anomaly_deviation":     p.get("anomaly_deviation"),
        })

    return meta


def _txn_to_dict(txn: dict) -> dict:
    """
    Konvertiert eine Transaktion in ein sauberes Firestore-Dict.
    Entfernt das 'pattern'-Feld (das liegt im Parent-Dokument),
    fügt Metadaten hinzu.
    """
    d = {k: v for k, v in txn.items() if k != "pattern"}

    # Sicherstellen dass Pflichtfelder vorhanden sind
    d.setdefault("datum",            None)
    d.setdefault("betrag",           None)
    d.setdefault("gegenpartei",      None)
    d.setdefault("verwendungszweck", None)
    d.setdefault("category_level1",  None)

    # IBAN normalisieren
    iban = _get_iban(txn)
    if iban:
        d["iban_normalized"] = re.sub(r"\s+", "", iban.upper())

    d["saved_at"] = datetime.now(timezone.utc).isoformat()
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPPIERUNG
# ═══════════════════════════════════════════════════════════════════════════════

def _group_transactions(transactions: list) -> tuple[dict, list]:
    """
    Gruppiert Transaktionen nach Pattern-Typ und Gruppierschlüssel.

    Rückgabe:
        pattern_groups : dict  group_key → {"ptype": str, "txns": list}
        no_pattern     : list  Transaktionen ohne Pattern
    """
    pattern_groups = defaultdict(lambda: {"ptype": None, "txns": []})
    no_pattern     = []

    for txn in transactions:
        p     = txn.get("pattern", {})
        ptype = _primary_pattern_type(p)

        if ptype is None:
            no_pattern.append(txn)
            continue

        gkey = _pattern_group_key(txn, ptype)
        pattern_groups[gkey]["ptype"] = ptype
        pattern_groups[gkey]["txns"].append(txn)

    return dict(pattern_groups), no_pattern


# ═══════════════════════════════════════════════════════════════════════════════
# FIRESTORE-SCHREIBFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _flush_batch(db: firestore.Client, ops: list):
    """
    Führt einen Firestore Batch-Write aus.
    ops: Liste von (ref, data, merge) Tupeln.
    Teilt automatisch in Chunks von FIRESTORE_BATCH_LIMIT auf.
    """
    if not ops:
        return

    for chunk_start in range(0, len(ops), FIRESTORE_BATCH_LIMIT):
        chunk = ops[chunk_start : chunk_start + FIRESTORE_BATCH_LIMIT]
        batch = db.batch()
        for ref, data, merge in chunk:
            if merge:
                batch.set(ref, data, merge=True)
            else:
                batch.set(ref, data)
        batch.commit()


def _write_patterns(db: firestore.Client, pattern_groups: dict) -> dict:
    """
    Schreibt alle Pattern-Gruppen in Firestore.

    Pro Gruppe:
      - 1 Dokument in pattern_db (merge=True → idempotent)
      - N Dokumente in pattern_db/{id}/transactions

    Rückgabe: Statistik-Dict
    """
    stats = {ptype: 0 for ptype in PATTERN_PRIORITY}
    stats["total_docs"]    = 0
    stats["total_txns"]    = 0
    stats["skipped_empty"] = 0

    now_ts = datetime.now(timezone.utc).isoformat()

    for group_key, group in pattern_groups.items():
        ptype     = group["ptype"]
        group_txns = group["txns"]

        if not group_txns:
            stats["skipped_empty"] += 1
            continue

        # ── Pattern-Dokument ─────────────────────────────────────────────────
        doc_id   = _pattern_doc_id(group_key)
        doc_ref  = db.collection(COLLECTION_PATTERNS).document(doc_id)
        meta     = _extract_pattern_meta(group_txns[0], ptype, group_txns)

        meta["updated_at"] = now_ts

        # created_at: nur setzen wenn Dokument noch nicht existiert
        # merge=True überschreibt es nicht wenn es bereits gesetzt ist
        meta_with_created = {**meta, "created_at": now_ts}

        # Batch für Pattern-Dokument + Transaktionen
        ops = []

        # Pattern-Dokument: merge=True damit created_at beim Update erhalten bleibt
        # Firestore merge überschreibt nur Felder die explizit angegeben sind.
        # Daher: created_at beim ersten Schreiben setzen, danach nicht mehr überschreiben.
        # Lösung: Server-seitiger Merge + separater created_at-Check zu teuer →
        # Pragmatische Lösung: created_at immer mitschreiben, merge=True ignoriert Konflikte nicht.
        # Für echte Erstanlage-Semantik: Transaction nutzen (hier: Performance-Kompromiss)
        ops.append((doc_ref, meta_with_created, True))

        # ── Transaktions-Dokumente (Untersammlung) ────────────────────────────
        for txn in group_txns:
            txn_id  = _txn_doc_id(txn)
            txn_ref = doc_ref.collection("transactions").document(txn_id)
            txn_data = _txn_to_dict(txn)
            # Pattern-Zusammenfassung direkt in der Transaktion speichern
            txn_data["pattern_type"]   = ptype
            txn_data["pattern_doc_id"] = doc_id
            ops.append((txn_ref, txn_data, False))

        _flush_batch(db, ops)

        stats[ptype]       += 1
        stats["total_docs"] += 1
        stats["total_txns"] += len(group_txns)

        _log(f"  ✅ pattern_db/{doc_id[:50]}")
        _log(f"     Typ: {ptype:<12} | "
             f"Gegenpartei: {group_txns[0].get('gegenpartei','')[:30]} | "
             f"{len(group_txns)} Transaktionen")

    return stats


def _write_distributions(db: firestore.Client, no_pattern: list) -> int:
    """
    Schreibt Transaktionen ohne Pattern in distributions_db.
    Jede Transaktion bekommt ein eigenes Dokument.

    Rückgabe: Anzahl geschriebener Dokumente
    """
    if not no_pattern:
        return 0

    ops = []
    for txn in no_pattern:
        txn_id  = _txn_doc_id(txn)
        txn_ref = db.collection(COLLECTION_DISTRIBUTIONS).document(txn_id)
        txn_data = _txn_to_dict(txn)
        txn_data["pattern_type"] = None   # Explizit kein Pattern
        ops.append((txn_ref, txn_data, False))

    _flush_batch(db, ops)
    return len(no_pattern)


# ═══════════════════════════════════════════════════════════════════════════════
# HAUPTFUNKTION
# ═══════════════════════════════════════════════════════════════════════════════

def save_to_firestore(
    transactions: list,
    service_account: str = SERVICE_ACCOUNT_FILE,
) -> dict:
    """
    Organisiert Pattern-Ergebnisse und speichert sie in Firestore.

    Args:
        transactions:    Output von detect_patterns() – Liste mit 'pattern'-Feld
        service_account: Pfad zur Firebase Service-Account JSON

    Rückgabe:
        dict mit Statistiken:
          pattern_groups  : Anzahl Pattern-Gruppen pro Typ
          distributions   : Anzahl Transaktionen in distributions_db
          total_patterns  : Gesamtzahl Pattern-Dokumente
          total_txns      : Gesamtzahl in pattern_db gespeicherte Transaktionen
    """
    _log_section("ORGANISATIONAL – Firestore Speicherung")
    _log(f"  {len(transactions)} Transaktionen | "
         f"Ziel: {COLLECTION_PATTERNS} + {COLLECTION_DISTRIBUTIONS}")

    # ── 1. Firestore initialisieren ───────────────────────────────────────────
    _log_subsection("1/3 | FIRESTORE INITIALISIERUNG")
    db = _init_firestore(service_account)

    # ── 2. Gruppieren ─────────────────────────────────────────────────────────
    _log_subsection("2/3 | GRUPPIERUNG")
    pattern_groups, no_pattern = _group_transactions(transactions)

    # Übersicht ausgeben
    type_counts = defaultdict(int)
    for g in pattern_groups.values():
        type_counts[g["ptype"]] += 1

    _log(f"  Pattern-Gruppen: {len(pattern_groups)}")
    for ptype in PATTERN_PRIORITY:
        if type_counts[ptype]:
            _log(f"    {ptype:<12} : {type_counts[ptype]:>3} Gruppen")
    _log(f"  Ohne Pattern   : {len(no_pattern)} Transaktionen → distributions_db")

    # ── 3. Speichern ──────────────────────────────────────────────────────────
    _log_subsection(f"3/3 | SPEICHERN → {COLLECTION_PATTERNS}")
    pattern_stats = _write_patterns(db, pattern_groups)

    _log_subsection(f"     SPEICHERN → {COLLECTION_DISTRIBUTIONS}")
    dist_count = _write_distributions(db, no_pattern)
    if dist_count:
        _log(f"  ✅ {dist_count} Transaktionen in {COLLECTION_DISTRIBUTIONS} gespeichert")
    else:
        _log(f"  ℹ️  Keine Transaktionen ohne Pattern")

    # ── Abschluss-Statistik ───────────────────────────────────────────────────
    result = {
        "pattern_groups": dict(type_counts),
        "distributions":  dist_count,
        "total_patterns": pattern_stats["total_docs"],
        "total_txns":     pattern_stats["total_txns"],
    }

    _log_section("ERGEBNIS")
    _log(f"  {'Sammlung':<25}  {'Dokumente':>10}  {'Transaktionen':>15}")
    _log(f"  {'─'*55}")
    _log(f"  {COLLECTION_PATTERNS:<25}  {pattern_stats['total_docs']:>10}  "
         f"{pattern_stats['total_txns']:>15}")
    for ptype in PATTERN_PRIORITY:
        if type_counts[ptype]:
            txns_in_type = sum(
                len(g["txns"]) for g in pattern_groups.values()
                if g["ptype"] == ptype
            )
            _log(f"    └─ {ptype:<21}  {type_counts[ptype]:>10}  {txns_in_type:>15}")
    _log(f"  {COLLECTION_DISTRIBUTIONS:<25}  {dist_count:>10}  {'(1 pro Dokument)':>15}")
    _log(f"  {'─'*55}")
    _log(f"  {'Total:':<25}  {pattern_stats['total_docs'] + dist_count:>10}  "
         f"{len(transactions):>15}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DIREKTTEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Eingabedatei bestimmen ────────────────────────────────────────────────
    # Varianten:
    #   python organisational.py                           → tink_categorized.json
    #   python organisational.py meine_daten.json          → beliebige kategorisierte JSON
    #   python organisational.py daten.json service.json   → mit eigenem Service-Account
    DEFAULT_INPUT   = "tink_categorized.json"
    INPUT_FILE      = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    SERVICE_ACCOUNT = sys.argv[2] if len(sys.argv) > 2 else SERVICE_ACCOUNT_FILE

    _log("=" * 70)
    _log("  organisational.py – Direkttest")
    _log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"  Eingabe         : {INPUT_FILE}")
    _log(f"  Service-Account : {SERVICE_ACCOUNT}")
    _log(f"  Ziel-Sammlungen : {COLLECTION_PATTERNS} | {COLLECTION_DISTRIBUTIONS}")
    _log("=" * 70)

    # ── Daten laden ───────────────────────────────────────────────────────────
    _log_section("DATEN LADEN + PATTERN DETECTION")

    if not os.path.exists(INPUT_FILE):
        _log(f"  ❌ Datei nicht gefunden: {INPUT_FILE}")
        _log(f"     Tipp: Zuerst pipeline.py oder detect_patterns.py ausführen.")
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        raw_transactions = json.load(f)

    _log(f"  ✅ {len(raw_transactions)} Transaktionen geladen")

    # Prüfen ob Transaktionen bereits Pattern haben (Output von detect_patterns)
    # oder ob detect_patterns noch ausgeführt werden muss
    has_patterns = any("pattern" in t for t in raw_transactions)

    if has_patterns:
        _log(f"  ℹ️  Pattern-Felder bereits vorhanden → Pattern Detection überspringen")
        transactions = raw_transactions
    else:
        _log(f"  ℹ️  Keine Pattern-Felder gefunden → führe detect_patterns aus...")
        try:
            from detect_patterns import detect_patterns
            transactions = detect_patterns(raw_transactions)
            _log(f"  ✅ Pattern Detection abgeschlossen")
        except ImportError:
            _log(f"  ❌ detect_patterns.py nicht gefunden")
            _log(f"     Bitte sicherstellen dass detect_patterns.py im gleichen Ordner liegt.")
            sys.exit(1)

    # Kurze Statistik
    with_pattern = sum(1 for t in transactions if any([
        t.get("pattern",{}).get("is_recurring"),
        t.get("pattern",{}).get("is_batch"),
        t.get("pattern",{}).get("is_seasonal"),
        t.get("pattern",{}).get("is_sequential"),
        t.get("pattern",{}).get("is_counter"),
        t.get("pattern",{}).get("is_anomaly"),
    ]))
    _log(f"\n  Transaktionen mit Pattern : {with_pattern}")
    _log(f"  Transaktionen ohne Pattern: {len(transactions) - with_pattern}")

    # ── In Firestore speichern ────────────────────────────────────────────────
    result = save_to_firestore(transactions, service_account=SERVICE_ACCOUNT)

    # ── Abschluss ─────────────────────────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("  ✅ organisational.py – Direkttest abgeschlossen")
    _log(f"  Pattern-Dokumente : {result['total_patterns']}")
    _log(f"  Distributions     : {result['distributions']}")
    _log(f"  Abgeschlossen     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("=" * 70)
