"""
organisational.py
=================
Organisiert Pattern-Ergebnisse aus detect_patterns.py und speichert sie in Firestore.

DESIGN-PRINZIPIEN:
  1. Anomalien werden NICHT gespeichert – ihre Transaktionen sind bereits
     in anderen Patterns (recurring, batch etc.) oder in distributions_db
  2. Jedes Pattern-Dokument hat ein next_expected_date (für Frontend/Forecast)
  3. Einmalige Batches gehen in distributions_db, wiederkehrende in pattern_db
  4. Alle Schreibvorgänge sind idempotent (gleicher Lauf = gleiche Dokument-IDs)

FIRESTORE-STRUKTUR:
  pattern_db/
    {pattern_id}/
      pattern_type        : "recurring" | "batch" | "seasonal" | "sequential" | "counter"
      gegenpartei         : Gegenparteiname
      category            : Kategorie
      iban                : IBAN (wenn vorhanden)
      next_expected_date  : Datum der nächsten erwarteten Transaktion  ← IMMER vorhanden
      transaction_count   : Anzahl Transaktionen
      first_seen          : Frühestes Datum
      last_seen           : Spätestes Datum
      amount_avg          : Durchschnitt aller Beträge
      amount_sum          : Summe aller Beträge
      ...                 : Pattern-spezifische Felder
      created_at / updated_at

      transactions/
        {txn_id}/         : Einzelne Transaktion

  distributions_db/
    {txn_id}/             : Transaktionen ohne Pattern (inkl. einmalige Batches)

PATTERN-PRIORITÄT (Anomalie absichtlich ausgelassen):
  1. recurring   → zuverlässigstes Pattern
  2. batch       → nur wenn wiederkehrend (sonst distributions_db)
  3. seasonal
  4. sequential
  5. counter

VERWENDUNG:
  Als Modul:
    from organisational import save_to_firestore
    save_to_firestore(result)

  Direkttest:
    python organisational.py
    python organisational.py meine_daten.json
    python organisational.py daten.json service_account.json
"""

import json
import re
import hashlib
import sys
import os
import statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore
from dateutil.relativedelta import relativedelta


# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SERVICE_ACCOUNT_FILE     = "bank-417a7-firebase-adminsdk-fbsvc-60ba2be615.json"
COLLECTION_PATTERNS      = "pattern_db"
COLLECTION_DISTRIBUTIONS = "distributions_db"
FIRESTORE_BATCH_LIMIT    = 400

# Anomalie ist bewusst nicht enthalten:
# Anomalie-Transaktionen sind bereits in anderen Patterns oder distributions_db
PATTERN_PRIORITY = ["recurring", "batch", "seasonal", "sequential", "counter"]


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
    if not os.path.exists(service_account):
        raise FileNotFoundError(
            f"Service-Account nicht gefunden: {service_account}\n"
            f"Bitte sicherstellen dass die Datei im gleichen Ordner liegt."
        )
    try:
        firebase_admin.get_app()
        _log(f"  ℹ️  Firebase bereits initialisiert – verwende bestehende App")
    except ValueError:
        cred = credentials.Certificate(service_account)
        firebase_admin.initialize_app(cred)
        _log(f"  ✅ Firebase initialisiert mit: {service_account}")
    return firestore.client()


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _get_iban(txn: dict) -> Optional[str]:
    for field in ["iban", "iban_gegenpartei", "counterparty_iban", "empfaenger_iban"]:
        val = txn.get(field)
        if val and str(val).strip():
            return str(val).strip()
    return None


def _norm_key(name: str, category: str, iban: Optional[str] = None) -> str:
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


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _primary_pattern_type(p: dict) -> Optional[str]:
    """
    Gibt den primären Pattern-Typ zurück.
    Anomalie wird bewusst NICHT zurückgegeben – Anomalie-Transaktionen
    landen entweder in einem anderen Pattern oder in distributions_db.
    """
    for ptype in PATTERN_PRIORITY:
        if p.get(f"is_{ptype}"):
            return ptype
    return None


def _pattern_group_key(txn: dict, ptype: str) -> str:
    """
    Gruppierschlüssel für Pattern-Dokumente.

    Batch-Besonderheit: Wir gruppieren NICHT nach batch_id (die enthält das Datum
    und würde jeden Monat ein neues Dokument erzeugen), sondern nach
    Kategorie + Zahlungsrichtung. So erkennen wir: "monatlicher Lohnlauf" ist
    ein wiederkehrender Batch, nicht 12 verschiedene Batches.
    """
    p   = txn.get("pattern", {})
    cat = txn.get("category_level1", "")

    if ptype == "batch":
        direction = "ausgabe" if txn.get("betrag", 0) < 0 else "einnahme"
        cat_slug  = re.sub(r"[^a-z]", "_", cat.lower().split("–")[-1].strip())[:20]
        return f"batch|{cat_slug}|{direction}"

    if ptype == "sequential":
        trigger_id = p.get("sequential_trigger_txn_id", "unknown")
        cat_a      = p.get("sequential_trigger_category", "")
        cat_b      = p.get("sequential_follows_category", "")
        return f"sequential|trigger_{trigger_id}|{cat_a}→{cat_b}"

    if ptype == "counter":
        cat_trigger = p.get("counter_trigger_category", "")
        cat_result  = p.get("counter_result_category", "")
        return f"counter|{cat_trigger}→{cat_result}"

    # recurring, seasonal → nach Gegenpartei+Kategorie+IBAN
    key = _norm_key(txn.get("gegenpartei",""), cat, _get_iban(txn))
    return f"{ptype}|{key}"


def _pattern_doc_id(group_key: str) -> str:
    slug  = re.sub(r"[^a-zA-Z0-9_\-]", "_", group_key)
    slug  = re.sub(r"_+", "_", slug).strip("_")[:60]
    hash8 = hashlib.md5(group_key.encode()).hexdigest()[:8]
    return f"{slug}_{hash8}"


def _txn_doc_id(txn: dict) -> str:
    datum      = txn.get("datum", "0000-00-00").replace("-", "")
    betrag     = int(abs(txn.get("betrag", 0)) * 100)
    vorzeichen = "p" if txn.get("betrag", 0) >= 0 else "n"
    gp         = txn.get("gegenpartei", "")
    gp_slug    = re.sub(r"[^a-z0-9]", "_", gp.lower())
    gp_slug    = re.sub(r"_+", "_", gp_slug).strip("_")[:25]
    raw        = f"{datum}|{vorzeichen}{betrag}|{txn.get('verwendungszweck','')[:50]}"
    hash6      = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"{datum}_{vorzeichen}{betrag}_{gp_slug}_{hash6}"


# ═══════════════════════════════════════════════════════════════════════════════
# NEXT_EXPECTED_DATE – FÜR ALLE PATTERN-TYPEN
# Das Frontend kann dieses Feld direkt für den Forecast verwenden.
# ═══════════════════════════════════════════════════════════════════════════════

def _at_next_workday(d: datetime) -> datetime:
    """Verschiebt auf nächsten österreichischen Arbeitstag (Sa/So → Mo)."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _next_expected_for_pattern(ptype: str, p: dict, group_txns: list) -> Optional[str]:
    """
    Berechnet next_expected_date für jeden Pattern-Typ.

    recurring:
      → bereits von detect_patterns berechnet, direkt übernehmen

    batch (recurring):
      → Abstände zwischen den Batch-Daten berechnen,
        letztes Datum + Durchschnittsabstand
      → Nur wenn mehrere Batch-Daten existieren (sonst einmalig → distributions_db)

    seasonal:
      → Nächste Jahreszyklus-Instanz der seasonal_months

    sequential:
      → Letztes Trigger-Datum + avg_delay_hours / 24

    counter:
      → Letztes Datum + counter_avg_delay_hours / 24
    """
    dates = sorted([_parse_date(t["datum"]) for t in group_txns])
    if not dates:
        return None

    last = dates[-1]

    # ── recurring: detect_patterns hat es bereits berechnet ──────────────────
    if ptype == "recurring":
        return p.get("next_expected_date")

    # ── batch: Abstände zwischen Batch-Daten berechnen ───────────────────────
    if ptype == "batch":
        if len(dates) < 2:
            return None   # Einmaliger Batch → wird in distributions_db landen
        gaps    = [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_gap = statistics.mean(gaps)
        nxt     = last + timedelta(days=round(avg_gap))
        return _at_next_workday(nxt).strftime("%Y-%m-%d")

    # ── seasonal: nächste Instanz der saisonalen Monate ──────────────────────
    if ptype == "seasonal":
        seasonal_months = p.get("seasonal_months", [])
        if not seasonal_months:
            return None
        today   = datetime.now()
        # Nächsten Monat aus seasonal_months finden der in der Zukunft liegt
        for offset in range(0, 24):   # Max 2 Jahre suchen
            check = today + relativedelta(months=offset)
            if check.month in seasonal_months and check > last:
                # Tag: Durchschnitt der bisherigen Tage
                dom_avg = round(statistics.mean([d.day for d in dates]))
                try:
                    nxt = check.replace(day=dom_avg)
                except ValueError:
                    nxt = check + relativedelta(day=31)
                if nxt > today:
                    return _at_next_workday(nxt).strftime("%Y-%m-%d")
        return None

    # ── sequential: letztes Trigger-Datum + avg_delay ────────────────────────
    if ptype == "sequential":
        avg_delay_h = p.get("sequential_avg_delay_hours")
        if avg_delay_h is None:
            return None
        nxt = last + timedelta(hours=avg_delay_h)
        return _at_next_workday(nxt).strftime("%Y-%m-%d")

    # ── counter: letztes Datum + avg_delay ───────────────────────────────────
    if ptype == "counter":
        avg_delay_h = p.get("counter_avg_delay_hours")
        if avg_delay_h is None:
            return None
        nxt = last + timedelta(hours=avg_delay_h)
        return _at_next_workday(nxt).strftime("%Y-%m-%d")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN-METADATEN
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pattern_meta(txn: dict, ptype: str, group_txns: list) -> dict:
    """
    Baut das vollständige Metadaten-Dict für ein Pattern-Dokument.
    next_expected_date ist immer enthalten (None wenn nicht berechenbar).
    """
    p       = txn.get("pattern", {})
    amounts = [t.get("betrag", 0) for t in group_txns]
    dates   = sorted([t.get("datum", "") for t in group_txns])
    ned     = _next_expected_for_pattern(ptype, p, group_txns)

    meta = {
        # ── Basis ────────────────────────────────────────────────────────────
        "pattern_type":      ptype,
        "pattern_key":       _norm_key(
            txn.get("gegenpartei",""),
            txn.get("category_level1",""),
            _get_iban(txn)
        ),
        "gegenpartei":       txn.get("gegenpartei",""),
        "category":          txn.get("category_level1",""),
        "iban":              _get_iban(txn),

        # ── next_expected_date: das Frontend holt dieses Feld für den Forecast
        "next_expected_date": ned,

        # ── Aggregierte Statistiken ───────────────────────────────────────────
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
        })

    elif ptype == "batch":
        meta.update({
            "batch_size":         p.get("batch_size"),
            "batch_total":        p.get("batch_total"),
            "batch_confidence":   p.get("batch_confidence"),
            "batch_anomaly_type": p.get("batch_anomaly_type"),
            # Für recurring Batches: Intervall schätzen
            "batch_occurrence_count": len(dates),
        })
        # Intervall schätzen wenn mehrere Daten vorhanden
        parsed_dates = sorted([_parse_date(d) for d in dates])
        if len(parsed_dates) >= 2:
            gaps    = [(parsed_dates[i+1] - parsed_dates[i]).days
                       for i in range(len(parsed_dates) - 1)]
            avg_gap = statistics.mean(gaps)
            if   17 <= avg_gap <= 45:  meta["batch_recurrence_interval"] = "monthly"
            elif 76 <= avg_gap <= 105: meta["batch_recurrence_interval"] = "quarterly"
            elif  8 <= avg_gap <= 16:  meta["batch_recurrence_interval"] = "biweekly"
            elif  1 <= avg_gap <= 8:   meta["batch_recurrence_interval"] = "weekly"
            else:                      meta["batch_recurrence_interval"] = "irregular"

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

    return meta


def _txn_to_dict(txn: dict, pattern_type: str, pattern_doc_id: str) -> dict:
    d = {k: v for k, v in txn.items() if k != "pattern"}
    d.setdefault("datum",            None)
    d.setdefault("betrag",           None)
    d.setdefault("gegenpartei",      None)
    d.setdefault("verwendungszweck", None)
    d.setdefault("category_level1",  None)
    iban = _get_iban(txn)
    if iban:
        d["iban_normalized"] = re.sub(r"\s+", "", iban.upper())
    d["pattern_type"]    = pattern_type
    d["pattern_doc_id"]  = pattern_doc_id
    d["saved_at"]        = datetime.now(timezone.utc).isoformat()
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPPIERUNG
# ═══════════════════════════════════════════════════════════════════════════════

def _group_transactions(transactions: list) -> tuple[dict, list]:
    """
    Gruppiert Transaktionen nach Pattern-Typ und Gruppierschlüssel.

    Einmalige Batches (nur 1 Datum-Gruppe vorhanden nach Gruppierung)
    werden direkt in no_pattern verschoben.

    Anomalien kommen nicht in pattern_groups – sie haben keinen Platz
    in PATTERN_PRIORITY und landen in no_pattern → distributions_db.
    """
    pattern_groups: dict = defaultdict(lambda: {"ptype": None, "txns": []})
    no_pattern: list     = []

    for txn in transactions:
        p     = txn.get("pattern", {})
        ptype = _primary_pattern_type(p)

        if ptype is None:
            # Kein erkanntes Pattern (inkl. reine Anomalien ohne anderes Pattern)
            no_pattern.append(txn)
            continue

        gkey = _pattern_group_key(txn, ptype)
        pattern_groups[gkey]["ptype"] = ptype
        pattern_groups[gkey]["txns"].append(txn)

    # ── Einmalige Batches → distributions_db ─────────────────────────────────
    # Ein Batch ist "einmalig" wenn alle seine Transaktionen das gleiche Datum haben
    # (d.h. er kommt nur einmal vor, nicht monatlich wiederkehrend)
    final_groups: dict = {}
    for gkey, group in pattern_groups.items():
        if group["ptype"] == "batch":
            unique_dates = set(t["datum"] for t in group["txns"])
            if len(unique_dates) <= 1:
                # Nur ein Datum → einmaliger Batch → distributions_db
                _log(f"  ℹ️  Einmaliger Batch → distributions_db: "
                     f"{group['txns'][0].get('gegenpartei','')[:30]} "
                     f"({group['txns'][0].get('datum','')})")
                no_pattern.extend(group["txns"])
                continue
        final_groups[gkey] = group

    return final_groups, no_pattern


# ═══════════════════════════════════════════════════════════════════════════════
# FIRESTORE SCHREIBEN
# ═══════════════════════════════════════════════════════════════════════════════

def _flush_batch(db: firestore.Client, ops: list):
    if not ops:
        return
    for chunk_start in range(0, len(ops), FIRESTORE_BATCH_LIMIT):
        chunk = ops[chunk_start: chunk_start + FIRESTORE_BATCH_LIMIT]
        batch = db.batch()
        for ref, data, merge in chunk:
            batch.set(ref, data, merge=True) if merge else batch.set(ref, data)
        batch.commit()


def _write_patterns(db: firestore.Client, pattern_groups: dict) -> dict:
    stats          = defaultdict(int)
    stats["total"] = 0
    now_ts         = datetime.now(timezone.utc).isoformat()

    for group_key, group in pattern_groups.items():
        ptype      = group["ptype"]
        group_txns = group["txns"]
        if not group_txns:
            continue

        doc_id  = _pattern_doc_id(group_key)
        doc_ref = db.collection(COLLECTION_PATTERNS).document(doc_id)
        meta    = _extract_pattern_meta(group_txns[0], ptype, group_txns)
        meta["updated_at"] = now_ts
        meta["created_at"] = now_ts   # merge=True: wird nicht überschrieben wenn schon da

        ops = [(doc_ref, meta, True)]   # merge=True für Pattern-Dokument

        for txn in group_txns:
            txn_id   = _txn_doc_id(txn)
            txn_ref  = doc_ref.collection("transactions").document(txn_id)
            txn_data = _txn_to_dict(txn, ptype, doc_id)
            ops.append((txn_ref, txn_data, False))

        _flush_batch(db, ops)

        ned_str = f" → {meta['next_expected_date']}" if meta.get("next_expected_date") else " → kein Datum"
        _log(f"  ✅ {doc_id[:55]}")
        _log(f"     {ptype:<12} | {group_txns[0].get('gegenpartei','')[:30]} "
             f"| {len(group_txns)} Txns{ned_str}")

        stats[ptype]   += 1
        stats["total"] += 1

    return dict(stats)


def _write_distributions(db: firestore.Client, no_pattern: list) -> int:
    if not no_pattern:
        return 0
    ops = []
    for txn in no_pattern:
        txn_id   = _txn_doc_id(txn)
        txn_ref  = db.collection(COLLECTION_DISTRIBUTIONS).document(txn_id)
        txn_data = _txn_to_dict(txn, None, None)
        txn_data["pattern_type"] = None
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
    Organisiert und speichert Pattern-Ergebnisse in Firestore.

    Args:
        transactions:    Output von detect_patterns() – Liste mit 'pattern'-Feld
        service_account: Pfad zur Firebase Service-Account JSON

    Rückgabe:
        dict: Statistiken (pattern_groups, distributions, total_patterns, total_txns)
    """
    _log_section("ORGANISATIONAL – Firestore Speicherung")
    _log(f"  {len(transactions)} Transaktionen | "
         f"Ziel: {COLLECTION_PATTERNS} + {COLLECTION_DISTRIBUTIONS}")
    _log(f"  Anomalien werden nicht gespeichert (bereits in anderen Patterns/distributions_db)")

    # ── 1. Firestore initialisieren ───────────────────────────────────────────
    _log_subsection("1/3 | FIRESTORE INITIALISIERUNG")
    db = _init_firestore(service_account)

    # ── 2. Gruppieren ─────────────────────────────────────────────────────────
    _log_subsection("2/3 | GRUPPIERUNG")
    pattern_groups, no_pattern = _group_transactions(transactions)

    type_counts = defaultdict(int)
    for g in pattern_groups.values():
        type_counts[g["ptype"]] += 1

    _log(f"\n  Pattern-Gruppen: {len(pattern_groups)}")
    for ptype in PATTERN_PRIORITY:
        if type_counts[ptype]:
            _log(f"    {ptype:<12} : {type_counts[ptype]:>3} Gruppen")
    _log(f"  Ohne Pattern   : {len(no_pattern)} → distributions_db "
         f"(inkl. Anomalien + einmalige Batches)")

    # ── 3. Speichern ──────────────────────────────────────────────────────────
    _log_subsection(f"3/3 | SPEICHERN → {COLLECTION_PATTERNS}")
    pattern_stats = _write_patterns(db, pattern_groups)

    _log_subsection(f"     SPEICHERN → {COLLECTION_DISTRIBUTIONS}")
    dist_count = _write_distributions(db, no_pattern)
    _log(f"  ✅ {dist_count} Dokumente in {COLLECTION_DISTRIBUTIONS}")

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    total_pattern_docs = pattern_stats.get("total", 0)
    total_pattern_txns = sum(len(g["txns"]) for g in pattern_groups.values())

    result = {
        "pattern_groups": dict(type_counts),
        "distributions":  dist_count,
        "total_patterns": total_pattern_docs,
        "total_txns":     total_pattern_txns,
    }

    _log_section("ERGEBNIS")
    _log(f"  {'Sammlung':<25}  {'Dokumente':>10}  {'Transaktionen':>15}")
    _log(f"  {'─'*55}")
    _log(f"  {COLLECTION_PATTERNS:<25}  {total_pattern_docs:>10}  {total_pattern_txns:>15}")
    for ptype in PATTERN_PRIORITY:
        if type_counts[ptype]:
            txns_in = sum(len(g["txns"]) for g in pattern_groups.values()
                          if g["ptype"] == ptype)
            _log(f"    └─ {ptype:<21}  {type_counts[ptype]:>10}  {txns_in:>15}")
    _log(f"  {COLLECTION_DISTRIBUTIONS:<25}  {dist_count:>10}  {'(1 pro Dokument)':>15}")
    _log(f"  {'─'*55}")
    _log(f"  {'Total:':<25}  {total_pattern_docs + dist_count:>10}  "
         f"{len(transactions):>15}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DIREKTTEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DEFAULT_INPUT   = "tink_categorized.json"
    INPUT_FILE      = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    SERVICE_ACCOUNT = sys.argv[2] if len(sys.argv) > 2 else SERVICE_ACCOUNT_FILE

    _log("=" * 70)
    _log("  organisational.py – Direkttest")
    _log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"  Eingabe         : {INPUT_FILE}")
    _log(f"  Service-Account : {SERVICE_ACCOUNT}")
    _log("=" * 70)

    _log_section("DATEN LADEN + PATTERN DETECTION")

    if not os.path.exists(INPUT_FILE):
        _log(f"  ❌ Datei nicht gefunden: {INPUT_FILE}")
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    _log(f"  ✅ {len(raw)} Transaktionen geladen")

    has_patterns = any("pattern" in t for t in raw)
    if has_patterns:
        _log(f"  ℹ️  Pattern-Felder vorhanden → Detection überspringen")
        transactions = raw
    else:
        _log(f"  ℹ️  Keine Pattern-Felder → führe detect_patterns aus...")
        try:
            from detect_patterns import detect_patterns
            transactions = detect_patterns(raw)
        except ImportError:
            _log(f"  ❌ detect_patterns.py nicht gefunden")
            sys.exit(1)

    result = save_to_firestore(transactions, service_account=SERVICE_ACCOUNT)

    _log("\n" + "=" * 70)
    _log("  ✅ organisational.py abgeschlossen")
    _log(f"  Pattern-Dokumente : {result['total_patterns']}")
    _log(f"  Distributions     : {result['distributions']}")
    _log(f"  Abgeschlossen     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("=" * 70)
