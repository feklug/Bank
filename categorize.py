"""
categorize.py
=============
Kategorisiert Banktransaktionen via Claude API.

Verwendung (Pipeline / in-memory):
    from categorize import categorize
    categorized = categorize(transactions, api_key="sk-ant-...")

Verwendung (Standalone / CLI):
    python categorize.py
    → liest tink_demo_transactions.json, schreibt tink_categorized.json

Input:  Tink /data/v2/transactions Format (verschachtelt, mit status/dates/amount/...)
Output: Flaches Format pro Transaktion:
    {
        "datum":            "2024-01-05",
        "bookedDateTime":   "2024-01-05T08:23:41Z",
        "betrag":           -78.00,
        "verwendungszweck": "...",
        "gegenpartei":      "...",
        "iban":             "AT12 ...",   # oder null
        "category_level1":  "AUSGABEN – BETRIEBSKOSTEN",
        "confidence":       0.95,
        "reasoning":        "..."
    }
"""

import json
import os
import time
from collections import Counter
from datetime import datetime

import anthropic

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

INPUT_FILE  = "tink_demo_transactions.json"
OUTPUT_FILE = "tink_categorized.json"
BATCH_SIZE  = 20   # Transaktionen pro API-Call
MAX_RETRIES = 3    # Anzahl Wiederholungen bei API-Fehler
RETRY_DELAY = 5    # Sekunden zwischen Wiederholungen

# ─────────────────────────────────────────────
# KATEGORIEN
# ─────────────────────────────────────────────

CATEGORIES = """
1.  EINNAHMEN
    Kundenzahlungen, Gutschriften, Subventionen, Steuerrückerstattungen,
    Kapitaleinlagen, Zinserträge, Dividenden, Versicherungsleistungen

2.  AUSGABEN - INVESTITIONEN
    Maschinen, Einrichtung, Fahrzeugkauf, Immobilien,
    Firmenbeteiligungen, aktivierte Entwicklungskosten

3.  AUSGABEN - SOZIALVERSICHERUNGEN
    AHV, IV, EO, ALV, BVG/Pensionskasse, UVG, KTG, Familienzulagen

4.  AUSGABEN - BETRIEBSKOSTEN
    Miete, Nebenkosten (Strom, Wasser, Heizung), IT-Abos, Software,
    Hardware, Hosting, Telefonie, Buromaterial, Reinigung, Porto

5.  AUSGABEN - LIEFERANTEN & WARENEINKAUF
    Rohstoffe, Handelswaren, Verpackung, Subunternehmer,
    Freelancer, Logistik, Transport

6.  AUSGABEN - FAHRZEUG, REISE & SPESEN
    Leasing, Treibstoff, Fahrzeugunterhalt, Flug, Hotel,
    Bahn, Taxi, Geschaftsessen, Reprasentation,
    Spesenruckerstattungen an Mitarbeiter (LST SPESEN, Reisekosten)

7.  AUSGABEN - FINANZEN & BANKING
    Kontofuhrung, Transaktionsgebuhren, Uberziehungszinsen,
    Kreditruckzahlungen, Leasingraten, Fremdwahrungsgebuhren

8.  AUSGABEN - STEUERN & ABGABEN
    MWST-Zahlungen, direkte Steuern (Gewinn/Kapital),
    Quellensteuer, Steuervorauszahlungen

9.  AUSGABEN - VERSICHERUNGEN
    Betriebshaftpflicht, Sachversicherung, Rechtsschutz,
    Cyberversicherung, Transportversicherung

10. AUSGABEN - BERATUNG & DIENSTLEISTER
    Treuhandler, Buchfuhrung, Steuerberatung, Rechtsanwalt,
    Unternehmensberatung, HR, Recruitment

11. AUSGABEN - PERSONAL
    Lohne, Gehalter, Boni, Aushilfen, Geschaftsfuhrervergutung
    (nur direkte Lohnzahlungen - KEINE Spesenruckerstattungen,
     Spesen gehoren in Kategorie 6)

12. NEUTRALE / INTERNE BEWEGUNGEN
    Umbuchungen zwischen eigenen Konten, Korrekturen, Stornos,
    Festgeldanlagen, interne Transfers

13. SONDERKATEGORIEN
    Transaktionen die keiner anderen Kategorie zugeordnet werden konnen,
    oder wo die KI unsicher ist (confidence < 0.7)
"""

VALID_CATEGORIES = {
    "EINNAHMEN",
    "AUSGABEN - INVESTITIONEN",
    "AUSGABEN - SOZIALVERSICHERUNGEN",
    "AUSGABEN - BETRIEBSKOSTEN",
    "AUSGABEN - LIEFERANTEN & WARENEINKAUF",
    "AUSGABEN - FAHRZEUG, REISE & SPESEN",
    "AUSGABEN - FINANZEN & BANKING",
    "AUSGABEN - STEUERN & ABGABEN",
    "AUSGABEN - VERSICHERUNGEN",
    "AUSGABEN - BERATUNG & DIENSTLEISTER",
    "AUSGABEN - PERSONAL",
    "NEUTRALE / INTERNE BEWEGUNGEN",
    "SONDERKATEGORIEN",
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""Du bist ein Spezialist fuer Finanzbuchhaltung von KMUs im deutschsprachigen Raum (Oesterreich/Schweiz).
Deine Aufgabe ist es, Banktransaktionen praezise zu kategorisieren.

Die moeglichen Kategorien sind:
{CATEGORIES}

Wichtige Regeln:
- Weise jeder Transaktion GENAU eine Kategorie zu
- Nutze die EXAKTE Kategoriebezeichnung aus der Liste (kein Abweichen, kein Kuerzen)
- negativer Betrag = Ausgabe, positiver Betrag = Einnahme
- Bei Unsicherheit: waehle die wahrscheinlichste Kategorie und setze confidence unter 0.7
- Interne Umbuchungen erkennst du an: Gegenpartei ist das eigene Unternehmen / eigenes Konto,
  Schluesselwoerter wie "Umbuchung", "Transfer", "Projektkonto", runde Betraege ohne Geschaeftszweck
- SPESEN / Reisekosten-Erstattungen an Mitarbeiter -> Kategorie 6 (FAHRZEUG, REISE & SPESEN)
- GEHALTSÜBERWEISUNG / Lohnzahlungen -> Kategorie 11 (PERSONAL)
- SVA / Pensionskasse / Sozialversicherung -> Kategorie 3 (SOZIALVERSICHERUNGEN)
- Antworte NUR mit dem JSON-Array, absolut kein erklaerende Text davor oder danach"""


# ─────────────────────────────────────────────
# TINK FORMAT PARSER
# ─────────────────────────────────────────────

def _parse_betrag(amount: dict) -> float:
    """
    Rechnet Tink amount object in float um.
    Beispiel: { unscaledValue: "-780", scale: "2" }  ->  -7.80
    """
    try:
        unscaled = int(amount["value"]["unscaledValue"])
        scale    = int(amount["value"]["scale"])
        return round(unscaled / (10 ** scale), 2)
    except (KeyError, ValueError, ZeroDivisionError):
        return 0.0


def _parse_iban(counterparties: list) -> str | None:
    """Extrahiert IBAN der Gegenpartei aus counterparties-Array."""
    for cp in counterparties or []:
        iban = cp.get("identifiers", {}).get("iban", {}).get("iban")
        if iban:
            return iban
        acc = cp.get("identifiers", {}).get("financialInstitution", {}).get("accountNumber")
        if acc:
            return acc
    return None


def _parse_gegenpartei(txn: dict) -> str:
    """Gegenpartei: counterparties[0].name -> descriptions.display -> original."""
    for cp in txn.get("counterparties", []):
        name = cp.get("name", "").strip()
        if name:
            return name
    descs = txn.get("descriptions", {})
    return descs.get("display") or descs.get("original") or "Unbekannt"


def parse_tink_transactions(raw: list) -> list:
    """
    Konvertiert Tink /data/v2/transactions Response in internes Flat-Format.

    Filter: nur status == "BOOKED" (PENDING wird uebersprungen).

    Input-Felder (Tink verschachtelt):
        status
        dates.booked, dates.bookedDateTime
        amount.value.{unscaledValue, scale}
        descriptions.{original, display, detailed.unstructured}
        counterparties[].{name, identifiers.iban.iban}

    Output-Felder (intern flat):
        datum, bookedDateTime, betrag, verwendungszweck, gegenpartei, iban
    """
    result        = []
    skipped       = 0
    status_counts: dict[str, int] = {}

    for t in raw:
        status = t.get("status") or "UNKNOWN"
        status_counts[status] = status_counts.get(status, 0) + 1

        # Nur wirklich ungültige Transaktionen überspringen.
        # PENDING, UNKNOWN und alle anderen werden verarbeitet und mit Status markiert.
        if status in ("REJECTED", "REVERSED", "CANCELLED"):
            skipped += 1
            continue

        dates = t.get("dates", {})
        descs = t.get("descriptions", {})

        verwendungszweck = (
            descs.get("detailed", {}).get("unstructured")
            or descs.get("display")
            or descs.get("original")
            or ""
        )

        result.append({
            "datum":            dates.get("booked", ""),
            "bookedDateTime":   dates.get("bookedDateTime", ""),
            "betrag":           _parse_betrag(t.get("amount", {})),
            "verwendungszweck": verwendungszweck,
            "gegenpartei":      _parse_gegenpartei(t),
            "iban":             _parse_iban(t.get("counterparties", [])),
            "status":           status,
        })

    # Status-Übersicht immer ausgeben (nicht nur bei skipped)
    if status_counts:
        counts_str = "  ".join(f"{s}:{n}" for s, n in sorted(status_counts.items()))
        print(f"    i   Status: {counts_str}")
    if skipped:
        print(f"    i   {skipped} Transaktionen uebersprungen (REJECTED/REVERSED/CANCELLED)")

    return result


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def _make_client(api_key: str) -> anthropic.Anthropic:
    """Erstellt einen Anthropic-Client mit dem angegebenen API-Key."""
    return anthropic.Anthropic(api_key=api_key)


def extract_json(raw: str) -> list:
    """
    Extrahiert JSON-Array aus der API-Antwort.
    Unterstuetzt: reines JSON, ```json ... ```, ``` ... ```
    """
    text = raw.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def validate_result(result: dict) -> dict:
    """
    Prueft ein einzelnes Kategorisierungsergebnis.
    Fallback auf SONDERKATEGORIEN bei ungueltiger Kategorie.
    """
    cat = result.get("category_level1", "").strip()
    if cat not in VALID_CATEGORIES:
        # Normalisierung: em-dash "–" → normaler Bindestrich "-"
        normalized = cat.replace(" – ", " - ").replace(" — ", " - ")
        if normalized in VALID_CATEGORIES:
            result["category_level1"] = normalized
        else:
            result["category_level1"] = "SONDERKATEGORIEN"
            result["confidence"]      = min(result.get("confidence", 0.5), 0.5)
            result["reasoning"]       = (
                f"Ungueltige Kategorie '{cat}' -> SONDERKATEGORIEN. "
                f"Original: {result.get('reasoning', '')}"
            )

    try:
        conf = float(result.get("confidence", 0.5))
        result["confidence"] = round(max(0.0, min(1.0, conf)), 2)
    except (ValueError, TypeError):
        result["confidence"] = 0.5

    return result


def categorize_batch(transactions: list, batch_offset: int, client: anthropic.Anthropic) -> list:
    """
    Sendet einen Batch von Transaktionen an Claude zur Kategorisierung.

    Parameters:
        transactions  : Liste von internen Flat-Dicts (bereits geparst)
        batch_offset  : Globaler Index des ersten Elements
        client        : Anthropic-Client

    Returns:
        Liste von Dicts mit id, category_level1, confidence, reasoning.
        Laenge ist immer gleich len(transactions).
    """
    txns_for_prompt = [
        {
            "id":               i,
            "datum":            t["datum"],
            "betrag":           t["betrag"],
            "verwendungszweck": t.get("verwendungszweck", ""),
            "gegenpartei":      t.get("gegenpartei", ""),
        }
        for i, t in enumerate(transactions)
    ]

    user_message = (
        f"Kategorisiere diese {len(transactions)} Transaktionen.\n\n"
        "Antworte mit einem JSON-Array in exakt diesem Format (ein Objekt pro Transaktion):\n"
        "[\n"
        "  {\n"
        '    "id": 0,\n'
        '    "category_level1": "AUSGABEN - BETRIEBSKOSTEN",\n'
        '    "confidence": 0.95,\n'
        '    "reasoning": "Lastschrift Bueromiete"\n'
        "  }\n"
        "]\n\n"
        f"Transaktionen:\n{json.dumps(txns_for_prompt, ensure_ascii=False, indent=2)}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw     = response.content[0].text
            results = extract_json(raw)

            results_by_id = {r["id"]: r for r in results if "id" in r}

            final = []
            for i in range(len(transactions)):
                if i in results_by_id:
                    final.append(validate_result(results_by_id[i]))
                else:
                    final.append({
                        "id":              i,
                        "category_level1": "SONDERKATEGORIEN",
                        "confidence":      0.0,
                        "reasoning":       f"Kein Ergebnis von API fuer Index {batch_offset + i}",
                    })

            return final

        except json.JSONDecodeError as e:
            print(f"\n    API JSON-Fehler (Versuch {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except anthropic.APIStatusError as e:
            print(f"\n    API-Fehler {e.status_code} (Versuch {attempt}/{MAX_RETRIES}): {e.message}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

        except Exception as e:
            print(f"\n    Unbekannter Fehler (Versuch {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print(f"\n    Batch {batch_offset}-{batch_offset + len(transactions) - 1} konnte nicht kategorisiert werden.")
    return [
        {
            "id":              i,
            "category_level1": "SONDERKATEGORIEN",
            "confidence":      0.0,
            "reasoning":       "API-Fehler nach allen Wiederholungen",
        }
        for i in range(len(transactions))
    ]


def _merge_results(transactions: list, all_results: dict) -> list:
    """
    Fuehrt geparste Transaktionen und KI-Ergebnisse zusammen.

    Output-Format pro Eintrag (9 Felder):
        datum, bookedDateTime, betrag, verwendungszweck,
        gegenpartei, iban, category_level1, confidence, reasoning
    """
    categorized = []
    for i, txn in enumerate(transactions):
        r = all_results.get(i, {})
        categorized.append({
            "datum":            txn["datum"],
            "bookedDateTime":   txn.get("bookedDateTime", ""),
            "betrag":           txn["betrag"],
            "verwendungszweck": txn.get("verwendungszweck", ""),
            "gegenpartei":      txn.get("gegenpartei", ""),
            "iban":             txn.get("iban"),
            "status":           txn.get("status", "UNKNOWN"),
            "category_level1":  r.get("category_level1", "SONDERKATEGORIEN"),
            "confidence":       r.get("confidence", 0.0),
            "reasoning":        r.get("reasoning", ""),
        })
    return categorized


# ─────────────────────────────────────────────
# PUBLIC API  (fuer pipeline.py)
# ─────────────────────────────────────────────

def categorize(transactions: list, api_key: str) -> list:
    """
    Kategorisiert eine Liste von Tink-Rohtransaktionen vollstaendig im Memory.

    Parameters:
        transactions : Rohe Tink-Dicts aus tink_demo_transactions.json
        api_key      : Anthropic API-Key

    Returns:
        Liste mit 9 Feldern pro Transaktion:
        datum, bookedDateTime, betrag, verwendungszweck, gegenpartei,
        iban, category_level1, confidence, reasoning.
        Nur BOOKED. Bereit fuer detect_patterns.py.
    """
    # 1) Tink-Format parsen, nur BOOKED behalten
    parsed = parse_tink_transactions(transactions)

    client        = _make_client(api_key)
    total_batches = -(len(parsed) // -BATCH_SIZE)   # Ceiling division
    all_results: dict[int, dict] = {}

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(parsed))
        batch = parsed[start:end]

        print(f"    Batch {batch_num + 1:>2}/{total_batches}  ({start + 1:>3}-{end:>3})", end="  ", flush=True)

        results = categorize_batch(batch, batch_offset=start, client=client)

        for r in results:
            all_results[start + r["id"]] = r

        cats = sorted(set(
            r["category_level1"]
              .replace("AUSGABEN - ", "")
              .replace("NEUTRALE / INTERNE BEWEGUNGEN", "INTERN")
              .replace("SONDERKATEGORIEN", "SONDER")[:18]
            for r in results
        ))
        print(f"OK  {', '.join(cats)[:65]}")

    return _merge_results(parsed, all_results)


# ─────────────────────────────────────────────
# STANDALONE / CLI
# ─────────────────────────────────────────────

def print_summary(categorized: list):
    """Gibt Kategorien-Uebersicht und unsichere Transaktionen aus."""
    print(f"\n{'─'*60}")
    print("  KATEGORIEN-UEBERSICHT")
    print(f"{'─'*60}")

    counts = Counter(t["category_level1"] for t in categorized)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>3}x  {cat}")

    unsicher = [t for t in categorized if t.get("confidence", 1.0) < 0.7]
    if unsicher:
        print(f"\n  {len(unsicher)} unsichere Transaktionen (confidence < 0.7):")
        print(f"  {'Datum':<12} {'Zeit':<22} {'Betrag':>10}  {'Gegenpartei':<28}  Kategorie")
        print(f"  {'─'*12} {'─'*22} {'─'*10}  {'─'*28}  {'─'*28}")
        for t in unsicher:
            print(
                f"  {t['datum']:<12} {t.get('bookedDateTime',''):<22} {t['betrag']:>10.2f}  "
                f"{t.get('gegenpartei','')[:28]:<28}  "
                f"{t['category_level1']}  (conf: {t.get('confidence','?')})"
            )

    avg_conf = sum(t.get("confidence", 0) for t in categorized) / len(categorized) if categorized else 0
    print(f"\n  Durchschnitt Konfidenz : {avg_conf:.2%}")
    print(f"  Unsicher               : {len(unsicher)}/{len(categorized)} Transaktionen")


def main():
    print("=" * 60)
    print("  KI-Kategorisierung von Banktransaktionen")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n  ANTHROPIC_API_KEY nicht gesetzt.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"\n  Input-Datei nicht gefunden: {INPUT_FILE}")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    print(f"\n  {len(raw)} Tink-Transaktionen geladen")

    # Vorschau Parsing ohne API-Call
    parsed_preview = parse_tink_transactions(raw)
    total_batches  = -(len(parsed_preview) // -BATCH_SIZE)
    print(f"  BOOKED         : {len(parsed_preview)}")
    print(f"  Batch-Groesse  : {BATCH_SIZE}")
    print(f"  API-Calls      : {total_batches}")
    print(f"  Max Retries    : {MAX_RETRIES} pro Batch\n")

    categorized = categorize(raw, api_key=api_key)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(categorized, f, indent=2, ensure_ascii=False)

    print(f"\n  Gespeichert: {OUTPUT_FILE}  ({len(categorized)} Eintraege)")
    print(f"\n  Beispiel-Eintrag:")
    if categorized:
        for k, v in categorized[0].items():
            print(f"    {k:<18}: {v}")

    print_summary(categorized)

    print(f"\n{'─'*60}")
    print("  Naechster Schritt: python3 detect_patterns.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
