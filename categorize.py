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

2.  AUSGABEN – INVESTITIONEN
    Maschinen, Einrichtung, Fahrzeugkauf, Immobilien,
    Firmenbeteiligungen, aktivierte Entwicklungskosten

3.  AUSGABEN – SOZIALVERSICHERUNGEN
    AHV, IV, EO, ALV, BVG/Pensionskasse, UVG, KTG, Familienzulagen

4.  AUSGABEN – BETRIEBSKOSTEN
    Miete, Nebenkosten (Strom, Wasser, Heizung), IT-Abos, Software,
    Hardware, Hosting, Telefonie, Büromaterial, Reinigung, Porto

5.  AUSGABEN – LIEFERANTEN & WARENEINKAUF
    Rohstoffe, Handelswaren, Verpackung, Subunternehmer,
    Freelancer, Logistik, Transport

6.  AUSGABEN – FAHRZEUG, REISE & SPESEN
    Leasing, Treibstoff, Fahrzeugunterhalt, Flug, Hotel,
    Bahn, Taxi, Geschäftsessen, Repräsentation,
    Spesenrückerstattungen an Mitarbeiter (LST SPESEN, Reisekosten)

7.  AUSGABEN – FINANZEN & BANKING
    Kontoführung, Transaktionsgebühren, Überziehungszinsen,
    Kreditrückzahlungen, Leasingraten, Fremdwährungsgebühren

8.  AUSGABEN – STEUERN & ABGABEN
    MWST-Zahlungen, direkte Steuern (Gewinn/Kapital),
    Quellensteuer, Steuervorauszahlungen

9.  AUSGABEN – VERSICHERUNGEN
    Betriebshaftpflicht, Sachversicherung, Rechtsschutz,
    Cyberversicherung, Transportversicherung

10. AUSGABEN – BERATUNG & DIENSTLEISTER
    Treuhänder, Buchführung, Steuerberatung, Rechtsanwalt,
    Unternehmensberatung, HR, Recruitment

11. AUSGABEN – PERSONAL
    Löhne, Gehälter, Boni, Aushilfen, Geschäftsführervergütung
    (nur direkte Lohnzahlungen – KEINE Spesenrückerstattungen,
     Spesen gehören in Kategorie 6)

12. NEUTRALE / INTERNE BEWEGUNGEN
    Umbuchungen zwischen eigenen Konten, Korrekturen, Stornos,
    Festgeldanlagen, interne Transfers

13. SONDERKATEGORIEN
    Transaktionen die keiner anderen Kategorie zugeordnet werden können,
    oder wo die KI unsicher ist (confidence < 0.7)
"""

VALID_CATEGORIES = {
    "EINNAHMEN",
    "AUSGABEN – INVESTITIONEN",
    "AUSGABEN – SOZIALVERSICHERUNGEN",
    "AUSGABEN – BETRIEBSKOSTEN",
    "AUSGABEN – LIEFERANTEN & WARENEINKAUF",
    "AUSGABEN – FAHRZEUG, REISE & SPESEN",
    "AUSGABEN – FINANZEN & BANKING",
    "AUSGABEN – STEUERN & ABGABEN",
    "AUSGABEN – VERSICHERUNGEN",
    "AUSGABEN – BERATUNG & DIENSTLEISTER",
    "AUSGABEN – PERSONAL",
    "NEUTRALE / INTERNE BEWEGUNGEN",
    "SONDERKATEGORIEN",
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""Du bist ein Spezialist für Finanzbuchhaltung von KMUs im deutschsprachigen Raum (Schweiz).
Deine Aufgabe ist es, Banktransaktionen präzise zu kategorisieren.

Die möglichen Kategorien sind:
{CATEGORIES}

Wichtige Regeln:
- Weise jeder Transaktion GENAU eine Kategorie zu
- Nutze die EXAKTE Kategoriebezeichnung aus der Liste (kein Abweichen, kein Kürzen)
- negativer Betrag = Ausgabe, positiver Betrag = Einnahme
- Bei Unsicherheit: wähle die wahrscheinlichste Kategorie und setze confidence unter 0.7
- Interne Umbuchungen erkennst du an: Gegenpartei ist das eigene Unternehmen / eigenes Konto,
  Schlüsselwörter wie "Umbuchung", "Transfer", "Konto 4412", runde Beträge ohne Geschäftszweck
- LST SPESEN / Reisekosten-Erstattungen an Mitarbeiter → Kategorie 6 (FAHRZEUG, REISE & SPESEN)
- LST SAL / Lohnzahlungen → Kategorie 11 (PERSONAL)
- Antworte NUR mit dem JSON-Array, absolut kein erklärender Text davor oder danach"""


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def _make_client(api_key: str) -> anthropic.Anthropic:
    """Erstellt einen Anthropic-Client mit dem angegebenen API-Key."""
    return anthropic.Anthropic(api_key=api_key)


def extract_json(raw: str) -> list:
    """
    Extrahiert JSON-Array aus der API-Antwort.
    Unterstützt: reines JSON, ```json ... ```, ``` ... ```
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
    Prüft ein einzelnes Kategorisierungsergebnis:
    - category_level1 muss exakt in VALID_CATEGORIES sein
    - confidence muss float 0.0–1.0 sein
    - Fallback auf SONDERKATEGORIEN bei ungültigem Wert
    """
    cat = result.get("category_level1", "").strip()
    if cat not in VALID_CATEGORIES:
        normalized = cat.replace(" - ", " – ")
        if normalized in VALID_CATEGORIES:
            result["category_level1"] = normalized
        else:
            result["category_level1"] = "SONDERKATEGORIEN"
            result["confidence"]      = min(result.get("confidence", 0.5), 0.5)
            result["reasoning"]       = (
                f"Ungültige Kategorie '{cat}' → SONDERKATEGORIEN. "
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
        transactions  : Liste von Transaktions-Dicts (Rohformat)
        batch_offset  : Globaler Index des ersten Elements (für Fehlermeldungen)
        client        : Anthropic-Client (übergeben, nicht global)

    Returns:
        Liste von Dicts mit id, category_level1, confidence, reasoning.
        Länge ist immer gleich len(transactions) – fehlende werden mit SONDERKATEGORIEN gefüllt.
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
        '    "category_level1": "AUSGABEN – BETRIEBSKOSTEN",\n'
        '    "confidence": 0.95,\n'
        '    "reasoning": "MÜB = Mietüberweisung, Gegenpartei ist Immobilien AG"\n'
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
                        "reasoning":       f"Kein Ergebnis von API für Index {batch_offset + i}",
                    })

            return final

        except json.JSONDecodeError as e:
            print(f"\n    ⚠️  JSON-Fehler (Versuch {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except anthropic.APIStatusError as e:
            print(f"\n    ⚠️  API-Fehler {e.status_code} (Versuch {attempt}/{MAX_RETRIES}): {e.message}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # exponentielles Backoff

        except Exception as e:
            print(f"\n    ⚠️  Unbekannter Fehler (Versuch {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print(f"\n    ❌  Batch {batch_offset}–{batch_offset + len(transactions) - 1} konnte nicht kategorisiert werden.")
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
    Führt Original-Transaktionen und KI-Ergebnisse zu einer Liste zusammen.
    Gibt eine neue Liste zurück – die Originaldaten werden nicht verändert.
    """
    categorized = []
    for i, txn in enumerate(transactions):
        r = all_results.get(i, {})
        categorized.append({
            # Original-Felder
            "datum":            txn["datum"],
            "betrag":           txn["betrag"],
            "verwendungszweck": txn.get("verwendungszweck", ""),
            "gegenpartei":      txn.get("gegenpartei", ""),
            "iban":             txn.get("iban"),   # None bleibt None
            # KI-Ergebnis
            "category_level1":  r.get("category_level1", "SONDERKATEGORIEN"),
            "confidence":       r.get("confidence", 0.0),
            "reasoning":        r.get("reasoning", ""),
        })
    return categorized


# ─────────────────────────────────────────────
# PUBLIC API  (für pipeline.py)
# ─────────────────────────────────────────────

def categorize(transactions: list, api_key: str) -> list:
    """
    Kategorisiert eine Liste von Transaktionen vollständig im Memory.

    Parameters:
        transactions : Liste von Transaktions-Dicts (Rohformat aus tink_demo_transactions.json)
        api_key      : Anthropic API-Key

    Returns:
        Liste von Dicts mit allen Originalfeldern + category_level1, confidence, reasoning.
        Bereit für den nächsten Pipeline-Schritt (detect_patterns.py).
    """
    client        = _make_client(api_key)
    total_batches = -(len(transactions) // -BATCH_SIZE)  # Ceiling division
    all_results: dict[int, dict] = {}

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(transactions))
        batch = transactions[start:end]

        print(f"    Batch {batch_num + 1:>2}/{total_batches}  ({start + 1:>3}–{end:>3})", end="  ", flush=True)

        results = categorize_batch(batch, batch_offset=start, client=client)

        for r in results:
            all_results[start + r["id"]] = r

        # Kurze Kategorien-Übersicht pro Batch in der Pipeline-Ausgabe
        cats = sorted(set(
            r["category_level1"]
              .replace("AUSGABEN – ", "")
              .replace("NEUTRALE / INTERNE BEWEGUNGEN", "INTERN")
              .replace("SONDERKATEGORIEN", "SONDER")[:18]
            for r in results
        ))
        print(f"✅  {', '.join(cats)[:65]}")

    return _merge_results(transactions, all_results)


# ─────────────────────────────────────────────
# STANDALONE / CLI
# ─────────────────────────────────────────────

def print_summary(categorized: list):
    """Gibt Kategorien-Übersicht und unsichere Transaktionen aus."""
    print(f"\n{'─'*60}")
    print("  KATEGORIEN-ÜBERSICHT")
    print(f"{'─'*60}")

    counts = Counter(t["category_level1"] for t in categorized)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>3}×  {cat}")

    unsicher = [t for t in categorized if t.get("confidence", 1.0) < 0.7]
    if unsicher:
        print(f"\n  ⚠️  {len(unsicher)} unsichere Transaktionen (confidence < 0.7):")
        print(f"  {'Datum':<12} {'Betrag':>10}  {'Verwendungszweck':<35}  Kategorie")
        print(f"  {'─'*12} {'─'*10}  {'─'*35}  {'─'*30}")
        for t in unsicher:
            print(
                f"  {t['datum']:<12} {t['betrag']:>10.2f}  "
                f"{t.get('verwendungszweck', '')[:35]:<35}  "
                f"{t['category_level1']}  (conf: {t.get('confidence', '?')})"
            )

    avg_conf = sum(t.get("confidence", 0) for t in categorized) / len(categorized) if categorized else 0
    print(f"\n  Ø Konfidenz  : {avg_conf:.2%}")
    print(f"  Unsicher     : {len(unsicher)}/{len(categorized)} Transaktionen")


def main():
    print("=" * 60)
    print("  KI-Kategorisierung von Banktransaktionen")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌  ANTHROPIC_API_KEY nicht gesetzt.")
        print("    export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌  Input-Datei nicht gefunden: {INPUT_FILE}")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        transactions = json.load(f)

    total_batches = -(len(transactions) // -BATCH_SIZE)
    print(f"\n✅  {len(transactions)} Transaktionen geladen")
    print(f"    Batch-Grösse : {BATCH_SIZE}")
    print(f"    API-Calls    : {total_batches}")
    print(f"    Max Retries  : {MAX_RETRIES} pro Batch\n")

    # Standalone-Lauf nutzt die öffentliche categorize()-Funktion
    categorized = categorize(transactions, api_key=api_key)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(categorized, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Gespeichert: {OUTPUT_FILE}  ({len(categorized)} Einträge)")

    print_summary(categorized)

    print(f"\n{'─'*60}")
    print("  Nächster Schritt: python3 detect_patterns.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
