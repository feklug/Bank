"""
categorize.py  –  KI-Kategorisierung von Banktransaktionen
Läuft lokal UND als GitHub Action (Input/Output via Env-Variablen konfigurierbar).
"""

import json
import os
import time
from collections import Counter
from datetime import datetime

import anthropic

# ─────────────────────────────────────────────
# KONFIGURATION  (Env-Variablen überschreiben Defaults)
# ─────────────────────────────────────────────

INPUT_FILE   = os.environ.get("INPUT_FILE",  "tink_demo_transactions.json")
OUTPUT_FILE  = os.environ.get("OUTPUT_FILE", "tink_categorized.json")
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE",   "20"))   # Transaktionen pro API-Call
MAX_RETRIES  = int(os.environ.get("MAX_RETRIES",   "3"))   # Wiederholungen bei API-Fehler
RETRY_DELAY  = int(os.environ.get("RETRY_DELAY",   "5"))   # Sekunden zwischen Versuchen

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
# API CLIENT
# ─────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def extract_json(raw: str) -> list:
    """Extrahiert JSON-Array aus der API-Antwort (robust gegen Markdown-Fences)."""
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
    """Normalisiert Kategoriename und confidence; fällt bei ungültigem Wert auf SONDERKATEGORIEN zurück."""
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


def categorize_batch(transactions: list, batch_offset: int) -> list:
    """Sendet einen Batch an Claude und gibt kategorisierte Ergebnisse zurück."""
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
    model="claude-haiku-4-5-20251001",
    max_tokens=4096,
    temperature=0,   # ← diese Zeile hinzufügen
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_message}],
)

            raw          = response.content[0].text
            results      = extract_json(raw)
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
                time.sleep(RETRY_DELAY * attempt)

        except Exception as e:
            print(f"\n    ⚠️  Unbekannter Fehler (Versuch {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print(f"\n    ❌  Batch {batch_offset}–{batch_offset + len(transactions) - 1} fehlgeschlagen.")
    return [
        {
            "id":              i,
            "category_level1": "SONDERKATEGORIEN",
            "confidence":      0.0,
            "reasoning":       "API-Fehler nach allen Wiederholungen",
        }
        for i in range(len(transactions))
    ]


# ─────────────────────────────────────────────
# ZUSAMMENFASSUNG
# ─────────────────────────────────────────────

def print_summary(categorized: list):
    print(f"\n{'─'*60}")
    print("  KATEGORIEN-ÜBERSICHT")
    print(f"{'─'*60}")

    counts = Counter(t["category_level1"] for t in categorized)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>3}×  {cat}")

    unsicher = [t for t in categorized if t.get("confidence", 1.0) < 0.7]
    if unsicher:
        print(f"\n  ⚠️  {len(unsicher)} unsichere Transaktionen (confidence < 0.7):")
        print(f"  {'Datum/Zeit':<22} {'Betrag':>10}  {'Verwendungszweck':<35}  Kategorie")
        print(f"  {'─'*22} {'─'*10}  {'─'*35}  {'─'*30}")
        for t in unsicher:
            print(
                f"  {t['datum']:<22} {t['betrag']:>10.2f}  "
                f"{t.get('verwendungszweck', '')[:35]:<35}  "
                f"{t['category_level1']}  (conf: {t.get('confidence', '?')})"
            )

    avg_conf = sum(t.get("confidence", 0) for t in categorized) / len(categorized) if categorized else 0
    print(f"\n  Ø Konfidenz  : {avg_conf:.2%}")
    print(f"  Unsicher     : {len(unsicher)}/{len(categorized)} Transaktionen")


# ─────────────────────────────────────────────
# HAUPTPROGRAMM
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  KI-Kategorisierung von Banktransaktionen")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}")

    # ── API-Key prüfen ────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌  ANTHROPIC_API_KEY nicht gesetzt.")
        print("    Lokal  : export ANTHROPIC_API_KEY='sk-ant-...'")
        print("    GitHub : Settings → Secrets → ANTHROPIC_API_KEY")
        raise SystemExit(1)

    # ── Input laden ───────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌  Input-Datei nicht gefunden: {INPUT_FILE}")
        raise SystemExit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        raw_data = json.load(f)

    # ── Tink-Format → internes Format ────────────────────────────
    transactions = []
    skipped = 0
    for t in raw_data:
        if t.get("status", "").upper() != "BOOKED":
            skipped += 1
            continue

        amount_val = t.get("amount", {}).get("value", {})
        unscaled   = int(amount_val.get("unscaledValue", 0))
        scale      = int(amount_val.get("scale", 2))
        betrag     = unscaled / (10 ** scale)

        datum = t.get("dates", {}).get("bookedDateTime") or t.get("dates", {}).get("booked", "")

        desc             = t.get("descriptions", {})
        verwendungszweck = (
            desc.get("detailed", {}).get("unstructured")
            or desc.get("original", "")
        )

        counterparties = t.get("counterparties", [])
        gegenpartei    = counterparties[0].get("name", "") if counterparties else ""
        iban           = (
            counterparties[0].get("identifiers", {}).get("iban", {}).get("iban")
            if counterparties else None
        )

        transactions.append({
            "datum":            datum,
            "betrag":           betrag,
            "verwendungszweck": verwendungszweck,
            "gegenpartei":      gegenpartei,
            "iban":             iban,
        })

    total_batches = -(len(transactions) // -BATCH_SIZE)
    print(f"\n✅  {len(raw_data)} Einträge  →  {len(transactions)} BOOKED  ({skipped} übersprungen)")
    print(f"    Batches : {total_batches} × {BATCH_SIZE}\n")

    # ── Output-Verzeichnis anlegen (falls nötig) ──────────────────
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Batches verarbeiten ───────────────────────────────────────
    all_results: dict[int, dict] = {}

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(transactions))
        batch = transactions[start:end]

        print(f"  Batch {batch_num + 1:>2}/{total_batches}  ({start + 1:>3}–{end:>3})", end="  ", flush=True)

        results = categorize_batch(batch, batch_offset=start)

        for r in results:
            all_results[start + r["id"]] = r

        cats = sorted(set(
            r["category_level1"]
              .replace("AUSGABEN – ", "")
              .replace("NEUTRALE / INTERNE BEWEGUNGEN", "INTERN")
              .replace("SONDERKATEGORIEN", "SONDER")[:18]
            for r in results
        ))
        print(f"✅  {', '.join(cats)[:65]}")

    # ── Ergebnisse zusammenführen ─────────────────────────────────
    categorized = []
    for i, txn in enumerate(transactions):
        r = all_results.get(i, {})
        categorized.append({
            "datum":            txn["datum"],
            "betrag":           txn["betrag"],
            "verwendungszweck": txn.get("verwendungszweck", ""),
            "gegenpartei":      txn.get("gegenpartei", ""),
            "iban":             txn.get("iban"),
            "category_level1":  r.get("category_level1", "SONDERKATEGORIEN"),
            "confidence":       r.get("confidence", 0.0),
            "reasoning":        r.get("reasoning", ""),
        })

    # ── Output speichern ──────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(categorized, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Gespeichert: {OUTPUT_FILE}  ({len(categorized)} Einträge)")

    print_summary(categorized)

    print(f"\n{'─'*60}")
    print("  Nächster Schritt: python3 detect_patterns.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
