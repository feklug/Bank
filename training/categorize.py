"""
training/categorize.py  –  KI-Kategorisierung von Banktransaktionen
Läuft lokal UND als GitHub Action (Input/Output via Env-Variablen konfigurierbar).

Pfade:
  Input  : data/training.json          (via ENV INPUT_FILE)
  Output : data/tink_categorized.json  (via ENV OUTPUT_FILE)
"""

import json
import os
import pathlib
import time
from collections import Counter
from datetime import datetime

import anthropic

# ─────────────────────────────────────────────
# PFADE  (Env-Variablen überschreiben Defaults)
# ─────────────────────────────────────────────

_ROOT  = pathlib.Path(__file__).parent        # training/
_DATA  = _ROOT.parent / "data"                # data/

INPUT_FILE  = os.environ.get("INPUT_FILE",  str(_DATA / "training.json"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", str(_DATA / "tink_categorized.json"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES",  "3"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY",  "5"))

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
    (nur direkte Lohnzahlungen – KEINE Spesenrückerstattungen)

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

SYSTEM_PROMPT = f"""Du bist ein Spezialist für Finanzbuchhaltung von KMUs im deutschsprachigen Raum (Österreich/Schweiz).
Deine Aufgabe ist es, Banktransaktionen präzise zu kategorisieren.

Die möglichen Kategorien sind:
{CATEGORIES}

Wichtige Regeln:
- Weise jeder Transaktion GENAU eine Kategorie zu
- Nutze die EXAKTE Kategoriebezeichnung aus der Liste
- negativer Betrag = Ausgabe, positiver Betrag = Einnahme
- Bei Unsicherheit: wähle die wahrscheinlichste Kategorie und setze confidence unter 0.7
- Interne Umbuchungen erkennst du an: Gegenpartei ist das eigene Unternehmen / eigenes Konto
- LST SPESEN / Reisekosten-Erstattungen → Kategorie 6
- LST SAL / Lohnzahlungen → Kategorie 11
- Antworte NUR mit dem JSON-Array, absolut kein erklärender Text davor oder danach"""

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def extract_json(raw: str) -> list:
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
        "Antworte mit einem JSON-Array in exakt diesem Format:\n"
        "[\n"
        "  {\n"
        '    "id": 0,\n'
        '    "category_level1": "AUSGABEN – BETRIEBSKOSTEN",\n'
        '    "confidence": 0.95,\n'
        '    "reasoning": "Kurze Begründung"\n'
        "  }\n"
        "]\n\n"
        f"Transaktionen:\n{json.dumps(txns_for_prompt, ensure_ascii=False, indent=2)}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw           = response.content[0].text
            results       = extract_json(raw)
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
                        "reasoning":       f"Kein Ergebnis für Index {batch_offset + i}",
                    })
            return final

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except anthropic.APIStatusError as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return [
        {"id": i, "category_level1": "SONDERKATEGORIEN", "confidence": 0.0,
         "reasoning": "API-Fehler nach allen Wiederholungen"}
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
        print(f"\n  ⚠️  {len(unsicher)} unsichere Transaktionen (confidence < 0.7)")
    avg_conf = sum(t.get("confidence", 0) for t in categorized) / len(categorized) if categorized else 0
    print(f"\n  Ø Konfidenz : {avg_conf:.2%}")
    print(f"  Unsicher    : {len(unsicher)}/{len(categorized)}")


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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌  ANTHROPIC_API_KEY nicht gesetzt.")
        raise SystemExit(1)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌  Input-Datei nicht gefunden: {INPUT_FILE}")
        raise SystemExit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        raw_data = json.load(f)

    def _clean_iban(raw) -> str | None:
        """Gibt None zurück wenn IBAN leer/null, sonst getrimmten String."""
        if raw is None or str(raw).strip().lower() in ("null", "none", ""):
            return None
        return str(raw).strip()

    transactions = []
    for t in raw_data:
        transactions.append({
            "datum":            str(t.get("datum", "")),
            "betrag":           float(t.get("betrag", 0)),
            "verwendungszweck": str(t.get("verwendungszweck", "")),
            "gegenpartei":      str(t.get("gegenpartei", "")),
            "iban":             _clean_iban(t.get("iban")),
        })

    total_batches = -(len(transactions) // -BATCH_SIZE)
    print(f"\n✅  {len(raw_data)} Einträge → {len(transactions)} Transaktionen")
    print(f"    Batches: {total_batches} × {BATCH_SIZE}\n")

    # Output-Verzeichnis sicherstellen
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[int, dict] = {}
    for batch_num in range(total_batches):
        start   = batch_num * BATCH_SIZE
        end     = min(start + BATCH_SIZE, len(transactions))
        batch   = transactions[start:end]
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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(categorized, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Gespeichert: {OUTPUT_FILE}  ({len(categorized)} Einträge)")
    print_summary(categorized)
    print(f"\n{'─'*60}")
    print("  Nächster Schritt: detect_patterns.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
