"""
categorize.py
=============
Kategorisiert Banktransaktionen via Claude API.

Kann auf zwei Arten verwendet werden:

  1. Als Modul (von pipeline.py aufgerufen):
       from categorize import categorize
       result = categorize(transactions)   # gibt Liste zurück, kein File

  2. Direkt ausführen (zum Testen):
       python categorize.py
       → liest tink_demo_transactions.json
       → gibt Ergebnis auf der Konsole aus
"""

import json
import os
import anthropic
from datetime import datetime


# ── Konfiguration ─────────────────────────────────────────────────

BATCH_SIZE = 25   # Transaktionen pro API-Call


# ── Kategorien (unveränderter Inhalt) ─────────────────────────────

CATEGORIES = """
1.  EINNAHMEN
    Kundenzahlungen, Gutschriften, Subventionen, Steuerrückerstattungen,
    Kapitaleinlagen, Zinserträge, Dividenden, Versicherungsleistungen

2.  AUSGABEN – PERSONAL
    Löhne, Gehälter, Boni, Spesen, Aushilfen, Geschäftsführervergütung

3.  AUSGABEN – SOZIALVERSICHERUNGEN
    AHV, IV, EO, ALV, BVG/Pensionskasse, UVG, KTG, Familienzulagen

4.  AUSGABEN – BETRIEBSKOSTEN
    Miete, Nebenkosten (Strom, Wasser, Heizung), IT-Abos, Software,
    Hardware, Hosting, Telefonie, Büromaterial, Reinigung, Porto

5.  AUSGABEN – LIEFERANTEN & WARENEINKAUF
    Rohstoffe, Handelswaren, Verpackung, Subunternehmer,
    Freelancer, Logistik, Transport

6.  AUSGABEN – FAHRZEUG & REISE
    Leasing, Treibstoff, Fahrzeugunterhalt, Flug, Hotel,
    Bahn, Taxi, Geschäftsessen, Repräsentation

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

11. AUSGABEN – INVESTITIONEN
    Maschinen, Einrichtung, Fahrzeugkauf, Immobilien,
    Firmenbeteiligungen, aktivierte Entwicklungskosten

12. NEUTRALE / INTERNE BEWEGUNGEN
    Umbuchungen zwischen eigenen Konten, Korrekturen, Stornos,
    Festgeldanlagen, interne Transfers

13. SONDERKATEGORIEN
    Transaktionen die keiner anderen Kategorie zugeordnet werden können,
    oder wo die KI unsicher ist
"""

SYSTEM_PROMPT = f"""Du bist ein Spezialist für Finanzbuchhaltung von KMUs im deutschsprachigen Raum.
Deine Aufgabe ist es, Banktransaktionen zu kategorisieren.

Die möglichen Kategorien sind:
{CATEGORIES}

Regeln:
- Weise jeder Transaktion genau eine Kategorie zu
- Nutze die exakte Kategoriebezeichnung aus der Liste
- Berücksichtige: negativer Betrag = Ausgabe, positiver Betrag = Einnahme
- Bei Unsicherheit: wähle die wahrscheinlichste Kategorie und setze confidence < 0.7
- Interne Umbuchungen erkennst du an: gleiche Firma als Gegenpartei, "Umbuchung", "Transfer", runde Beträge ohne klaren Geschäftszweck
- Antworte NUR mit dem JSON-Array, kein erklärender Text"""


# ── Einen Batch kategorisieren ────────────────────────────────────

def _categorize_batch(client: anthropic.Anthropic, batch: list) -> list:
    """
    Sendet einen Batch an Claude und gibt die Kategorisierungen zurück.
    Interner Hilfsaufruf – wird von categorize() verwendet.
    """
    # Nur die Felder schicken die Claude braucht
    slim = [
        {
            "id":              i,
            "datum":           t["datum"],
            "betrag":          t["betrag"],
            "verwendungszweck": t["verwendungszweck"],
            "gegenpartei":     t["gegenpartei"],
        }
        for i, t in enumerate(batch)
    ]

    user_message = f"""Kategorisiere diese {len(batch)} Transaktionen.

Antworte mit einem JSON-Array in diesem Format:
[
  {{
    "id": 0,
    "category_level1": "AUSGABEN – BETRIEBSKOSTEN",
    "confidence": 0.95,
    "reasoning": "Mietüberweisung erkennbar an MÜB-Kürzel und Gewerberaum-Gegenpartei"
  }}
]

Transaktionen:
{json.dumps(slim, ensure_ascii=False, indent=2)}"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()

    # JSON sauber extrahieren (falls Claude Markdown-Backticks hinzufügt)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    return json.loads(raw)


# ── Hauptfunktion (wird von pipeline.py aufgerufen) ───────────────

def categorize(transactions: list, api_key: str = None) -> list:
    """
    Kategorisiert alle Transaktionen via Claude API.

    Eingabe:  Liste von Transaktionen (datum, betrag, verwendungszweck, gegenpartei)
    Ausgabe:  Gleiche Liste + category_level1, confidence, reasoning pro Eintrag

    Kein File wird geschrieben – alles bleibt im Memory.

    Args:
        transactions:  Liste aus tink_auth.py (oder direkt aus JSON geladen)
        api_key:       Anthropic API Key. Falls leer: wird aus
                       Umgebungsvariable ANTHROPIC_API_KEY gelesen.
    """

    # API Key holen
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "Kein API Key gefunden.\n"
            "Setze die Umgebungsvariable: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=key)

    total_batches = -(len(transactions) // -BATCH_SIZE)  # Aufrundung ohne math
    print(f"  {len(transactions)} Transaktionen → {total_batches} API-Calls (Batch-Grösse: {BATCH_SIZE})")

    # Ergebnisse sammeln: {original_index: result}
    result_map = {}

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(transactions))
        batch = transactions[start:end]

        print(f"  Batch {batch_num + 1}/{total_batches} ({start + 1}–{end})...", end=" ", flush=True)

        results = _categorize_batch(client, batch)

        # Ergebnisse mit globalem Index speichern
        for r in results:
            global_index = start + r["id"]
            result_map[global_index] = r

        # Kurze Zusammenfassung was erkannt wurde
        cats = list(set(
            r["category_level1"]
             .replace("AUSGABEN – ", "")
             .replace("AUSGABEN - ", "")[:18]
            for r in results
        ))
        print(f"✅  ({', '.join(cats[:3])}{'...' if len(cats) > 3 else ''})")

    # Kategorien zu den originalen Transaktionen hinzufügen
    # Die originalen Felder bleiben unverändert – wir fügen nur hinzu
    categorized = []
    for i, txn in enumerate(transactions):
        result = result_map.get(i, {})
        enriched = {
            **txn,                          # alle originalen Felder behalten
            "category_level1": result.get("category_level1", "SONDERKATEGORIEN"),
            "confidence":      result.get("confidence", 0.0),
            "reasoning":       result.get("reasoning", ""),
        }
        categorized.append(enriched)

    # Warnhinweis für unsichere Kategorisierungen
    unsicher = [t for t in categorized if t.get("confidence", 1.0) < 0.7]
    if unsicher:
        print(f"\n  ⚠️  {len(unsicher)} unsichere Kategorisierungen (confidence < 0.7):")
        for t in unsicher:
            print(f"     {t['datum']} | {t['betrag']:>10.2f} | "
                  f"{t['verwendungszweck'][:35]}")
            print(f"     → {t['category_level1']} "
                  f"(confidence: {t.get('confidence', '?')})")

    return categorized   # ← gibt Liste zurück, schreibt NICHTS auf Disk


# ── Direkt ausführen (nur zum Testen) ────────────────────────────
# Wenn du "python categorize.py" ausführst, lädt es tink_demo_transactions.json
# und gibt das Ergebnis aus. So kannst du die Datei einzeln testen.

if __name__ == "__main__":
    INPUT_FILE = "tink_demo_transactions.json"

    print("=" * 55)
    print("  categorize.py – Direkttest")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    # Datei laden
    with open(INPUT_FILE, encoding="utf-8") as f:
        transactions = json.load(f)
    print(f"✅ {len(transactions)} Transaktionen aus {INPUT_FILE} geladen\n")

    # Kategorisieren
    result = categorize(transactions)

    # Zusammenfassung ausgeben
    from collections import Counter
    counts = Counter(t["category_level1"] for t in result)
    print(f"\n{'─' * 55}")
    print("  KATEGORIEN-ÜBERSICHT")
    print(f"{'─' * 55}")
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>3}x  {cat}")

    print(f"\n✅ Fertig. Nächster Schritt: detect_patterns.py")
