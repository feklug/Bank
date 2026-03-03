"""
simulate/categorize_simulation.py
==================================
Kategorisiert alle Transaktionen aus data/simulate.json —
EINE nach der anderen (ein API-Call pro Transaktion, kein Batch).

Unterschied zu training/categorize.py:
  - 1 TX = 1 API-Call  (statt 20 TX pro Batch)
  - Fortschritt wird TX-für-TX angezeigt
  - PENDING-Transaktionen werden mitverarbeitet (confidence ≤ 0.6)
  - Ausgabe-Format identisch mit training/categorize.py →
    is_there_a_pattern.py kann direkt damit arbeiten

Pfade:
  Input  : data/simulate.json             (via ENV INPUT_FILE)
  Output : data/tink_sim_categorized.json (via ENV OUTPUT_FILE)
"""

import json
import os
import pathlib
import time
from datetime import datetime

import anthropic

# ─────────────────────────────────────────────
# PFADE
# ─────────────────────────────────────────────

_ROOT = pathlib.Path(__file__).parent          # simulate/
_DATA = _ROOT.parent / "data"                  # data/

INPUT_FILE  = os.environ.get("INPUT_FILE",  str(_DATA / "simulate.json"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", str(_DATA / "tink_sim_categorized.json"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "5"))

# ─────────────────────────────────────────────
# KATEGORIEN
# ─────────────────────────────────────────────

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

SYSTEM_PROMPT = """Du bist ein Spezialist für Finanzbuchhaltung von KMUs im deutschsprachigen Raum (Österreich/Schweiz).
Deine Aufgabe ist es, eine einzelne Banktransaktion präzise zu kategorisieren.

Die möglichen Kategorien sind:
1.  EINNAHMEN
2.  AUSGABEN – INVESTITIONEN
3.  AUSGABEN – SOZIALVERSICHERUNGEN
4.  AUSGABEN – BETRIEBSKOSTEN
5.  AUSGABEN – LIEFERANTEN & WARENEINKAUF
6.  AUSGABEN – FAHRZEUG, REISE & SPESEN
7.  AUSGABEN – FINANZEN & BANKING
8.  AUSGABEN – STEUERN & ABGABEN
9.  AUSGABEN – VERSICHERUNGEN
10. AUSGABEN – BERATUNG & DIENSTLEISTER
11. AUSGABEN – PERSONAL
12. NEUTRALE / INTERNE BEWEGUNGEN
13. SONDERKATEGORIEN

Regeln:
- Weise GENAU eine Kategorie zu
- Nutze die EXAKTE Kategoriebezeichnung
- negativer Betrag = Ausgabe, positiver Betrag = Einnahme
- Bei Unsicherheit: confidence < 0.7 und SONDERKATEGORIEN
- LST SPESEN → Kategorie 6, LST SAL → Kategorie 11
- Antworte NUR mit einem einzigen JSON-Objekt, kein erklärender Text"""


# ─────────────────────────────────────────────
# TINK → FLACHES FORMAT
# ─────────────────────────────────────────────

def tink_to_flat(tx: dict) -> dict:
    """Konvertiert Tink-Format → internes flaches Format. Passthrough wenn bereits flach."""
    if "betrag" in tx:
        return tx

    amount_val = tx.get("amount", {}).get("value", {})
    unscaled   = int(amount_val.get("unscaledValue", 0))
    scale      = int(amount_val.get("scale", 2))
    betrag     = unscaled / (10 ** scale)

    datum = (
        tx.get("dates", {}).get("bookedDateTime")
        or tx.get("dates", {}).get("booked", "")
    )
    desc             = tx.get("descriptions", {})
    verwendungszweck = (
        desc.get("detailed", {}).get("unstructured")
        or desc.get("original", "")
    )
    counterparties = tx.get("counterparties", [])
    gegenpartei    = counterparties[0].get("name", "") if counterparties else ""
    iban           = (
        counterparties[0].get("identifiers", {}).get("iban", {}).get("iban")
        if counterparties else None
    )

    return {
        "datum":            datum,
        "betrag":           betrag,
        "verwendungszweck": verwendungszweck,
        "gegenpartei":      gegenpartei,
        "iban":             iban,
        "status":           tx.get("status", "BOOKED"),
        "tink_id":          tx.get("id", ""),
    }


# ─────────────────────────────────────────────
# VALIDIERUNG
# ─────────────────────────────────────────────

def _validate(result: dict) -> dict:
    cat = result.get("category_level1", "").strip()
    if cat not in VALID_CATEGORIES:
        normalized = cat.replace(" - ", " – ")
        if normalized in VALID_CATEGORIES:
            result["category_level1"] = normalized
        else:
            result["category_level1"] = "SONDERKATEGORIEN"
            result["confidence"]      = min(result.get("confidence", 0.5), 0.5)
            result["reasoning"]       = f"Ungültige Kategorie '{cat}' → SONDERKATEGORIEN"
    try:
        conf = float(result.get("confidence", 0.5))
        result["confidence"] = round(max(0.0, min(1.0, conf)), 2)
    except (ValueError, TypeError):
        result["confidence"] = 0.5
    return result


# ─────────────────────────────────────────────
# KERN: EINE TRANSAKTION KATEGORISIEREN
# Exportiert für simulate/pipeline.py
# ─────────────────────────────────────────────

def categorize_one(tx: dict, client: anthropic.Anthropic) -> dict:
    """
    Kategorisiert genau eine Transaktion.

    Parameter:
        tx     : Tink-Rohtransaktion ODER bereits flaches Dict
        client : anthropic.Anthropic Instanz (von pipeline.py übergeben)

    Rückgabe:
        Flaches Dict — identisches Format wie training/categorize.py Output:
        {
          "datum", "betrag", "verwendungszweck", "gegenpartei", "iban",
          "status", "tink_id",
          "category_level1", "confidence", "reasoning"
        }
    """
    flat       = tink_to_flat(tx)
    is_pending = flat.get("status", "BOOKED").upper() == "PENDING"

    user_message = (
        "Kategorisiere diese einzelne Transaktion.\n\n"
        "Antworte mit GENAU diesem JSON-Objekt (kein Array, kein Text drumherum):\n"
        "{\n"
        '  "category_level1": "AUSGABEN – BETRIEBSKOSTEN",\n'
        '  "confidence": 0.95,\n'
        '  "reasoning": "Kurze Begründung"\n'
        "}\n\n"
        f"Transaktion:\n"
        f"  Datum       : {flat.get('datum', '')}\n"
        f"  Betrag      : {flat.get('betrag', 0):.2f} EUR\n"
        f"  Verwendung  : {flat.get('verwendungszweck', '')}\n"
        f"  Gegenpartei : {flat.get('gegenpartei', '')}\n"
        f"  IBAN        : {flat.get('iban') or '–'}\n"
        + ("  Status      : PENDING (noch nicht final gebucht)\n" if is_pending else "")
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()

            if "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
                if raw.startswith("json"):
                    raw = raw[4:]
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            result = _validate(json.loads(raw.strip()))

            if is_pending:
                result["confidence"] = min(result["confidence"], 0.6)
                result["reasoning"]  = f"[PENDING] {result.get('reasoning', '')}"

            return {
                **flat,
                "category_level1": result["category_level1"],
                "confidence":      result["confidence"],
                "reasoning":       result.get("reasoning", ""),
            }

        except (json.JSONDecodeError, anthropic.APIStatusError, Exception):
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    return {
        **flat,
        "category_level1": "SONDERKATEGORIEN",
        "confidence":      0.0,
        "reasoning":       "API-Fehler nach allen Wiederholungen",
    }


# ─────────────────────────────────────────────
# HAUPTPROGRAMM
# ─────────────────────────────────────────────

def main():
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  KATEGORISIERUNG (Simulation)  –  1 TX pro API-Call")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
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

    # Chronologisch sortieren (Simulation muss zeitlich korrekt sein)
    def _sort_key(tx):
        d = tx.get("dates", {})
        return d.get("bookedDateTime") or d.get("booked") or tx.get("datum", "")

    raw_data.sort(key=_sort_key)

    # Nur BOOKED + PENDING verarbeiten
    to_process = [t for t in raw_data if t.get("status", "BOOKED").upper() in ("BOOKED", "PENDING")]
    skipped    = len(raw_data) - len(to_process)
    total      = len(to_process)

    print(f"\n  {len(raw_data)} Transaktionen geladen")
    print(f"  {total} werden kategorisiert  ({skipped} übersprungen)\n")

    client  = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    results = []
    errors  = 0

    for i, tx in enumerate(to_process, 1):
        result = categorize_one(tx, client)
        results.append(result)

        # Fortschritt (eine Zeile pro TX)
        cat_short = (
            result["category_level1"]
            .replace("AUSGABEN – ", "")
            .replace("NEUTRALE / INTERNE BEWEGUNGEN", "INTERN")
            .replace("SONDERKATEGORIEN", "SONDER")[:20]
        )
        flag = "⚠️ " if result["confidence"] < 0.7 else "✅ "
        print(
            f"  {flag} {i:>4}/{total}  "
            f"{result.get('datum', '')[:16]:<16}  "
            f"{result.get('betrag', 0):>10.2f} EUR  "
            f"{cat_short:<20}  "
            f"conf: {result['confidence']:.0%}",
            flush=True,
        )

        if result["confidence"] == 0.0:
            errors += 1

    # ── Speichern — identisches Format wie training/categorize.py ─
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*60}")
    print(f"  ✅  {len(results)} Transaktionen kategorisiert")
    if errors:
        print(f"  ⚠️   {errors} Fehler (API-Fallback → SONDERKATEGORIEN)")
    print(f"  💾  Gespeichert : {OUTPUT_FILE}")
    print(f"{'─'*60}")
    print(f"  Nächster Schritt: simulate/pipeline.py → is_there_a_pattern.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
