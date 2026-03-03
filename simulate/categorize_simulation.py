"""
simulate/categorize_simulation.py
==================================
Kategorisiert GENAU EINE Transaktion via Anthropic API.
Wird von simulate/pipeline.py pro Run aufgerufen.

Kein main()-Loop — nur die Funktion categorize_one(tx) → dict.

Rückgabe-Format:
  {
    "datum":            "2025-01-01T01:30:40Z",
    "betrag":           -3200.00,
    "verwendungszweck": "Büromiete 01/2025",
    "gegenpartei":      "Bürowelt GmbH",
    "iban":             "AT61 1904 3002 3457 3201",
    "category_level1":  "AUSGABEN – BETRIEBSKOSTEN",
    "confidence":       0.95,
    "reasoning":        "Lastschrift Büromiete → Betriebskosten",
    "status":           "BOOKED",
    "tink_id":          "1744cb47dd51de6c8f82d28a",
  }
"""

import json
import os
import time

import anthropic

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "5"))

VALID_CATEGORIES = {
    "EINNAHMEN", "AUSGABEN – INVESTITIONEN", "AUSGABEN – SOZIALVERSICHERUNGEN",
    "AUSGABEN – BETRIEBSKOSTEN", "AUSGABEN – LIEFERANTEN & WARENEINKAUF",
    "AUSGABEN – FAHRZEUG, REISE & SPESEN", "AUSGABEN – FINANZEN & BANKING",
    "AUSGABEN – STEUERN & ABGABEN", "AUSGABEN – VERSICHERUNGEN",
    "AUSGABEN – BERATUNG & DIENSTLEISTER", "AUSGABEN – PERSONAL",
    "NEUTRALE / INTERNE BEWEGUNGEN", "SONDERKATEGORIEN",
}

SYSTEM_PROMPT = """Du bist ein Spezialist für Finanzbuchhaltung von KMUs im deutschsprachigen Raum (Österreich/Schweiz).
Deine Aufgabe ist es, eine einzelne Banktransaktion präzise zu kategorisieren.

Die möglichen Kategorien sind:
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


def tink_to_flat(tx: dict) -> dict:
    """Konvertiert Tink-Rohtransaktion → flaches Format. Passthrough wenn bereits flach."""
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
        "datum": datum, "betrag": betrag,
        "verwendungszweck": verwendungszweck,
        "gegenpartei": gegenpartei, "iban": iban,
        "status": tx.get("status", "BOOKED"),
        "tink_id": tx.get("id", ""),
    }


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


def categorize_one(tx: dict) -> dict:
    """
    Kategorisiert genau eine Transaktion.

    Parameter:
        tx : Tink-Rohtransaktion ODER bereits flaches Dict

    Rückgabe:
        Flaches Dict mit category_level1, confidence, reasoning.
        Bei PENDING: confidence max 0.6.
        Bei API-Fehler: SONDERKATEGORIEN, confidence 0.0.
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
        f"  Datum        : {flat.get('datum', '')}\n"
        f"  Betrag       : {flat.get('betrag', 0):.2f} EUR\n"
        f"  Verwendung   : {flat.get('verwendungszweck', '')}\n"
        f"  Gegenpartei  : {flat.get('gegenpartei', '')}\n"
        f"  IBAN         : {flat.get('iban') or '–'}\n"
        + (f"  Status       : PENDING (noch nicht final gebucht)\n" if is_pending else "")
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
            return {**flat, **{
                "category_level1": result["category_level1"],
                "confidence":      result["confidence"],
                "reasoning":       result.get("reasoning", ""),
            }}
        except (json.JSONDecodeError, anthropic.APIStatusError, Exception):
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    return {**flat, **{
        "category_level1": "SONDERKATEGORIEN",
        "confidence":      0.0,
        "reasoning":       "API-Fehler nach allen Wiederholungen",
    }}
