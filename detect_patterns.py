"""
detect_patterns.py
==================
Pattern Detection Engine – KMU Liquiditätsforecasting
Österreich (AT) | Version 2.1

WICHTIG: Diese Datei verwendet KEINE KI.
  → Statistik ist deterministisch: gleiche Daten = immer gleiche Antwort
  → Jedes Ergebnis ist erklärbar und auditierbar
  → Kein API-Call = kostenlos, sofort, skalierbar

VERWENDUNG:
  Als Modul (von pipeline.py):
    from detect_patterns import detect_patterns
    result = detect_patterns(categorized_transactions)

  Direkt zum Testen (loggt alles):
    python detect_patterns.py
    → liest tink_categorized.json
    → gibt alle erkannten Patterns detailliert aus
    → schreibt KEINE Datei (alles Memory)

VERARBEITUNGSREIHENFOLGE (kritisch – Reihenfolge nicht ändern):
  1. BATCH       → Mehrere Transaktionen gleicher Kategorie am gleichen Tag
  2. ANOMALIE    → Ausreisser erkennen VOR Recurring (sonst verfälscht)
  3. RECURRING   → Wiederkehrende Zahlungen (Anomalien ausgeschlossen)
  4. SEASONAL    → Saisonale Muster (nur wenn NICHT recurring)
  5. SEQUENTIAL  → Kausale Abfolgen A → B (Content-Match Pflicht)
  6. GEGENLÄUFIG → Ausgabe korreliert mit nachfolgender Einnahme

ÄNDERUNGEN v2.1:
  FIX  #1: unexpected_new feuert nicht mehr auf bekannte Gegenparteien (n=1,2)
           → n=0 → unexpected_new (Gegenpartei wirklich unbekannt)
           → n=1,2 → insufficient_data (kein Ausschluss aus Recurring!)
           → n>=3 → statistischer Vergleich
  FIX  #2: anomaly_reference_avg zeigt jetzt echten Median statt 0.0
  FIX  #3: anomaly_deviation korrekt None wenn keine Referenz-Basis vorhanden
  FIX  #4: Sequential verhindert Cross-Entity False Positives bei Gehaltsläufen
           (verschiedene Gegenparteien + gleiche Kategorie + gleiche Richtung)
  FIX  #5: Inflation-Toleranz gilt nur für Ausgaben (nicht Einnahmen)
  FIX  #6: Strengere Zeit-Konsistenz bei n == recurring_min_txns
  NEU  #7: IBAN-basiertes Matching (optional, schlägt Name wenn vorhanden)
  NEU  #8: Content-Match-Cache → O(n²) statt O(n⁴) für Sequential
  NEU  #9: Anomalie-Typ insufficient_data (is_anomaly=False, kein Recurring-Ausschluss)
"""

import json
import re
import statistics
import math
from datetime import datetime, timedelta, date
from collections import defaultdict
from typing import Optional
from dateutil.relativedelta import relativedelta


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# Einheitliche Log-Funktionen damit GitHub Actions Logs gut lesbar sind.
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    """Standard-Log-Ausgabe."""
    print(msg)

def _log_section(title: str):
    """Sektions-Trenner für bessere Lesbarkeit."""
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")

def _log_subsection(title: str):
    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")

def _log_pattern(label: str, fields: dict, txn: dict):
    """Gibt ein erkanntes Pattern detailliert aus."""
    betrag_str = f"{txn['betrag']:>10.2f} CHF"
    _log(f"\n  ✅ {label}")
    _log(f"     Transaktion : {txn['datum']} | {betrag_str} | {txn['gegenpartei']}")
    _log(f"     Zweck       : {txn['verwendungszweck']}")
    _log(f"     Kategorie   : {txn.get('category_level1', 'n/a')}")
    for k, v in fields.items():
        if v is not None and v is not False and v != [] and k not in ["is_batch","is_anomaly","is_recurring","is_seasonal","is_sequential","is_counter"]:
            _log(f"     {k:<35} {v}")

def _log_anomaly(txn: dict, fields: dict, ref_amounts: list, score: float, method: str):
    """Gibt eine erkannte Anomalie mit voller Begründung aus."""
    severity = fields.get('anomaly_severity', 'info')
    icon = "ℹ️ " if severity == "info" else "⚠️ "
    _log(f"\n  {icon} ANOMALIE [{severity.upper()}]")
    _log(f"     Transaktion : {txn['datum']} | {txn['betrag']:>10.2f} | {txn['gegenpartei']}")
    _log(f"     Typ         : {fields['anomaly_type']}")
    _log(f"     Methode     : {method}")
    if ref_amounts:
        _log(f"     Referenz-Ø  : {fields['anomaly_reference_avg']:.2f}  "
             f"(aus {len(ref_amounts)} Transaktionen: "
             f"{[round(x,2) for x in sorted(ref_amounts)]})")
    if fields.get('anomaly_deviation') is not None:
        _log(f"     Abweichung  : {fields['anomaly_deviation']:+.1%} vom Referenz-Median")
    if score is not None:
        _log(f"     Score       : {score:.3f}")

def _log_recurring(key: str, fields: dict, txn: dict, dates: list, amounts: list,
                   time_cons: float, amount_cons: float):
    """Gibt ein Recurring-Pattern mit statistischer Begründung aus."""
    _log(f"\n  🔁 RECURRING")
    _log(f"     Gegenpartei : {txn['gegenpartei']}")
    iban = txn.get("iban")
    if iban:
        _log(f"     IBAN        : {iban}")
    _log(f"     Kategorie   : {txn.get('category_level1','n/a')}")
    _log(f"     Intervall   : {fields['recurrence_interval']}")
    _log(f"     Stichprobe  : {fields['recurrence_sample_size']} Transaktionen")
    _log(f"     Daten       : {[d.strftime('%Y-%m-%d') for d in sorted(dates)]}")
    _log(f"     Beträge     : {[round(a,2) for a in amounts]} CHF")
    _log(f"     Betrag-Ø    : {fields['recurrence_amount_avg']:.2f} CHF")
    _log(f"     Zeit-Konsistenz   : {time_cons:.3f}  (Schwelle: < {CFG['recurring_time_consistency']})")
    _log(f"     Betrags-Konsistenz: {amount_cons:.3f}  (Schwelle: < {CFG['recurring_amount_tolerance']})")
    _log(f"     Confidence  : {fields['recurrence_confidence']:.3f}  "
         f"(Minimum: {CFG['recurring_min_confidence']})")
    _log(f"     Nächstes    : {fields['next_expected_date']}")
    if fields.get('recurrence_day_of_month'):
        _log(f"     Tag/Monat   : {fields['recurrence_day_of_month']}.")
    if fields.get('recurrence_has_gaps'):
        _log(f"     ⚠️  Lücken erkannt (Monate ohne Buchung)")


# ═══════════════════════════════════════════════════════════════════════════════
# ÖSTERREICHISCHER KALENDER
# Wird verwendet für:
#   - next_expected_date auf nächsten Arbeitstag verschieben
#   - WE/Feiertag-Verschiebungen werden nicht als Timing-Anomalie gewertet
# ═══════════════════════════════════════════════════════════════════════════════

def _ostersonntag(jahr: int) -> date:
    """
    Berechnet Ostersonntag – Anonymer Gregorianischer Algorithmus.
    Keine externe Bibliothek nötig.
    """
    a = jahr % 19
    b = jahr // 100
    c = jahr % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    monat = (h + l - 7 * m + 114) // 31
    tag   = ((h + l - 7 * m + 114) % 31) + 1
    return date(jahr, monat, tag)


_AT_FEIERTAGE_CACHE: dict = {}

def _at_feiertage(jahr: int) -> set:
    """
    Österreichische Feiertage (bundesweit).
    Gecacht pro Jahr – wird nur einmal berechnet.

    Feste Feiertage:
      01.01 Neujahr
      06.01 Heilige Drei Könige
      01.05 Staatsfeiertag
      15.08 Mariä Himmelfahrt
      26.10 Nationalfeiertag
      01.11 Allerheiligen
      08.12 Mariä Empfängnis
      25.12 Weihnachten
      26.12 Stephanstag

    Bewegliche Feiertage (berechnet aus Ostersonntag):
      Ostermontag   = Ostern + 1
      Christi Himmelfahrt = Ostern + 39
      Pfingstmontag = Ostern + 50
      Fronleichnam  = Ostern + 60
    """
    if jahr not in _AT_FEIERTAGE_CACHE:
        ostern = _ostersonntag(jahr)
        _AT_FEIERTAGE_CACHE[jahr] = {
            date(jahr,  1,  1),                      # Neujahr
            date(jahr,  1,  6),                      # Heilige Drei Könige
            date(jahr,  5,  1),                      # Staatsfeiertag
            date(jahr,  8, 15),                      # Mariä Himmelfahrt
            date(jahr, 10, 26),                      # Nationalfeiertag
            date(jahr, 11,  1),                      # Allerheiligen
            date(jahr, 12,  8),                      # Mariä Empfängnis
            date(jahr, 12, 25),                      # Weihnachten
            date(jahr, 12, 26),                      # Stephanstag
            ostern + timedelta(days=1),              # Ostermontag
            ostern + timedelta(days=39),             # Christi Himmelfahrt
            ostern + timedelta(days=50),             # Pfingstmontag
            ostern + timedelta(days=60),             # Fronleichnam
        }
    return _AT_FEIERTAGE_CACHE[jahr]


def ist_arbeitstag(d: datetime) -> bool:
    """True wenn d ein österreichischer Bankarbeitstag ist."""
    if d.weekday() >= 5:
        return False
    return date(d.year, d.month, d.day) not in _at_feiertage(d.year)


def naechster_arbeitstag(d: datetime) -> datetime:
    """
    Verschiebt d auf den nächsten Arbeitstag.
    Beispiel: Fällt Zahlung auf Sa 1.2. → verschoben auf Mo 3.2.
    """
    original = d
    while not ist_arbeitstag(d):
        d += timedelta(days=1)
    if d != original:
        _log(f"     Kalender-Shift: {original.strftime('%Y-%m-%d')} "
             f"({original.strftime('%A')}) → {d.strftime('%Y-%m-%d')} "
             f"({d.strftime('%A')})")
    return d


def ist_kalender_verschiebung(d1: datetime, d2: datetime) -> bool:
    """
    Gibt True zurück wenn die Differenz zwischen d1 und d2 durch
    Wochenenden oder Feiertage erklärbar ist (max. 3 Tage).
    Verhindert false-positive Timing-Anomalien.
    """
    diff = abs((d1 - d2).days)
    if diff == 0: return True
    if diff > 3:  return False
    earlier = min(d1, d2)
    for offset in range(diff + 1):
        check = earlier + timedelta(days=offset)
        if not ist_arbeitstag(check):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── BATCH ──────────────────────────────────────────────────────────────────
    # Mehrere Zahlungen gleicher Kategorie am gleichen Tag = ein Batch
    # Beispiel: 5 Gehaltszahlungen am 25. jeden Monats
    "batch_day_tolerance":          2,      # Tage ± (Löhne können sich über 2 Tage ziehen)
    "batch_amount_factor_max":      3.0,    # Max. Faktor zum Median (sonst Zufall)
    "batch_confidence_base":        0.70,
    "batch_confidence_max":         0.95,

    # ── ANOMALIE ───────────────────────────────────────────────────────────────
    # Hybridansatz je nach Stichprobengrösse:
    #   n = 0  → unexpected_new (Gegenpartei wirklich unbekannt)
    #   n 1-2  → insufficient_data (kein Ausschluss aus Recurring!)
    #   n 3-4  → % Abweichung  (zu wenig für Statistik)
    #   n 5-9  → MAD-Score     (robust gegen einzelne Ausreisser)
    #   n ≥ 10 → Z-Score       (klassisch, ausreichend Daten)
    "anomaly_min_ref_txns":         3,      # Minimum für statistische Auswertung
    "anomaly_z_low":                2.0,    # Z-Score Schwelle LOW
    "anomaly_z_medium":             2.5,    # Z-Score Schwelle MEDIUM
    "anomaly_z_high":               3.0,    # Z-Score Schwelle HIGH (fast sicher)
    "anomaly_mad_low":              2.5,    # MAD-Score Schwelle LOW
    "anomaly_mad_medium":           3.5,    # MAD-Score Schwelle MEDIUM
    "anomaly_mad_high":             5.0,    # MAD-Score Schwelle HIGH
    "anomaly_pct_medium":           2.0,    # 200% Abweichung → MEDIUM (n<5)
    "anomaly_pct_high":             4.0,    # 400% Abweichung → HIGH (n<5)
    "anomaly_unexpected_low":       500,    # Wirklich unbekannte Gegenpartei (n=0): ab €500 = LOW
    "anomaly_unexpected_medium":    2000,   # ab €2000 = MEDIUM
    "anomaly_unexpected_high":      10000,  # ab €10000 = HIGH
    "anomaly_inflation_pa":         0.05,   # 5% jährliche Preissteigerung toleriert
                                            # (FIX #5: gilt nur für Ausgaben, nicht Einnahmen)

    # ── RECURRING ──────────────────────────────────────────────────────────────
    # Zeit-Konsistenz: stddev(Abstände) / avg(Abstände) < 0.20
    # Bedeutet: Abstände dürfen max. 20% schwanken
    # Betrags-Konsistenz: stddev(Beträge) / avg(Beträge) < 0.15
    # Bedeutet: Beträge dürfen max. 15% schwanken
    "recurring_min_txns":           3,      # Mindestens 3 Transaktionen nötig
    "recurring_time_consistency":   0.20,   # Max. Zeitabweichung (20%)
    "recurring_time_consistency_min_n": 0.15,  # Strengere Schwelle bei n == min_txns (FIX #6)
    "recurring_amount_tolerance":   0.15,   # Max. Betragsabweichung (15%)
    "recurring_min_confidence":     0.70,   # Mindest-Confidence für is_recurring=True

    # ── SEASONAL ───────────────────────────────────────────────────────────────
    "seasonal_min_years":           2,      # Mindestens 2 Jahreszyklen
    "seasonal_month_tolerance":     1,      # ±1 Monat Toleranz
    "seasonal_min_confidence":      0.65,

    # ── SEQUENTIAL ─────────────────────────────────────────────────────────────
    # Content-Match ist PFLICHT (> 0.40) – verhindert Geister-Sequenzen
    # Beispiel ohne Content-Match: Miete am 1. + Telefon am 15. wären
    # fälschlicherweise als Abfolge erkannt worden.
    #
    # FIX #4: Zusätzliche Bedingung – kein Sequential wenn:
    #   - verschiedene Gegenparteien UND
    #   - gleiche Kategorie UND
    #   - gleiche Zahlungsrichtung (beide Ausgaben oder beide Einnahmen)
    # → Verhindert Gehaltsrunden-Cross-Matching (Anna→Max, Thomas→Max etc.)
    "sequential_min_observations":  2,      # Mindest-Beobachtungen
    "sequential_max_delay_days":    30,     # Max. Tage zwischen A und B
    "sequential_delay_consistency": 0.50,   # Max. Delay-Schwankung
    "sequential_content_min":       0.40,   # PFLICHT: inhaltliche Ähnlichkeit
    "sequential_min_confidence":    0.60,

    # ── GEGENLÄUFIG ────────────────────────────────────────────────────────────
    "counter_min_observations":     3,
    "counter_max_delay_days":       90,
    "counter_delay_consistency":    0.40,
    "counter_correlation_min":      0.60,
    "counter_min_confidence":       0.75,   # Strenger weil unsicherstes Pattern
}

# Sequential bekannte Muster (erhöhen Confidence wenn gefunden)
SEQUENTIAL_SEEDS = [
    {
        "name":             "MWST-Vorauszahlung → Abrechnung",
        "trigger_keywords": ["vorauszahlung", "mwst", "voraus"],
        "follow_keywords":  ["mwst", "abrechnung", "finanzamt", "fa "],
        "trigger_category": "AUSGABEN – STEUERN & ABGABEN",
        "confidence_bonus": 0.10,
    },
    {
        "name":             "Steuer-Rate → Jahresausgleich",
        "trigger_keywords": ["rate", "vorauszahlung", "kvz"],
        "follow_keywords":  ["steuer", "jahresausgleich", "bescheid"],
        "trigger_category": "AUSGABEN – STEUERN & ABGABEN",
        "confidence_bonus": 0.10,
    },
    {
        "name":             "Anzahlung → Restzahlung",
        "trigger_keywords": ["anzahlung", "teilzahlung", "deposit", "akonto"],
        "follow_keywords":  ["restzahlung", "schlusszahlung", "restbetrag", "saldo"],
        "trigger_category": None,           # Kategorieunabhängig
        "confidence_bonus": 0.08,
    },
    {
        "name":             "SV-Vorauszahlung → Nachzahlung",
        "trigger_keywords": ["sv", "sozialversicherung", "voraus"],
        "follow_keywords":  ["sv", "nachzahlung", "differenz"],
        "trigger_category": "AUSGABEN – SOZIALVERSICHERUNGEN",
        "confidence_bonus": 0.09,
    },
]

# Gegenläufige bekannte Paare (Ausgabe → nachfolgende Einnahme)
COUNTER_SEEDS = [
    {
        "name":             "Wareneinkauf → Kundeneinnahme",
        "trigger_category": "AUSGABEN – LIEFERANTEN & WARENEINKAUF",
        "result_category":  "EINNAHMEN",
        "ratio_min":        1.10,   # Einnahme mind. 10% mehr als Ausgabe
        "ratio_max":        2.50,   # Einnahme max. 150% mehr
        "max_delay_days":   60,
    },
    {
        "name":             "Investition → USt-Rückerstattung",
        "trigger_category": "AUSGABEN – INVESTITIONEN",
        "result_category":  "EINNAHMEN",
        "ratio_min":        0.18,   # Österreich: 20% USt
        "ratio_max":        0.22,
        "max_delay_days":   90,
    },
    {
        "name":             "Beratungskosten → Projekteinnahme",
        "trigger_category": "AUSGABEN – BERATUNG & DIENSTLEISTER",
        "result_category":  "EINNAHMEN",
        "ratio_min":        1.20,
        "ratio_max":        5.00,
        "max_delay_days":   90,
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTISCHE HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_key(name: str, category: str, iban: str = None) -> str:
    """
    Erstellt eindeutigen Schlüssel für Transaktions-Gruppierung.

    Priorität (NEU #7 – IBAN-Matching):
      1. IBAN (wenn vorhanden und nicht leer) → zuverlässigste Identifikation
         Vorteil: Namensvarianten ("Müller GmbH" vs "Mueller GmbH") werden
         korrekt zusammengeführt. Keine Normalisierungs-Fehler möglich.
      2. Normalisierter Name + Kategorie (Fallback wenn keine IBAN)

    IBAN-Matching ist optional:
      - Felder "iban", "iban_gegenpartei", "counterparty_iban" werden geprüft
      - Wenn keines vorhanden → Fallback auf Namens-Matching wie bisher
      - Gleiche Gegenpartei mit verschiedenen Kategorien → verschiedene Schlüssel
        (Vermieter mit Kaution ≠ Vermieter mit Miete)

    Beispiel:
      IBAN vorhanden:  "AT61 1904 3002 3457 3201|AUSGABEN – BETRIEBSKOSTEN"
      IBAN fehlt:      "gewerberaum_immobilien_ag|AUSGABEN – BETRIEBSKOSTEN"
    """
    # ── IBAN-Matching wenn verfügbar ──────────────────────────────────────────
    if iban and iban.strip():
        iban_clean = re.sub(r"\s+", "", iban.strip().upper())
        if len(iban_clean) >= 15:   # Mindestlänge für valide IBAN
            return iban_clean + "|" + (category or "UNBEKANNT")

    # ── Fallback: Name-basiertes Matching ────────────────────────────────────
    generic = ["", "bank", "system", "intern", "unbekannt", "unbekannte gegenpartei",
               "eigene buchung", "intern transfer"]
    if not name or name.strip().lower() in generic:
        return "SYSTEM|ZINSEN" if "einnahm" in (category or "").lower() else "SYSTEM|BANK"
    n = name.lower()
    for src, dst in [("ä","ae"),("ö","oe"),("ü","ue"),("ß","ss")]:
        n = n.replace(src, dst)
    n = re.sub(r"[^a-z0-9\s]", "", n)
    return "_".join(n.split()) + "|" + (category or "UNBEKANNT")


def _get_iban(txn: dict) -> Optional[str]:
    """
    Extrahiert IBAN aus Transaktion – prüft mehrere mögliche Feldnamen.
    Rückgabe: IBAN-String oder None wenn nicht vorhanden.
    """
    for field in ["iban", "iban_gegenpartei", "counterparty_iban", "empfaenger_iban"]:
        val = txn.get(field)
        if val and str(val).strip():
            return str(val).strip()
    return None


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _median(v: list) -> float:
    if not v: return 0.0
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 == 1 else (s[m-1] + s[m]) / 2


def _mad(v: list) -> float:
    """Median Absolute Deviation – robust gegen einzelne Ausreisser."""
    m = _median(v)
    return _median([abs(x - m) for x in v])


def _pearson(x: list, y: list) -> float:
    """Pearson-Korrelationskoeffizient."""
    if len(x) < 3: return 0.0
    try:
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        mx, my = sum(x)/n, sum(y)/n
        num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
        den = math.sqrt(sum((v-mx)**2 for v in x) * sum((v-my)**2 for v in y))
        return num / den if den != 0 else 0.0
    except Exception: return 0.0


def _interval_name(avg: float) -> Optional[str]:
    """
    Klassifiziert Tagesabstand in Intervall-Typ.
    Toleranzbereiche sind bewusst grosszügig damit z.B.
    28-Tage-Monat und 31-Tage-Monat beide als 'monthly' erkannt werden.
    """
    if   1  <= avg <=  8:  return "weekly"
    elif 9  <= avg <= 16:  return "biweekly"
    elif 17 <= avg <= 45:  return "monthly"
    elif 46 <= avg <= 75:  return "bimonthly"
    elif 76 <= avg <= 105: return "quarterly"
    elif 106<= avg <= 180: return "halfyear"
    elif 181<= avg <= 400: return "annual"
    return None


def _content_match(za: str, zb: str, ga: str, gb: str) -> float:
    """
    Inhaltliche Ähnlichkeit zwischen zwei Transaktionen.
    PFLICHTBEDINGUNG für Sequential: muss > 0.40 sein.

    Warum nötig: Ohne Content-Match würde z.B.
      Miete am 1. → Telefon am 15.
    als Sequential erkannt, obwohl kein Zusammenhang besteht.

    Scoring:
      1.0 → gleiche Rechnungsnummer (RE-2024-001 in beiden)
      0.8 → gleiche Referenznummer
      0.7 → gleiches Fachkürzel (MWST, AHV, UST, etc.)
      0.4 → gleiche Gegenpartei (schwächeres Signal)
      0.0 → nur zeitliche Nähe → KEIN Sequential
    """
    za, zb = za.lower(), zb.lower()

    # 1.0: Gleiche Rechnungsnummer
    re_pat = r"(?:re|rg|inv)[-\s]?\d{2,6}[-\s]?\d{0,6}"
    ra = re.findall(re_pat, za)
    rb = re.findall(re_pat, zb)
    if ra and rb and any(r in rb for r in ra):
        return 1.0

    # 0.8: Gleiche Referenznummer
    ref_pat = r"ref\.?\s*[\w\d\-]{4,}"
    ra2 = re.findall(ref_pat, za)
    rb2 = re.findall(ref_pat, zb)
    if ra2 and rb2 and any(r in rb2 for r in ra2):
        return 0.8

    # 0.7: Fachliche Schlagworte (Österreich-spezifisch erweitert)
    kws = [
        "mwst", "ust", "umsatzsteuer", "mehrwertsteuer",
        "vorauszahlung", "kvz",                            # Kvz = Körperschaftsteuer-Vorauszahlung
        "finanzamt", "fa ", "estg",
        "sv", "sozialversicherung", "svs", "wgkk",
        "ahv", "bvg", "pensionskasse",
        "miete", "betriebskosten",
        "lohn", "gehalt", "sal",
        "anzahlung", "akonto", "restzahlung",
    ]
    if any(k in za and k in zb for k in kws):
        return 0.7

    # 0.4: Gleiche Gegenpartei
    if ga.lower().strip() == gb.lower().strip() and ga.strip():
        return 0.4

    return 0.0


def _build_cms_cache(transactions: list) -> dict:
    """
    Vorberechnung aller Content-Match-Scores (NEU #8).

    Reduktion von O(n⁴) auf O(n²) für die Sequential-Erkennung.
    Wird einmal vor _run_sequential aufgebaut und dann nur noch gelesen.

    Rückgabe: dict mit (i, j) → float für alle Paare i < j.
    Zugriff:  cache[(i,j)] oder cache[(j,i)] – beide Richtungen gespeichert.
    """
    n = len(transactions)
    cache = {}
    for i in range(n):
        for j in range(i + 1, n):
            score = _content_match(
                transactions[i]["verwendungszweck"],
                transactions[j]["verwendungszweck"],
                transactions[i]["gegenpartei"],
                transactions[j]["gegenpartei"],
            )
            cache[(i, j)] = score
            cache[(j, i)] = score
    return cache


def _next_expected(dates: list, interval: str) -> Optional[str]:
    """
    Berechnet nächstes erwartetes Datum inkl. AT-Kalender.

    Logik bei monthly:
      1. Wenn Tag des Monats konsistent (stddev < 2 Tage):
         → Nächsten Monat, gleicher Tag
      2. Sonst: Letztes Datum + Durchschnittsabstand

    Dann immer: auf nächsten österr. Arbeitstag verschieben.
    """
    if not dates: return None
    last = max(dates)

    if interval == "monthly":
        dom = [d.day for d in dates]
        std = statistics.stdev(dom) if len(dom) > 1 else 0
        if std < 2:
            target = round(statistics.mean(dom))
            nxt = last + relativedelta(months=1)
            try:    nxt = nxt.replace(day=target)
            except ValueError: nxt = nxt + relativedelta(day=31)  # letzter Tag
        else:
            gaps = [(dates[i+1]-dates[i]).days for i in range(len(dates)-1)]
            nxt  = last + timedelta(days=round(statistics.mean(gaps)))
    elif interval == "weekly":    nxt = last + timedelta(weeks=1)
    elif interval == "biweekly":  nxt = last + timedelta(weeks=2)
    elif interval == "quarterly": nxt = last + relativedelta(months=3)
    elif interval == "halfyear":  nxt = last + relativedelta(months=6)
    elif interval == "annual":    nxt = last + relativedelta(years=1)
    else:
        gaps = [(dates[i+1]-dates[i]).days for i in range(len(dates)-1)]
        nxt  = last + timedelta(days=round(statistics.mean(gaps)))

    return naechster_arbeitstag(nxt).strftime("%Y-%m-%d")


def _empty_pattern() -> dict:
    """Leeres Pattern-Template – jede Transaktion bekommt dieses als Basis."""
    return {
        # BATCH
        "is_batch":                       False,
        "batch_id":                       None,
        "batch_size":                     None,
        "batch_total":                    None,
        "batch_confidence":               None,
        "batch_anomaly_type":             None,
        # ANOMALIE
        "is_anomaly":                     False,
        "anomaly_type":                   None,   # neu: "insufficient_data" möglich (is_anomaly=False)
        "anomaly_deviation":              None,
        "anomaly_severity":               None,
        "anomaly_reference_avg":          None,
        "anomaly_score":                  None,
        # RECURRING
        "is_recurring":                   False,
        "recurrence_interval":            None,
        "recurrence_day_of_month":        None,
        "recurrence_day_of_week":         None,
        "recurrence_amount_avg":          None,
        "recurrence_amount_tolerance":    None,
        "recurrence_confidence":          None,
        "recurrence_sample_size":         None,
        "recurrence_has_gaps":            False,
        "next_expected_date":             None,
        # SEASONAL
        "is_seasonal":                    False,
        "seasonal_months":                [],
        "seasonal_week_of_year":          None,
        "seasonal_amount_avg":            None,
        "seasonal_confidence":            None,
        "seasonal_years_observed":        None,
        # SEQUENTIAL
        "is_sequential":                  False,
        "sequential_trigger_txn_id":      None,
        "sequential_trigger_category":    None,
        "sequential_follows_category":    None,
        "sequential_avg_delay_hours":     None,
        "sequential_delay_tolerance_hours": None,
        "sequential_amount_ratio":        None,
        "sequential_confidence":          None,
        "sequential_observations":        None,
        # GEGENLÄUFIG
        "is_counter":                     False,
        "counter_trigger_category":       None,
        "counter_result_category":        None,
        "counter_avg_delay_hours":        None,
        "counter_delay_tolerance_hours":  None,
        "counter_amount_ratio":           None,
        "counter_confidence":             None,
        "counter_observations":           None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1: BATCH-ERKENNUNG
# Mehrere Transaktionen gleicher Kategorie am gleichen Tag (±2 Tage)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_batches(transactions: list) -> dict:
    result = {}
    used   = set()

    for i, t_i in enumerate(transactions):
        if i in used: continue
        cat = t_i.get("category_level1", "")

        # Interne Umbuchungen nie batchen
        if "INTERN" in cat.upper() or "NEUTRAL" in cat.upper(): continue

        d_i   = _parse(t_i["datum"])
        group = [i]

        for j in range(i + 1, len(transactions)):
            if j in used: continue
            t_j = transactions[j]
            if t_j.get("category_level1","") != cat: continue
            if abs((_parse(t_j["datum"]) - d_i).days) <= CFG["batch_day_tolerance"]:
                group.append(j)

        if len(group) < 2: continue

        amounts = [abs(transactions[k]["betrag"]) for k in group]
        med     = _median(amounts)
        if med == 0: continue

        # Beträge die mehr als Faktor 3x abweichen → kein Batch (Zufall)
        valid = [k for k in group
                 if abs(transactions[k]["betrag"]) / med <= CFG["batch_amount_factor_max"]]
        if len(valid) < 2: continue

        date_slug = t_i["datum"].replace("-", "")
        cat_slug  = re.sub(r"[^a-z]", "", cat.lower().split("–")[-1].strip())[:12]
        batch_id  = f"batch_{date_slug}_{cat_slug}"
        total     = sum(transactions[k]["betrag"] for k in valid)
        size      = len(valid)

        conf = CFG["batch_confidence_base"]
        if size >= 3:                                          conf += 0.10
        if abs(round(total) - total) < abs(total) * 0.01:    conf += 0.05
        conf = min(conf, CFG["batch_confidence_max"])

        for k in valid:
            result[k] = {
                "is_batch": True, "batch_id": batch_id,
                "batch_size": size, "batch_total": round(total, 2),
                "batch_confidence": round(conf, 3), "batch_anomaly_type": None,
            }
            used.add(k)

    return result


def _detect_batch_anomalies(transactions: list, batch_results: dict) -> dict:
    """
    Erkennt Anomalien auf BATCH-Ebene.
    Beispiel: Gehaltsrunde mit 4 Mitarbeitern → plötzlich nur 3
              → batch_anomaly_type = 'employee_left'
    """
    by_id   = defaultdict(list)
    updated = dict(batch_results)

    for idx, b in batch_results.items():
        by_id[b["batch_id"]].append((idx, b, transactions[idx]))

    for bid, entries in by_id.items():
        if len(entries) < 2: continue
        totals = [e[1]["batch_total"] for e in entries]
        sizes  = [e[1]["batch_size"]  for e in entries]
        li     = entries[-1][0]

        # Grössen-Anomalie (Mitarbeiterwechsel)
        if len(set(sizes)) > 1:
            prev = round(statistics.mean(sizes[:-1]))
            curr = sizes[-1]
            if curr < prev:
                updated[li]["batch_anomaly_type"] = "employee_left"
                _log(f"  ⚠️  BATCH-ANOMALIE: employee_left "
                     f"(Batch {bid}: vorher {prev}, jetzt {curr} Einträge)")
            elif curr > prev:
                updated[li]["batch_anomaly_type"] = "new_employee"
                _log(f"  ℹ️  BATCH-ANOMALIE: new_employee "
                     f"(Batch {bid}: vorher {prev}, jetzt {curr} Einträge)")

        # Betrags-Anomalie (Bonus oder Kürzung)
        if len(totals) >= 3:
            avg_t = statistics.mean(totals[:-1])
            dev   = (totals[-1] - avg_t) / abs(avg_t) if avg_t != 0 else 0
            if dev < -0.10:
                updated[li]["batch_anomaly_type"] = "salary_reduced"
                _log(f"  ⚠️  BATCH-ANOMALIE: salary_reduced "
                     f"(Batch {bid}: Ø {avg_t:.0f} → {totals[-1]:.0f}, -{abs(dev):.0%})")
            elif dev > 0.20:
                updated[li]["batch_anomaly_type"] = "bonus_payment"
                _log(f"  ℹ️  BATCH-ANOMALIE: bonus_payment "
                     f"(Batch {bid}: Ø {avg_t:.0f} → {totals[-1]:.0f}, +{dev:.0%})")

    return updated


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2: ANOMALIE-ERKENNUNG
# MUSS vor Recurring laufen – echte Anomalien verfälschen sonst den Durchschnitt
# ═══════════════════════════════════════════════════════════════════════════════

def _check_anomaly(txn: dict, ref_txns: list) -> dict:
    """
    Prüft ob txn eine Anomalie gegenüber ref_txns ist.

    Stufenmodell (FIX #1):
      n = 0   → unexpected_new (Gegenpartei wirklich nie zuvor gesehen)
                 is_anomaly=True → wird aus Recurring ausgeschlossen
      n = 1-2 → insufficient_data (Gegenpartei bekannt, aber zu wenig Daten)
                 is_anomaly=False → wird NICHT aus Recurring ausgeschlossen!
                 Wird nur geloggt als Info, nicht als Warnung.
      n >= 3  → Statistischer Vergleich (%, MAD oder Z-Score)
                 is_anomaly=True nur wenn Schwellen überschritten

    FIX #5: Inflationstoleranz gilt nur für Ausgaben.
    FIX #2: anomaly_reference_avg = echter Median, nicht hardcoded 0.0.
    FIX #3: anomaly_deviation = None wenn keine Basis vorhanden.
    """
    p = {"is_anomaly": False, "anomaly_type": None, "anomaly_deviation": None,
         "anomaly_severity": None, "anomaly_reference_avg": None, "anomaly_score": None}

    amount = abs(txn["betrag"])
    refs   = [abs(t["betrag"]) for t in ref_txns]
    n      = len(refs)

    # ── Richtungsänderung: Ausgabe wird plötzlich Einnahme oder umgekehrt ──────
    if ref_txns:
        signs = [1 if t["betrag"] > 0 else -1 for t in ref_txns]
        curr  = 1 if txn["betrag"] > 0 else -1
        dom   = 1 if sum(signs) > 0 else -1
        if curr != dom:
            p.update({"is_anomaly": True, "anomaly_type": "direction_reversal",
                      "anomaly_severity": "medium",
                      "anomaly_reference_avg": round(_median(refs), 2)})
            _log_anomaly(txn, p, refs, None, "Richtungsänderung")
            return p

    # ── n = 0: Wirklich unbekannte Gegenpartei → unexpected_new ───────────────
    # is_anomaly=True → wird aus Recurring ausgeschlossen (korrekt)
    if n == 0:
        sev = None
        if   amount >= CFG["anomaly_unexpected_high"]:   sev = "high"
        elif amount >= CFG["anomaly_unexpected_medium"]: sev = "medium"
        elif amount >= CFG["anomaly_unexpected_low"]:    sev = "low"
        if sev:
            p.update({
                "is_anomaly":            True,
                "anomaly_type":          "unexpected_new",
                "anomaly_severity":      sev,
                "anomaly_reference_avg": 0.0,   # Keine Referenz vorhanden – korrekt 0.0
                "anomaly_deviation":     None,  # FIX #3: keine Basis, kein Prozentwert
            })
            _log_anomaly(txn, p, refs, None, "Unbekannte Gegenpartei (n=0)")
        return p

    # ── n = 1-2: Gegenpartei bekannt, aber zu wenig Daten ─────────────────────
    # insufficient_data: is_anomaly=False → KEIN Ausschluss aus Recurring!
    # Logging passiert in _run_anomaly einmal pro Gruppe, nicht hier.
    if n < CFG["anomaly_min_ref_txns"]:
        return p

    # ── n >= 3: Statistischer Vergleich ────────────────────────────────────────
    ref_med = _median(refs)

    # ── Inflationstoleranz: nur für Ausgaben (FIX #5) ─────────────────────────
    # Einnahmen-Wachstum ist ein Signal, kein Grund zum Ignorieren.
    is_expense = txn["betrag"] < 0
    if is_expense and ref_med > 0 and amount <= ref_med * (1 + CFG["anomaly_inflation_pa"]):
        return p

    sev = None
    score_val = None
    method = ""

    # ── n 3-4: Prozentuale Abweichung ──────────────────────────────────────────
    if n < 5:
        if ref_med == 0: return p
        dev = (amount - ref_med) / ref_med
        score_val = dev
        method = f"%-Abweichung (n={n}, Median={ref_med:.2f})"
        if abs(dev) >= CFG["anomaly_pct_high"]:    sev = "high"
        elif abs(dev) >= CFG["anomaly_pct_medium"]: sev = "medium"

    # ── n 5-9: MAD-Score ────────────────────────────────────────────────────────
    elif n < 10:
        mad_v = _mad(refs)
        if mad_v == 0: return p
        score_val = abs(amount - ref_med) / mad_v
        method = f"MAD-Score (n={n}, Median={ref_med:.2f}, MAD={mad_v:.2f})"
        if   score_val >= CFG["anomaly_mad_high"]:   sev = "high"
        elif score_val >= CFG["anomaly_mad_medium"]: sev = "medium"
        elif score_val >= CFG["anomaly_mad_low"]:    sev = "low"

    # ── n ≥ 10: Z-Score ────────────────────────────────────────────────────────
    else:
        avg_v = statistics.mean(refs)
        std_v = statistics.stdev(refs)
        if std_v == 0: return p
        score_val = (amount - avg_v) / std_v
        method = f"Z-Score (n={n}, Ø={avg_v:.2f}, σ={std_v:.2f})"
        if   abs(score_val) >= CFG["anomaly_z_high"]:   sev = "high"
        elif abs(score_val) >= CFG["anomaly_z_medium"]: sev = "medium"
        elif abs(score_val) >= CFG["anomaly_z_low"]:    sev = "low"

    if sev:
        # FIX #2: anomaly_reference_avg = echter Median (nicht 0.0)
        # FIX #3: anomaly_deviation als Prozentwert nur wenn Basis vorhanden
        dev_pct = (amount - ref_med) / ref_med if ref_med > 0 else None
        p.update({
            "is_anomaly":            True,
            "anomaly_type":          "amount_spike" if amount > ref_med else "amount_drop",
            "anomaly_severity":      sev,
            "anomaly_reference_avg": round(ref_med, 2),   # FIX #2: echter Median
            "anomaly_deviation":     round(dev_pct, 4) if dev_pct is not None else None,
            "anomaly_score":         round(score_val, 4) if score_val is not None else None,
        })
        _log_anomaly(txn, p, refs, score_val, method)

    return p


def _run_anomaly(transactions: list) -> dict:
    """
    Gruppiert Transaktionen nach Gegenpartei+Kategorie (mit IBAN wenn verfügbar)
    und prüft jede auf Anomalie.

    Rückgabe: dict idx → anomaly_fields
    NUR Transaktionen mit is_anomaly=True landen hier.
    insufficient_data (is_anomaly=False) wird EINMAL pro Gruppe geloggt
    (nicht pro Transaktion) und NICHT zurückgegeben → kein Ausschluss aus Recurring.
    """
    groups = defaultdict(list)
    for i, t in enumerate(transactions):
        key = _norm_key(t["gegenpartei"], t.get("category_level1",""), _get_iban(t))
        groups[key].append((i, t))

    result = {}
    for key, entries in groups.items():
        n_group = len(entries)
        logged_insufficient = False  # Pro Gruppe nur einmal loggen

        for pos, (idx, txn) in enumerate(entries):
            ref   = [t for j, (_, t) in enumerate(entries) if j != pos]
            n_ref = len(ref)

            # insufficient_data: einmal pro Gruppe loggen, nicht pro Transaktion
            if 0 < n_ref < CFG["anomaly_min_ref_txns"] and not logged_insufficient:
                _log(f"  ℹ️  INSUFFICIENT_DATA: {txn['gegenpartei']} "
                     f"({n_group} Transaktionen, n={n_ref} Referenzen) "
                     f"→ zu wenig Daten für Statistik, kein Recurring-Ausschluss")
                logged_insufficient = True

            a = _check_anomaly(txn, ref)
            # Nur echte Anomalien (is_anomaly=True) in Ergebnis aufnehmen
            if a["is_anomaly"]:
                result[idx] = a
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: RECURRING-ERKENNUNG
# ═══════════════════════════════════════════════════════════════════════════════

def _run_recurring(transactions: list, anomaly_idx: set) -> dict:
    """
    Erkennt wiederkehrende Zahlungen.

    FIX #1-Effekt: insufficient_data-Transaktionen sind NICHT in anomaly_idx
    → werden korrekt in Recurring-Gruppen einbezogen
    → Sarah Brunner (3×3800), Netzwerk Profis (3×3200) etc. werden jetzt
      als Recurring erkannt statt fälschlich ausgeschlossen.

    FIX #6: Strengere Zeit-Konsistenz-Schwelle bei n == recurring_min_txns.
    """
    groups = defaultdict(list)
    for i, t in enumerate(transactions):
        key = _norm_key(t["gegenpartei"], t.get("category_level1",""), _get_iban(t))
        groups[key].append((i, t))

    result = {}

    for key, entries in groups.items():
        # Echte Anomalien aus der Berechnung ausschliessen
        clean = [(i, t) for i, t in entries if i not in anomaly_idx]
        if len(clean) < CFG["recurring_min_txns"]: continue

        dates   = sorted([_parse(t["datum"]) for _, t in clean])
        amounts = [abs(t["betrag"]) for _, t in clean]
        gaps    = [(dates[k+1]-dates[k]).days for k in range(len(dates)-1)]
        if not gaps: continue

        avg_gap = statistics.mean(gaps)
        std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0
        avg_amt = statistics.mean(amounts)
        std_amt = statistics.stdev(amounts) if len(amounts) > 1 else 0

        # Konsistenz-Check (niedrigerer Wert = konsistenter)
        tc = std_gap / avg_gap if avg_gap > 0 else 1.0   # Zeit-Konsistenz
        ac = std_amt / avg_amt if avg_amt > 0 else 1.0   # Betrags-Konsistenz

        # FIX #6: Strengere Zeit-Konsistenz bei Minimum-Stichprobe
        n = len(clean)
        tc_threshold = (CFG["recurring_time_consistency_min_n"]
                        if n == CFG["recurring_min_txns"]
                        else CFG["recurring_time_consistency"])

        if tc >= tc_threshold:
            continue   # Zeitabstände zu unregelmässig
        if ac >= CFG["recurring_amount_tolerance"]:
            continue   # Beträge zu unterschiedlich

        interval = _interval_name(avg_gap)
        if not interval: continue

        # Lücken prüfen (Monate wo die Zahlung ausblieb)
        has_gaps = any(g > 2 * avg_gap for g in gaps)

        # Confidence-Berechnung
        conf = (1 - tc) * 0.6 + (1 - ac) * 0.4
        if n >= 8:   conf += 0.05   # Bonus für grosse Stichprobe
        if n >= 5:   conf += 0.05
        if std_amt == 0: conf += 0.03   # Bonus für exakt gleiche Beträge
        if n == CFG["recurring_min_txns"]: conf -= 0.10   # Malus für Minimum
        if has_gaps: conf -= 0.15   # Malus für Lücken
        conf = max(0.0, min(1.0, conf))

        if conf < CFG["recurring_min_confidence"]: continue

        # Tag des Monats / Wochentag bestimmen
        dom_vals = [d.day for d in dates]
        dow_vals = [d.weekday() for d in dates]
        dom_std  = statistics.stdev(dom_vals) if len(dom_vals) > 1 else 0
        dow_std  = statistics.stdev(dow_vals) if len(dow_vals) > 1 else 0
        dom = round(statistics.mean(dom_vals)) if dom_std < 2 else None
        dow = round(statistics.mean(dow_vals)) if dow_std < 1 else None

        txn_sample = clean[0][1]
        fields = {
            "is_recurring":                True,
            "recurrence_interval":         interval,
            "recurrence_day_of_month":     dom,
            "recurrence_day_of_week":      dow,
            "recurrence_amount_avg":       round(avg_amt, 2),
            "recurrence_amount_tolerance": round(ac, 4),
            "recurrence_confidence":       round(conf, 4),
            "recurrence_sample_size":      n,
            "recurrence_has_gaps":         has_gaps,
            "next_expected_date":          _next_expected(dates, interval),
        }

        _log_recurring(key, fields, txn_sample, dates, amounts, tc, ac)

        for (i, _) in clean:
            result[i] = fields

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4: SEASONAL-ERKENNUNG
# ═══════════════════════════════════════════════════════════════════════════════

def _run_seasonal(transactions: list, recurring_idx: set) -> dict:
    groups = defaultdict(list)
    for i, t in enumerate(transactions):
        if i in recurring_idx: continue   # Recurring hat Vorrang
        key = _norm_key(t["gegenpartei"], t.get("category_level1",""), _get_iban(t))
        groups[key].append((i, t))

    result = {}

    for key, entries in groups.items():
        if len(entries) < 2: continue
        dates  = [_parse(t["datum"]) for _, t in entries]
        years  = list(set(d.year for d in dates))
        months = [d.month for d in dates]

        if len(years) < CFG["seasonal_min_years"]: continue

        month_std = statistics.stdev(months) if len(months) > 1 else 0
        if month_std > CFG["seasonal_month_tolerance"] * 2: continue

        yc   = len(entries) / (len(years) * max(len(set(months)), 1))
        mc   = 1 - (month_std / 12)
        conf = yc * 0.7 + mc * 0.3
        if len(years) >= 3: conf += 0.10
        if month_std == 0:  conf += 0.05
        if conf < CFG["seasonal_min_confidence"]: continue

        amounts = [abs(t["betrag"]) for _, t in entries]
        weeks   = [_parse(t["datum"]).isocalendar()[1] for _, t in entries]
        fields  = {
            "is_seasonal":           True,
            "seasonal_months":       sorted(set(months)),
            "seasonal_week_of_year": round(statistics.mean(weeks)),
            "seasonal_amount_avg":   round(statistics.mean(amounts), 2),
            "seasonal_confidence":   round(conf, 4),
            "seasonal_years_observed": len(years),
        }
        _log(f"\n  📅 SEASONAL: {entries[0][1]['gegenpartei']} "
             f"| Monate: {sorted(set(months))} "
             f"| Ø {round(statistics.mean(amounts),2)} | conf: {conf:.3f}")

        for (i, _) in entries:
            result[i] = fields

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5: SEQUENTIAL-ERKENNUNG
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sequential(transactions: list) -> dict:
    """
    Erkennt kausale Abfolgen A → B.

    FIX #4: Cross-Entity False Positives verhindert.
    Bedingung zum Überspringen: verschiedene Gegenparteien UND gleiche
    Kategorie UND gleiche Zahlungsrichtung (beide Ausgaben oder beide Einnahmen).
    → Verhindert falsche Ketten bei Gehaltsrunden (Anna→Max, Thomas→Max etc.)
    → Legitime Ketten (AHV Jan→AHV Feb, gleiche Gegenpartei) bleiben erhalten.

    NEU #8: Content-Match-Cache.
    cms_cache wird einmal vorberechnet (O(n²)) statt für jedes Paar
    neu berechnet (wäre O(n⁴) mit den inneren Schleifen).
    """
    result = {}
    n = len(transactions)

    # NEU #8: Cache vorberechnen
    cms_cache = _build_cms_cache(transactions)

    for i in range(n):
        t_a   = transactions[i]
        cat_a = t_a.get("category_level1","")
        d_a   = _parse(t_a["datum"])

        # Systemtransaktionen lösen keine Ketten aus
        key_a = _norm_key(t_a["gegenpartei"], cat_a, _get_iban(t_a))
        if key_a.startswith("SYSTEM|"): continue

        best_match = None
        best_conf  = 0.0

        for j in range(i + 1, n):
            t_b   = transactions[j]
            d_b   = _parse(t_b["datum"])
            delay = (d_b - d_a).days
            if delay < 0 or delay > CFG["sequential_max_delay_days"]: continue

            cat_b = t_b.get("category_level1","")

            # FIX #4: Verhindere Cross-Entity False Positives
            # Beispiel: Anna Schneider (Feb) → Max Müller (Mrz) war falsch,
            # weil "sal" in beiden Verwendungszwecken steht.
            # Bedingung: verschiedene Gegenparteien + gleiche Kategorie + gleiche Richtung
            gp_a = t_a["gegenpartei"].strip().lower()
            gp_b = t_b["gegenpartei"].strip().lower()
            if (gp_a != gp_b
                    and cat_a == cat_b
                    and (t_a["betrag"] < 0) == (t_b["betrag"] < 0)):
                continue

            # Content-Match aus Cache lesen (NEU #8)
            cms = cms_cache.get((i, j), 0.0)

            # PFLICHTBEDINGUNG: ohne inhaltlichen Bezug kein Sequential
            if cms < CFG["sequential_content_min"]: continue

            # Zähle historische Beobachtungen dieser Abfolge
            obs = 1
            delays = [delay]
            for k in range(n):
                if k in (i, j): continue
                cms2 = cms_cache.get((i, k), 0.0)
                if cms2 >= 0.6:
                    d_c = _parse(transactions[k]["datum"])
                    for l in range(k + 1, n):
                        fd = (_parse(transactions[l]["datum"]) - d_c).days
                        if 0 < fd <= CFG["sequential_max_delay_days"]:
                            cms3 = cms_cache.get((j, l), 0.0)
                            if cms3 >= 0.6:
                                obs += 1; delays.append(fd); break

            if obs < CFG["sequential_min_observations"]: continue

            avg_d = statistics.mean(delays)
            std_d = statistics.stdev(delays) if len(delays) > 1 else 0
            dc    = 1 - (std_d / avg_d) if avg_d > 0 else 0
            if dc < (1 - CFG["sequential_delay_consistency"]): continue

            conf = (min(obs / 5, 1.0) * 0.25) + (dc * 0.35) + (cms * 0.40)

            # Seed-Bonus für bekannte Österreich-Muster
            for seed in SEQUENTIAL_SEEDS:
                za_l = t_a["verwendungszweck"].lower()
                zb_l = t_b["verwendungszweck"].lower()
                if (any(k in za_l for k in seed["trigger_keywords"]) and
                    any(k in zb_l for k in seed["follow_keywords"]) and
                    (seed["trigger_category"] is None or seed["trigger_category"] in cat_a)):
                    conf += seed["confidence_bonus"]
                    _log(f"     Seed-Match: {seed['name']} (+{seed['confidence_bonus']})")
                    break

            if obs >= 5: conf += 0.05
            conf = min(conf, 1.0)

            if conf < CFG["sequential_min_confidence"]: continue
            if conf > best_conf:
                best_conf  = conf
                best_match = {
                    "j": j, "avg_d": avg_d, "std_d": std_d, "obs": obs,
                    "conf": conf, "cat_a": cat_a, "cat_b": cat_b, "cms": cms,
                    "ratio": round(abs(t_b["betrag"]) / abs(t_a["betrag"]), 4)
                              if t_a["betrag"] != 0 else None,
                }

        if best_match:
            j = best_match["j"]
            fields = {
                "is_sequential":                    True,
                "sequential_trigger_txn_id":        str(i),
                "sequential_trigger_category":      best_match["cat_a"],
                "sequential_follows_category":      best_match["cat_b"],
                "sequential_avg_delay_hours":       round(best_match["avg_d"] * 24, 1),
                "sequential_delay_tolerance_hours": round(best_match["std_d"] * 24, 1),
                "sequential_amount_ratio":          best_match["ratio"],
                "sequential_confidence":            round(best_match["conf"], 4),
                "sequential_observations":          best_match["obs"],
            }
            _log(f"\n  🔗 SEQUENTIAL")
            _log(f"     Trigger  : Txn {i:>3} | {transactions[i]['datum']} | "
                 f"{transactions[i]['gegenpartei']} | {transactions[i]['verwendungszweck'][:40]}")
            _log(f"     Folge    : Txn {j:>3} | {transactions[j]['datum']} | "
                 f"{transactions[j]['gegenpartei']} | {transactions[j]['verwendungszweck'][:40]}")
            _log(f"     Delay Ø  : {best_match['avg_d']:.1f} Tage")
            _log(f"     Content  : {best_match['cms']:.2f}")
            _log(f"     Confidence: {best_match['conf']:.4f}")
            result[j] = fields

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 6: GEGENLÄUFIG-ERKENNUNG
# ═══════════════════════════════════════════════════════════════════════════════

def _run_counter(transactions: list) -> dict:
    result = {}

    for seed in COUNTER_SEEDS:
        triggers = [(i, t) for i, t in enumerate(transactions)
                    if t.get("category_level1","") == seed["trigger_category"]]
        results_ = [(j, t) for j, t in enumerate(transactions)
                    if t.get("category_level1","") == seed["result_category"]
                    and t["betrag"] > 0]

        if len(triggers) < CFG["counter_min_observations"]: continue

        pairs = []
        for (i, t_a) in triggers:
            d_a = _parse(t_a["datum"])
            for (j, t_b) in results_:
                d_b   = _parse(t_b["datum"])
                delay = (d_b - d_a).days
                if 0 < delay <= seed["max_delay_days"]:
                    ratio = t_b["betrag"] / abs(t_a["betrag"]) if t_a["betrag"] != 0 else 0
                    if seed["ratio_min"] <= ratio <= seed["ratio_max"]:
                        pairs.append({"ti": i, "ri": j, "delay": delay, "ratio": ratio})
                        break

        if len(pairs) < CFG["counter_min_observations"]: continue

        delays = [p["delay"] for p in pairs]
        ratios = [p["ratio"] for p in pairs]
        amt_a  = [abs(transactions[p["ti"]]["betrag"]) for p in pairs]
        amt_b  = [transactions[p["ri"]]["betrag"]      for p in pairs]
        avg_d  = statistics.mean(delays)
        std_d  = statistics.stdev(delays) if len(delays) > 1 else 0
        dc     = 1 - (std_d / avg_d) if avg_d > 0 else 0
        if dc < (1 - CFG["counter_delay_consistency"]): continue

        corr = _pearson(amt_a, amt_b)
        if corr < CFG["counter_correlation_min"]: continue

        conf = (corr * 0.4) + (dc * 0.3) + (min(len(pairs) / 5, 1.0) * 0.3)
        conf += 0.10   # Seed-Bonus
        if len(pairs) >= 5: conf += 0.05
        conf = min(conf, 1.0)
        if conf < CFG["counter_min_confidence"]: continue

        fields = {
            "is_counter":                   True,
            "counter_trigger_category":     seed["trigger_category"],
            "counter_result_category":      seed["result_category"],
            "counter_avg_delay_hours":      round(avg_d * 24, 1),
            "counter_delay_tolerance_hours": round(std_d * 24, 1),
            "counter_amount_ratio":         round(statistics.mean(ratios), 4),
            "counter_confidence":           round(conf, 4),
            "counter_observations":         len(pairs),
        }
        _log(f"\n  🔄 GEGENLÄUFIG: {seed['name']}")
        _log(f"     Paare: {len(pairs)} | Korrelation: {corr:.3f} | Confidence: {conf:.4f}")
        _log(f"     Delay Ø: {avg_d:.0f} Tage | Ratio Ø: {statistics.mean(ratios):.3f}")

        for p in pairs:
            result[p["ri"]] = fields

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HAUPTFUNKTION (wird von pipeline.py aufgerufen)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_patterns(transactions: list) -> list:
    """
    Erkennt alle 6 Pattern-Typen für eine Liste von Transaktionen.

    Eingabe:  Kategorisierte Transaktionen (Output von categorize.py)
    Ausgabe:  Gleiche Liste + 'pattern'-Feld pro Transaktion

    Kein File wird geschrieben – alles bleibt im Memory.
    Kein KI-Call – reine Statistik und Datumsberechnungen.

    IBAN-Matching (NEU #7):
      Wenn Transaktionen ein 'iban', 'iban_gegenpartei', 'counterparty_iban'
      oder 'empfaenger_iban' Feld haben, wird dieses für die Gruppierung
      verwendet. Das ist zuverlässiger als Namens-Matching.
    """
    _log_section("PATTERN DETECTION – Österreich (AT)")
    _log(f"  {len(transactions)} Transaktionen | Kalender: Österreich")

    # IBAN-Verfügbarkeit prüfen und loggen
    txns_mit_iban = sum(1 for t in transactions if _get_iban(t))
    if txns_mit_iban > 0:
        _log(f"  ℹ️  IBAN-Matching aktiv: {txns_mit_iban}/{len(transactions)} "
             f"Transaktionen mit IBAN ({txns_mit_iban/len(transactions):.0%})")
    else:
        _log(f"  ℹ️  IBAN-Matching: keine IBAN-Felder gefunden → Namens-Matching")

    # ── Sicherheitscheck ──────────────────────────────────────────────────────
    unkategorisiert = [t for t in transactions if not t.get("category_level1")]
    if unkategorisiert:
        _log(f"\n  ⚠️  {len(unkategorisiert)} Transaktionen ohne Kategorie – "
             f"Pattern-Genauigkeit reduziert!")

    # ── Schritt 1: BATCH ──────────────────────────────────────────────────────
    _log_subsection("1/6 | BATCH-ERKENNUNG")
    batch_results = _detect_batches(transactions)
    batch_results = _detect_batch_anomalies(transactions, batch_results)
    batch_count   = sum(1 for v in batch_results.values() if v["is_batch"])
    _log(f"  → {batch_count} Transaktionen in "
         f"{len(set(v['batch_id'] for v in batch_results.values()))} Batches")

    # ── Schritt 2: ANOMALIE ───────────────────────────────────────────────────
    _log_subsection("2/6 | ANOMALIE-ERKENNUNG")
    _log("  WICHTIG: Echte Anomalien (is_anomaly=True) werden VOR Recurring erkannt")
    _log("           und aus der Recurring-Berechnung ausgeschlossen.")
    _log("  NEU:     insufficient_data (n=1,2) → is_anomaly=False → kein Ausschluss!")
    anomaly_results = _run_anomaly(transactions)
    anomaly_idx     = set(anomaly_results.keys())   # Nur echte Anomalien
    hi = sum(1 for v in anomaly_results.values() if v["anomaly_severity"] == "high")
    me = sum(1 for v in anomaly_results.values() if v["anomaly_severity"] == "medium")
    lo = sum(1 for v in anomaly_results.values() if v["anomaly_severity"] == "low")
    _log(f"\n  → {len(anomaly_results)} echte Anomalien erkannt  "
         f"(HIGH: {hi} | MEDIUM: {me} | LOW: {lo})")
    if anomaly_idx:
        _log(f"  → Diese Transaktionen werden aus Recurring ausgeschlossen: "
             f"{sorted(anomaly_idx)}")

    # ── Schritt 3: RECURRING ─────────────────────────────────────────────────
    _log_subsection("3/6 | RECURRING-ERKENNUNG")
    _log(f"  Schwellen: Zeit {CFG['recurring_time_consistency']} "
         f"(min_n: {CFG['recurring_time_consistency_min_n']}) | "
         f"Betrag {CFG['recurring_amount_tolerance']} | "
         f"Min-Confidence {CFG['recurring_min_confidence']}")
    recurring_results = _run_recurring(transactions, anomaly_idx)
    recurring_idx     = set(recurring_results.keys())
    if recurring_results:
        from collections import Counter
        ivs = Counter(v["recurrence_interval"] for v in recurring_results.values())
        _log(f"\n  → {len(recurring_results)} Recurring erkannt: "
             f"{dict(ivs)}")
    else:
        _log("  → Keine Recurring-Patterns gefunden")

    # ── Schritt 4: SEASONAL ───────────────────────────────────────────────────
    _log_subsection("4/6 | SEASONAL-ERKENNUNG")
    seasonal_results = _run_seasonal(transactions, recurring_idx)
    _log(f"\n  → {len(seasonal_results)} Seasonal erkannt")

    # ── Schritt 5: SEQUENTIAL ─────────────────────────────────────────────────
    _log_subsection("5/6 | SEQUENTIAL-ERKENNUNG")
    _log(f"  Pflicht: Content-Match > {CFG['sequential_content_min']} "
         f"(verhindert Geister-Sequenzen)")
    _log(f"  FIX #4: Cross-Entity False Positives werden unterdrückt")
    sequential_results = _run_sequential(transactions)
    _log(f"\n  → {len(sequential_results)} Sequential erkannt")

    # ── Schritt 6: GEGENLÄUFIG ────────────────────────────────────────────────
    _log_subsection("6/6 | GEGENLÄUFIG-ERKENNUNG")
    counter_results = _run_counter(transactions)
    _log(f"\n  → {len(counter_results)} Gegenläufig erkannt")

    # ── Zusammenführen ────────────────────────────────────────────────────────
    output = []
    for i, txn in enumerate(transactions):
        p = _empty_pattern()
        for src in [batch_results, anomaly_results, recurring_results,
                    seasonal_results, sequential_results, counter_results]:
            if i in src:
                for k, v in src[i].items():
                    if k in p: p[k] = v
        output.append({**txn, "pattern": p})

    # ── Abschluss-Statistik ───────────────────────────────────────────────────
    no_pat = sum(1 for t in output if not any([
        t["pattern"]["is_batch"],     t["pattern"]["is_recurring"],
        t["pattern"]["is_seasonal"],  t["pattern"]["is_sequential"],
        t["pattern"]["is_counter"],   t["pattern"]["is_anomaly"],
    ]))

    _log_section("ERGEBNIS")
    _log(f"  {'Batch:':<20} {batch_count:>4}  Transaktionen")
    _log(f"  {'Anomalie:':<20} {len(anomaly_results):>4}  "
         f"(HIGH:{hi} MEDIUM:{me} LOW:{lo})")
    _log(f"  {'Recurring:':<20} {len(recurring_results):>4}")
    _log(f"  {'Seasonal:':<20} {len(seasonal_results):>4}")
    _log(f"  {'Sequential:':<20} {len(sequential_results):>4}")
    _log(f"  {'Gegenläufig:':<20} {len(counter_results):>4}")
    _log(f"  {'Kein Pattern:':<20} {no_pat:>4}  → gehen in distributions_db")
    _log(f"  {'─'*35}")
    _log(f"  {'Total:':<20} {len(output):>4}")

    # Recurring-Übersicht (kompakt)
    if recurring_results:
        _log_subsection("Recurring-Übersicht")
        seen = set()
        for i, f in recurring_results.items():
            k = _norm_key(transactions[i]["gegenpartei"],
                          transactions[i].get("category_level1",""),
                          _get_iban(transactions[i]))
            if k in seen: continue
            seen.add(k)
            _log(f"  {transactions[i]['gegenpartei'][:35]:<35} "
                 f"{f['recurrence_interval']:<12} "
                 f"Ø {f['recurrence_amount_avg']:>10.2f}  "
                 f"→ {f['next_expected_date']}  "
                 f"(conf: {f['recurrence_confidence']:.3f})")

    return output   # ← gibt Liste zurück, schreibt NICHTS auf Disk


# ═══════════════════════════════════════════════════════════════════════════════
# DIREKTTEST (python detect_patterns.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os

    # ── Eingabedatei bestimmen ────────────────────────────────────────────────
    # Aufruf-Varianten:
    #   python detect_patterns.py                          → Fallback: tink_categorized.json
    #   python detect_patterns.py meine_daten.json        → beliebige kategorisierte JSON
    #   python detect_patterns.py /pfad/zur/datei.json    → absoluter Pfad
    #
    # Format der JSON: Liste von Transaktionen mit mindestens:
    #   datum, betrag, verwendungszweck, gegenpartei
    # Optional (verbessert Pattern-Qualität):
    #   category_level1, iban / iban_gegenpartei / counterparty_iban
    DEFAULT_INPUT = "tink_categorized.json"
    INPUT_FILE    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

    # ── Startbanner ───────────────────────────────────────────────────────────
    _log("=" * 70)
    _log("  detect_patterns.py – Direkttest v2.1")
    _log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("  Kalender: Österreich (AT)")
    _log("  Kein KI-Call. Reine Statistik + Datumsberechnungen.")
    _log("  Fixes: #1 insufficient_data | #2/#3 Referenz-Log | #4 Sequential")
    _log("         #5 Inflation nur Ausgaben | #6 Min-n strenger | #7 IBAN | #8 Cache")
    _log(f"  Eingabe : {INPUT_FILE}")
    _log("=" * 70)

    # ── Kalender-Selbsttest ───────────────────────────────────────────────────
    _log_subsection("KALENDER-SELBSTTEST (AT)")

    test_jahre = [2024, 2025, 2026]
    for jahr in test_jahre:
        ostern = _ostersonntag(jahr)
        feiertage = _at_feiertage(jahr)
        feiertage_sorted = sorted(feiertage)
        _log(f"\n  Jahr {jahr}:")
        _log(f"    Ostersonntag : {ostern.strftime('%d.%m.%Y')}")
        _log(f"    Feiertage    : {len(feiertage)} Tage")
        for ft in feiertage_sorted:
            wochentag = ["Mo","Di","Mi","Do","Fr","Sa","So"][ft.weekday()]
            _log(f"      {ft.strftime('%d.%m.%Y')} ({wochentag})")

    _log(f"\n  Arbeitstag-Test:")
    test_dates = [
        ("2025-01-01", False, "Neujahr AT"),
        ("2025-01-06", False, "Heilige Drei Könige"),
        ("2025-04-18", False, "Karfreitag 2025"),
        ("2025-04-21", False, "Ostermontag 2025"),
        ("2025-05-01", False, "Staatsfeiertag"),
        ("2025-05-29", False, "Christi Himmelfahrt 2025"),
        ("2025-06-09", False, "Pfingstmontag 2025"),
        ("2025-06-19", False, "Fronleichnam 2025"),
        ("2025-10-26", False, "Nationalfeiertag"),
        ("2025-12-08", False, "Mariä Empfängnis"),
        ("2025-01-02", True,  "Normaler Donnerstag"),
        ("2025-03-17", True,  "Normaler Montag"),
        ("2025-06-14", False, "Samstag"),
        ("2025-06-15", False, "Sonntag"),
    ]
    alle_ok = True
    for datum_str, erwartet, beschreibung in test_dates:
        d = datetime.strptime(datum_str, "%Y-%m-%d")
        ergebnis = ist_arbeitstag(d)
        status   = "✅" if ergebnis == erwartet else "❌"
        if ergebnis != erwartet:
            alle_ok = False
        _log(f"    {status} {datum_str} ({beschreibung}): "
             f"{'Arbeitstag' if ergebnis else 'Kein Arbeitstag'}")

    if alle_ok:
        _log("\n  ✅ Kalender-Selbsttest bestanden – alle Feiertage korrekt")
    else:
        _log("\n  ❌ FEHLER im Kalender-Selbsttest – bitte prüfen!")
        sys.exit(1)

    _log(f"\n  Arbeitstag-Verschiebungs-Test:")
    shift_tests = [
        ("2025-01-04", "2025-01-06", "Samstag → Montag"),
        ("2025-04-19", "2025-04-22", "Samstag vor Ostermontag → Dienstag"),
        ("2025-05-01", "2025-05-02", "Staatsfeiertag Do → Freitag"),
        ("2025-12-26", "2025-12-29", "Stephanstag Fr → Montag"),
    ]
    for von_str, erwartet_str, beschreibung in shift_tests:
        von      = datetime.strptime(von_str, "%Y-%m-%d")
        erwartet = datetime.strptime(erwartet_str, "%Y-%m-%d")
        ergebnis = naechster_arbeitstag(von)
        ok = "✅" if ergebnis.date() == erwartet.date() else "❌"
        _log(f"    {ok} {von_str} ({beschreibung}) → {ergebnis.strftime('%Y-%m-%d')}"
             f"  (erwartet: {erwartet_str})")

    # ── IBAN-Matching Selbsttest ──────────────────────────────────────────────
    _log_subsection("IBAN-MATCHING SELBSTTEST (NEU #7)")
    iban_tests = [
        # (name, category, iban, erwarteter_key_prefix)
        ("Müller GmbH",     "AUSGABEN – BETRIEBSKOSTEN", "AT61 1904 3002 3457 3201",
         "AT611904300234573201|"),
        ("Mueller GmbH",    "AUSGABEN – BETRIEBSKOSTEN", "AT61 1904 3002 3457 3201",
         "AT611904300234573201|"),   # gleicher Key trotz Namensvariant!
        ("Müller GmbH",     "AUSGABEN – BETRIEBSKOSTEN", None,
         "muller_gmbh|"),            # Fallback ohne IBAN
        ("Müller GmbH",     "AUSGABEN – BETRIEBSKOSTEN", "",
         "muller_gmbh|"),            # Leere IBAN → Fallback
        ("unbekannt",       "EINNAHMEN",                 None,
         "SYSTEM|"),                 # Generischer Name
    ]
    iban_ok = True
    for name, cat, iban, expected_prefix in iban_tests:
        key = _norm_key(name, cat, iban)
        ok  = "✅" if key.startswith(expected_prefix) else "❌"
        if not key.startswith(expected_prefix): iban_ok = False
        _log(f"  {ok}  name={name!r:20}  iban={str(iban)!r:30}  → {key}")
    _log(f"\n  {'✅ IBAN-Matching korrekt' if iban_ok else '❌ IBAN-Matching fehlerhaft!'}")

    # ── Anomalie-Fix Selbsttest ───────────────────────────────────────────────
    _log_subsection("ANOMALIE-FIX SELBSTTEST (FIX #1, #2, #3, #5)")

    # Test FIX #1: n=0 → unexpected_new, n=1 → kein is_anomaly, n=2 → kein is_anomaly
    def _make_txn(betrag, datum="2025-01-01", gp="TestGP"):
        return {"betrag": betrag, "datum": datum, "gegenpartei": gp,
                "verwendungszweck": "Test", "category_level1": "AUSGABEN – BETRIEBSKOSTEN"}

    t_neu = _make_txn(-5000)
    a0 = _check_anomaly(t_neu, [])                # n=0 → unexpected_new
    a1 = _check_anomaly(t_neu, [_make_txn(-5000)])  # n=1 → insufficient_data
    a2 = _check_anomaly(t_neu, [_make_txn(-5000), _make_txn(-5000)])  # n=2 → insufficient_data
    _log(f"  n=0: is_anomaly={a0['is_anomaly']} type={a0['anomaly_type']}  "
         f"(erwartet: True / unexpected_new)  {'✅' if a0['is_anomaly'] and a0['anomaly_type']=='unexpected_new' else '❌'}")
    _log(f"  n=1: is_anomaly={a1['is_anomaly']} type={a1['anomaly_type']}  "
         f"(erwartet: False / None)  {'✅' if not a1['is_anomaly'] else '❌'}")
    _log(f"  n=2: is_anomaly={a2['is_anomaly']} type={a2['anomaly_type']}  "
         f"(erwartet: False / None)  {'✅' if not a2['is_anomaly'] else '❌'}")

    # Test FIX #2: reference_avg korrekt
    t_spike = _make_txn(-9000)
    refs3 = [_make_txn(-1000), _make_txn(-1100), _make_txn(-900)]
    a3 = _check_anomaly(t_spike, refs3)
    _log(f"  FIX #2: reference_avg={a3['anomaly_reference_avg']} "
         f"(erwartet: ~1000, nicht 0.0)  "
         f"{'✅' if a3.get('anomaly_reference_avg', 0) > 0 else '❌'}")

    # Test FIX #5: Inflation-Toleranz nur für Ausgaben
    t_einnahme = _make_txn(+1050)   # +5% Einnahmen-Wachstum → SOLL gemeldet werden
    t_ausgabe  = _make_txn(-1050)   # +5% Ausgaben-Wachstum → darf ignoriert werden
    refs_1000 = [_make_txn(1000), _make_txn(1000), _make_txn(1000),
                 _make_txn(1000), _make_txn(1000)]
    ae = _check_anomaly(t_einnahme, refs_1000)
    aa = _check_anomaly(t_ausgabe,  refs_1000)
    # Bei Einnahmen: +5% über Median sollte nicht durch Inflationstoleranz ignoriert werden
    _log(f"  FIX #5 Einnahmen: is_anomaly={ae['is_anomaly']}  "
         f"(+5% Einnahmen, erwartet: False da zu klein für Z-Score)  ℹ️")
    _log(f"  FIX #5 Ausgaben:  is_anomaly={aa['is_anomaly']}  "
         f"(+5% Ausgaben, erwartet: False wegen Inflation-Toleranz)  "
         f"{'✅' if not aa['is_anomaly'] else '❌'}")

    # ── Datei laden ───────────────────────────────────────────────────────────
    _log_section("DATEN LADEN")

    if not os.path.exists(INPUT_FILE):
        _log(f"  ❌ Datei nicht gefunden: {INPUT_FILE}")
        _log(f"     Bitte sicherstellen dass {INPUT_FILE} im gleichen Ordner liegt.")
        _log(f"     Tipp: Zuerst categorize.py ausführen.")
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        transactions = json.load(f)

    _log(f"  ✅ {len(transactions)} Transaktionen geladen aus: {INPUT_FILE}")

    # Datei-Qualitätsprüfung
    _log(f"\n  Qualitätsprüfung der Eingabedaten:")

    pflicht_felder = ["datum", "betrag", "verwendungszweck", "gegenpartei"]
    fehler_felder  = []
    for i, t in enumerate(transactions):
        fehlend = [f for f in pflicht_felder if f not in t]
        if fehlend:
            fehler_felder.append((i, fehlend))

    if fehler_felder:
        _log(f"  ⚠️  {len(fehler_felder)} Transaktionen mit fehlenden Pflichtfeldern:")
        for idx, fehlend in fehler_felder[:5]:
            _log(f"     Txn {idx}: fehlt {fehlend}")
        if len(fehler_felder) > 5:
            _log(f"     ... und {len(fehler_felder)-5} weitere")
    else:
        _log(f"  ✅ Alle Pflichtfelder vorhanden")

    ohne_kat = [t for t in transactions if not t.get("category_level1")]
    if ohne_kat:
        _log(f"  ⚠️  {len(ohne_kat)} Transaktionen ohne category_level1")
        _log(f"     → Erst categorize.py ausführen für beste Ergebnisse")
    else:
        _log(f"  ✅ Alle Transaktionen kategorisiert")

    # IBAN-Verfügbarkeit in Datei
    txns_mit_iban = sum(1 for t in transactions if _get_iban(t))
    _log(f"  {'✅' if txns_mit_iban > 0 else 'ℹ️ '} IBAN-Felder: "
         f"{txns_mit_iban}/{len(transactions)} Transaktionen "
         f"({'IBAN-Matching aktiv' if txns_mit_iban > 0 else 'Namens-Matching Fallback'})")

    # Datums-Übersicht
    daten = sorted([t["datum"] for t in transactions])
    betraege = [t["betrag"] for t in transactions]
    einnahmen = sum(b for b in betraege if b > 0)
    ausgaben  = sum(b for b in betraege if b < 0)

    _log(f"\n  Zeitraum      : {daten[0]} → {daten[-1]}")
    _log(f"  Einnahmen     : +{einnahmen:>12,.2f} €")
    _log(f"  Ausgaben      :  {ausgaben:>12,.2f} €")
    _log(f"  Netto         :  {einnahmen + ausgaben:>12,.2f} €")

    # Kategorien-Übersicht
    _log(f"\n  Kategorien-Verteilung:")
    from collections import Counter
    kat_counts = Counter(t.get("category_level1", "KEINE") for t in transactions)
    for kat, anzahl in sorted(kat_counts.items(), key=lambda x: -x[1]):
        balken = "█" * min(anzahl, 30)
        _log(f"    {anzahl:>3}x  {kat[:45]:<45}  {balken}")

    # ── Pattern Detection ausführen ───────────────────────────────────────────
    result = detect_patterns(transactions)

    # ── Detaillierte Ergebnis-Analyse ─────────────────────────────────────────
    _log_section("DETAILLIERTE ERGEBNIS-ANALYSE")

    # Recurring im Detail
    recurring_txns = [t for t in result if t["pattern"]["is_recurring"]]
    if recurring_txns:
        _log_subsection("ALLE RECURRING PATTERNS – Details")
        seen_keys = set()
        for t in recurring_txns:
            p   = t["pattern"]
            key = _norm_key(t["gegenpartei"], t.get("category_level1",""), _get_iban(t))
            if key in seen_keys: continue
            seen_keys.add(key)

            gruppe = [x for x in recurring_txns
                      if _norm_key(x["gegenpartei"], x.get("category_level1",""),
                                   _get_iban(x)) == key]

            _log(f"\n  ┌─ {t['gegenpartei']}")
            if _get_iban(t):
                _log(f"  │  IBAN           : {_get_iban(t)}")
            _log(f"  │  Kategorie     : {t.get('category_level1','n/a')}")
            _log(f"  │  Intervall     : {p['recurrence_interval']}")
            _log(f"  │  Stichprobe    : {p['recurrence_sample_size']} Transaktionen")
            _log(f"  │  Betrag-Ø      : {p['recurrence_amount_avg']:>10.2f} €")
            _log(f"  │  Betrag-Tol.   : ±{p['recurrence_amount_tolerance']*100:.1f}%")
            _log(f"  │  Confidence    : {p['recurrence_confidence']:.4f}")
            _log(f"  │  Nächstes      : {p['next_expected_date']}")
            if p.get('recurrence_day_of_month'):
                _log(f"  │  Zahltag       : {p['recurrence_day_of_month']}. des Monats")
            if p.get('recurrence_has_gaps'):
                _log(f"  │  ⚠️  Lücken     : Ja (nicht jeden Monat gebucht)")
            _log(f"  │  Alle Buchungen:")
            for g in sorted(gruppe, key=lambda x: x["datum"]):
                _log(f"  │    {g['datum']}  {g['betrag']:>10.2f} €  "
                     f"{g['verwendungszweck'][:40]}")
            _log(f"  └─")

    # Anomalien im Detail
    anomaly_txns = [t for t in result if t["pattern"]["is_anomaly"]]
    if anomaly_txns:
        _log_subsection("ALLE ANOMALIEN – Nach Schweregrad")
        for sev in ["high", "medium", "low"]:
            gruppe = [t for t in anomaly_txns if t["pattern"]["anomaly_severity"] == sev]
            if not gruppe: continue
            _log(f"\n  [{sev.upper()}] – {len(gruppe)} Anomalie(n):")
            for t in gruppe:
                p = t["pattern"]
                _log(f"    {t['datum']}  {t['betrag']:>10.2f} €  "
                     f"{t['gegenpartei'][:30]}")
                _log(f"    Typ: {p['anomaly_type']}  |  "
                     f"Referenz-Ø: {p['anomaly_reference_avg']}  |  "
                     f"Score: {p['anomaly_score']}")
                if p.get('anomaly_deviation') is not None:
                    _log(f"    Abweichung: {p['anomaly_deviation']:+.1%}")
                _log(f"    Zweck: {t['verwendungszweck']}")
                _log("")

    # Batch im Detail
    batch_txns = [t for t in result if t["pattern"]["is_batch"]]
    if batch_txns:
        _log_subsection("ALLE BATCHES – Details")
        seen_batches = set()
        for t in batch_txns:
            p  = t["pattern"]
            bid = p["batch_id"]
            if bid in seen_batches: continue
            seen_batches.add(bid)
            gruppe = [x for x in batch_txns if x["pattern"]["batch_id"] == bid]
            _log(f"\n  Batch: {bid}")
            _log(f"    Grösse     : {p['batch_size']} Transaktionen")
            _log(f"    Summe      : {p['batch_total']:>10.2f} €")
            _log(f"    Confidence : {p['batch_confidence']:.3f}")
            if p.get("batch_anomaly_type"):
                _log(f"    ⚠️  Anomalie : {p['batch_anomaly_type']}")
            for g in sorted(gruppe, key=lambda x: x["datum"]):
                _log(f"      {g['datum']}  {g['betrag']:>10.2f} €  {g['gegenpartei']}")

    # Sequential im Detail
    seq_txns = [t for t in result if t["pattern"]["is_sequential"]]
    if seq_txns:
        _log_subsection("ALLE SEQUENTIAL PATTERNS – Details")
        for t in seq_txns:
            p = t["pattern"]
            trigger_id = int(p["sequential_trigger_txn_id"])
            trigger    = transactions[trigger_id]
            _log(f"\n  Kette:")
            _log(f"    TRIGGER  [{trigger_id:>3}] : {trigger['datum']}  "
                 f"{trigger['betrag']:>10.2f} €  {trigger['gegenpartei']}")
            _log(f"             Zweck: {trigger['verwendungszweck'][:55]}")
            _log(f"    FOLGE    [{result.index(t):>3}] : {t['datum']}  "
                 f"{t['betrag']:>10.2f} €  {t['gegenpartei']}")
            _log(f"             Zweck: {t['verwendungszweck'][:55]}")
            _log(f"    Delay Ø  : {p['sequential_avg_delay_hours']/24:.1f} Tage")
            _log(f"    Ratio    : {p['sequential_amount_ratio']}")
            _log(f"    Beob.    : {p['sequential_observations']}")
            _log(f"    Conf.    : {p['sequential_confidence']:.4f}")

    # Gegenläufig im Detail
    counter_txns = [t for t in result if t["pattern"]["is_counter"]]
    if counter_txns:
        _log_subsection("ALLE GEGENLÄUFIGEN PATTERNS – Details")
        for t in counter_txns:
            p = t["pattern"]
            _log(f"\n  {t['datum']}  {t['betrag']:>10.2f} €  {t['gegenpartei']}")
            _log(f"    Auslöser-Kat : {p['counter_trigger_category']}")
            _log(f"    Folge-Kat    : {p['counter_result_category']}")
            _log(f"    Delay Ø      : {p['counter_avg_delay_hours']/24:.0f} Tage")
            _log(f"    Ratio Ø      : {p['counter_amount_ratio']:.3f}")
            _log(f"    Confidence   : {p['counter_confidence']:.4f}")

    # ── Transaktionen ohne Pattern ────────────────────────────────────────────
    no_pat_txns = [t for t in result if not any([
        t["pattern"]["is_batch"],     t["pattern"]["is_recurring"],
        t["pattern"]["is_seasonal"],  t["pattern"]["is_sequential"],
        t["pattern"]["is_counter"],   t["pattern"]["is_anomaly"],
    ])]

    _log_subsection("TRANSAKTIONEN OHNE PATTERN → gehen in distributions_db")
    _log(f"  Anzahl: {len(no_pat_txns)}")
    if no_pat_txns:
        _log(f"\n  {'Datum':<12} {'Betrag':>10}  {'Gegenpartei':<30}  Kategorie")
        _log(f"  {'─'*12} {'─'*10}  {'─'*30}  {'─'*25}")
        for t in no_pat_txns:
            _log(f"  {t['datum']:<12} {t['betrag']:>10.2f} €  "
                 f"{t['gegenpartei'][:30]:<30}  "
                 f"{(t.get('category_level1') or 'n/a')[:30]}")

    # ── Forecast-Vorschau ─────────────────────────────────────────────────────
    _log_section("FORECAST-VORSCHAU (nächste 30 Tage)")

    from datetime import date as date_type
    heute     = datetime.now().date()
    in_30     = heute + timedelta(days=30)
    vorschau  = []

    for t in result:
        p = t["pattern"]
        ned = p.get("next_expected_date")
        if not ned: continue
        ned_date = datetime.strptime(ned, "%Y-%m-%d").date()
        if heute <= ned_date <= in_30:
            vorschau.append({
                "datum":       ned,
                "gegenpartei": t["gegenpartei"],
                "betrag_avg":  p.get("recurrence_amount_avg") or p.get("seasonal_amount_avg"),
                "intervall":   p.get("recurrence_interval",""),
                "confidence":  p.get("recurrence_confidence") or p.get("seasonal_confidence"),
                "kategorie":   t.get("category_level1",""),
            })

    # Duplikate entfernen (mehrere Txns gleicher Gruppe)
    seen_fc = set()
    vorschau_dedup = []
    for v in sorted(vorschau, key=lambda x: x["datum"]):
        key = v["gegenpartei"] + v["datum"]
        if key not in seen_fc:
            seen_fc.add(key)
            vorschau_dedup.append(v)

    if vorschau_dedup:
        _log(f"\n  {'Datum':<12}  {'Betrag-Ø':>10}  "
             f"{'Gegenpartei':<30}  {'Intervall':<12}  Conf.")
        _log(f"  {'─'*12}  {'─'*10}  {'─'*30}  {'─'*12}  {'─'*6}")
        kumuliert = 0.0
        for v in vorschau_dedup:
            betrag = v["betrag_avg"] or 0
            kumuliert += betrag
            richtung = "📈" if betrag > 0 else "📉"
            _log(f"  {v['datum']:<12}  {betrag:>10.2f} €  "
                 f"{v['gegenpartei'][:30]:<30}  "
                 f"{v['intervall']:<12}  {v['confidence']:.3f}  {richtung}")
        _log(f"\n  Kumuliert (nächste 30 Tage): {kumuliert:>10.2f} €")
    else:
        _log(f"\n  Keine Recurring-Zahlungen in den nächsten 30 Tagen erwartet.")
        _log(f"  (Möglicherweise zu wenig historische Daten für Recurring-Erkennung)")

    # ── Abschluss ─────────────────────────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("  ✅ detect_patterns.py v2.1 – Direkttest abgeschlossen")
    _log(f"  Verarbeitet : {len(transactions)} Transaktionen")
    _log(f"  Mit Pattern : {len(transactions) - len(no_pat_txns)}")
    _log(f"  Ohne Pattern: {len(no_pat_txns)}")
    _log(f"  Abgeschlossen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("  Nächster Schritt: organisational.py")
    _log("=" * 70)
