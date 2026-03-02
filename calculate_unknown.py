"""
calculate_unknown.py
====================
Berechnet Wahrscheinlichkeitsverteilungen für Transaktionen ohne Muster
(Quelle: Firestore-Sammlung distributions_db) und speichert die Ergebnisse
in der Sammlung forecast_distribution für den Frontend-Zugriff.

Statistisches Modell (verbessert):
  ┌─────────────────────────────────────────────────────────────────┐
  │  HÄUFIGKEIT    Poisson-Prozess mit DOM-Verteilung               │
  │                λ/Monat + P(Buchung an Tag d) → tägl. Prognose  │
  │                                                                 │
  │  BETRÄGE       Log-Normal (besser als Normal für Finanzdaten:   │
  │                immer positiv, rechtsschiefe Verteilung)         │
  │                Fallback auf Normal bei wenig Daten              │
  │                                                                 │
  │  RICHTUNG      INFLOW / OUTFLOW getrennt behandelt              │
  │                → Netto-Cashflow-Prognose                        │
  │                                                                 │
  │  ZEITEBENE     Tagesgenaue 180-Tage-Prognose                   │
  │                Typische Buchungszeit aus bookedDateTime          │
  │                Kumulierter Cashflow + Liquiditätsrisikoerkennung │
  └─────────────────────────────────────────────────────────────────┘

Zeitfenster:
  Referenzdatum = letztes Transaktionsdatum in distributions_db
  Cutoff        = Referenzdatum - LOOKBACK_MONTHS

Konfiguration via Env-Variablen:
  GOOGLE_APPLICATION_CREDENTIALS  Pfad zur Firebase Service Account JSON

PUBLIC API (für pipeline.py):
    from calculate_unknown import calculate_unknown
    result = calculate_unknown()
    # result = {"groups_written": int, "transactions_analyzed": int}

CLI:
    python calculate_unknown.py
"""

import math
import os
import re
import sys
from calendar import monthrange
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta, timezone
from statistics import mean, stdev
from typing import Optional

# ─────────────────────────────────────────────
# KONFIGURATION  (Env-Variablen überschreiben Defaults)
# ─────────────────────────────────────────────

LOOKBACK_MONTHS       = 6
FORECAST_HORIZON_DAYS = 180
MIN_SAMPLE_SINGLE     = 2
MIN_SAMPLE_CATEGORY   = 1

CI_90_Z = 1.645
CI_95_Z = 1.960

RECENCY_DECAY = 0.80

SOURCE_COLLECTION = "distributions_db"
TARGET_COLLECTION = "forecast_distribution"


# ─────────────────────────────────────────────
# FIRESTORE SETUP
# ─────────────────────────────────────────────

def _init_firestore():
    """
    Initialisiert Firebase Admin SDK.
    Liest Credentials aus GOOGLE_APPLICATION_CREDENTIALS (Env-Variable).
    Idempotent bei Mehrfachaufruf.

    Lokal  : export GOOGLE_APPLICATION_CREDENTIALS=/pfad/zu/serviceaccount.json
    GitHub : wird automatisch von calculate_unknown.yml gesetzt
    """
    import firebase_admin
    from firebase_admin import credentials, firestore as fs

    if not firebase_admin._apps:
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

        if not creds_path:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS nicht gesetzt.\n"
                "  Lokal  : export GOOGLE_APPLICATION_CREDENTIALS=/pfad/zu/serviceaccount.json\n"
                "  GitHub : Secret FIREBASE_SERVICE_ACCOUNT wird automatisch gesetzt."
            )
        if not os.path.exists(creds_path):
            raise RuntimeError(
                f"Service-Account-Datei nicht gefunden: {creds_path}"
            )

        cred = credentials.Certificate(creds_path)
        firebase_admin.initialize_app(cred)

    return fs.client()


# ─────────────────────────────────────────────
# DATEN LADEN & VORBEREITEN
# ─────────────────────────────────────────────

def _parse_datum(datum_str: str) -> tuple[date, Optional[int], Optional[int]]:
    s = datum_str.strip()
    if "T" in s:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.date(), dt.hour, dt.minute
        except ValueError:
            pass
    try:
        return date.fromisoformat(s[:10]), None, None
    except ValueError:
        return date.today(), None, None


def _load_all_distributions(db) -> list[dict]:
    docs = db.collection(SOURCE_COLLECTION).stream()
    transactions = []
    for doc in docs:
        data = doc.to_dict()
        datum_str = data.get("datum", "")
        if not datum_str:
            continue
        d, hour, minute = _parse_datum(datum_str)
        data["_date_obj"] = d
        data["_hour"]     = hour
        data["_minute"]   = minute
        transactions.append(data)
    return transactions


def _determine_window(transactions: list[dict]) -> tuple[date, date, int]:
    if not transactions:
        today = date.today()
        m = today.month - LOOKBACK_MONTHS
        y = today.year
        while m <= 0:
            m += 12; y -= 1
        return date(y, m, 1), today, LOOKBACK_MONTHS

    ref = max(t["_date_obj"] for t in transactions)
    m = ref.month - LOOKBACK_MONTHS
    y = ref.year
    while m <= 0:
        m += 12; y -= 1
    return date(y, m, 1), ref, LOOKBACK_MONTHS


def _filter_by_window(transactions: list[dict], cutoff: date, ref: date) -> list[dict]:
    return [t for t in transactions if cutoff <= t["_date_obj"] <= ref]


# ─────────────────────────────────────────────
# GRUPPIERUNG
# ─────────────────────────────────────────────

def _build_groups(transactions: list[dict]) -> dict[str, list[dict]]:
    primary: defaultdict[str, list] = defaultdict(list)
    for t in transactions:
        cat = t.get("category_level1", "UNBEKANNT")
        geg = (t.get("gegenpartei") or "").strip() or "Unbekannt"
        primary[f"{cat} :: {geg}"].append(t)

    category: defaultdict[str, list] = defaultdict(list)
    final: dict[str, list] = {}

    for key, txs in primary.items():
        if len(txs) >= MIN_SAMPLE_SINGLE:
            final[key] = txs
        else:
            category[key.split(" :: ")[0]].extend(txs)

    for cat, txs in category.items():
        if len(txs) >= MIN_SAMPLE_CATEGORY and cat not in final:
            final[cat] = txs

    return final


# ─────────────────────────────────────────────
# ZEITANALYSE
# ─────────────────────────────────────────────

def _circular_mean_time(hours: list[int], minutes: list[int]) -> Optional[str]:
    if not hours:
        return None
    total_minutes = [h * 60 + m for h, m in zip(hours, minutes)]
    angles = [t / (24 * 60) * 2 * math.pi for t in total_minutes]
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    mean_angle = math.atan2(sin_sum, cos_sum) % (2 * math.pi)
    mean_minutes = round(mean_angle / (2 * math.pi) * 24 * 60)
    h = (mean_minutes // 60) % 24
    m = mean_minutes % 60
    return f"{h:02d}:{m:02d}"


def _extract_times(transactions: list[dict]) -> Optional[str]:
    valid = [(t["_hour"], t["_minute"])
             for t in transactions
             if t["_hour"] is not None and t["_minute"] is not None]
    if not valid:
        return None
    return _circular_mean_time([v[0] for v in valid], [v[1] for v in valid])


# ─────────────────────────────────────────────
# BETRAG & VERTEILUNG
# ─────────────────────────────────────────────

def _direction(transactions: list[dict]) -> str:
    pos = sum(1 for t in transactions if t.get("betrag", 0) > 0)
    neg = sum(1 for t in transactions if t.get("betrag", 0) < 0)
    if pos > 0 and neg == 0:
        return "INFLOW"
    if neg > 0 and pos == 0:
        return "OUTFLOW"
    return "MIXED"


def _fit_lognormal(amounts: list[float]) -> dict:
    pos = [a for a in amounts if a > 0]
    if len(pos) < 2:
        return {}

    log_a  = [math.log(a) for a in pos]
    mu_ln  = mean(log_a)
    sd_ln  = stdev(log_a) if len(log_a) >= 2 else 0.0

    ln_mean = math.exp(mu_ln + sd_ln ** 2 / 2)
    ln_var  = (math.exp(sd_ln ** 2) - 1) * math.exp(2 * mu_ln + sd_ln ** 2)
    ln_std  = math.sqrt(max(0.0, ln_var))

    return {
        "dist_type":        "lognormal",
        "lognorm_mu":       round(mu_ln, 4),
        "lognorm_sigma":    round(sd_ln, 4),
        "amount_mean":      round(ln_mean, 2),
        "amount_std":       round(ln_std, 2),
        "amount_min":       round(min(pos), 2),
        "amount_max":       round(max(pos), 2),
        "amount_median":    round(math.exp(mu_ln), 2),
        "amount_p5":        round(math.exp(mu_ln - 1.645 * sd_ln), 2),
        "amount_p95":       round(math.exp(mu_ln + 1.645 * sd_ln), 2),
        "amount_p25":       round(math.exp(mu_ln - 0.674 * sd_ln), 2),
        "amount_p75":       round(math.exp(mu_ln + 0.674 * sd_ln), 2),
        "amount_ci_90_low": round(max(0.0, ln_mean - CI_90_Z * ln_std), 2),
        "amount_ci_90_high":round(ln_mean + CI_90_Z * ln_std, 2),
        "amount_ci_95_low": round(max(0.0, ln_mean - CI_95_Z * ln_std), 2),
        "amount_ci_95_high":round(ln_mean + CI_95_Z * ln_std, 2),
    }


def _fit_normal_fallback(amounts: list[float]) -> dict:
    if not amounts:
        return {}
    mu = mean(amounts)
    sd = stdev(amounts) if len(amounts) >= 2 else 0.0
    return {
        "dist_type":        "normal",
        "amount_mean":      round(mu, 2),
        "amount_std":       round(sd, 2),
        "amount_min":       round(min(amounts), 2),
        "amount_max":       round(max(amounts), 2),
        "amount_median":    round(sorted(amounts)[len(amounts) // 2], 2),
        "amount_p5":        round(max(0.0, mu - 1.645 * sd), 2),
        "amount_p95":       round(mu + 1.645 * sd, 2),
        "amount_ci_90_low": round(max(0.0, mu - CI_90_Z * sd), 2),
        "amount_ci_90_high":round(mu + CI_90_Z * sd, 2),
        "amount_ci_95_low": round(max(0.0, mu - CI_95_Z * sd), 2),
        "amount_ci_95_high":round(mu + CI_95_Z * sd, 2),
    }


def _amount_stats(amounts: list[float]) -> dict:
    result = _fit_lognormal(amounts)
    if not result:
        result = _fit_normal_fallback(amounts)
    return result


# ─────────────────────────────────────────────
# POISSON + TAG-IM-MONAT VERTEILUNG
# ─────────────────────────────────────────────

def _recency_weight(tx_date: date, reference_date: date) -> float:
    months_ago = (
        (reference_date.year - tx_date.year) * 12
        + (reference_date.month - tx_date.month)
    )
    return RECENCY_DECAY ** max(0, months_ago)


def _weighted_lambda(transactions: list[dict], reference_date: date, n_months: int) -> float:
    if not transactions:
        return 0.0
    total_weight = sum(_recency_weight(t["_date_obj"], reference_date) for t in transactions)
    return round(total_weight / n_months, 4)


def _dom_probability(transactions: list[dict], reference_date: date) -> dict[str, float]:
    dom_weights: defaultdict[int, float] = defaultdict(float)
    for t in transactions:
        dom = t["_date_obj"].day
        dom_weights[dom] += _recency_weight(t["_date_obj"], reference_date)

    total = sum(dom_weights.values())
    if total == 0:
        return {}
    return {f"{dom:02d}": round(w / total, 4) for dom, w in sorted(dom_weights.items())}


def _dow_probability(transactions: list[dict], reference_date: date) -> dict[str, float]:
    dow_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    dow_weights: defaultdict[int, float] = defaultdict(float)
    for t in transactions:
        dow = t["_date_obj"].weekday()
        dow_weights[dow] += _recency_weight(t["_date_obj"], reference_date)

    total = sum(dow_weights.values())
    if total == 0:
        return {}
    return {dow_names[dow]: round(w / total, 4) for dow, w in sorted(dow_weights.items())}


def _poisson_stats(lm: float) -> dict:
    p_any = round(1.0 - math.exp(-lm), 4)
    return {
        "lambda_per_month":    lm,
        "probability_any":     p_any,
        "probability_any_pct": round(p_any * 100, 1),
    }


# ─────────────────────────────────────────────
# MONATLICHE AUFSCHLÜSSELUNG
# ─────────────────────────────────────────────

def _month_label(d: date) -> str:
    return d.strftime("%Y-%m")


def _iter_months(start: date, end: date):
    cursor = start.replace(day=1)
    while cursor <= end:
        yield cursor
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )


def _compute_monthly_breakdown(transactions: list[dict], cutoff: date, ref: date) -> dict[str, dict]:
    breakdown: dict[str, dict] = {
        _month_label(m): {"count": 0, "inflow": 0.0, "outflow": 0.0, "net": 0.0}
        for m in _iter_months(cutoff, ref)
    }
    for t in transactions:
        lbl = _month_label(t["_date_obj"])
        if lbl not in breakdown:
            continue
        amt = t.get("betrag", 0.0)
        breakdown[lbl]["count"] += 1
        if amt >= 0:
            breakdown[lbl]["inflow"]  = round(breakdown[lbl]["inflow"]  + amt, 2)
        else:
            breakdown[lbl]["outflow"] = round(breakdown[lbl]["outflow"] + abs(amt), 2)
        breakdown[lbl]["net"] = round(breakdown[lbl]["inflow"] - breakdown[lbl]["outflow"], 2)
    return breakdown


# ─────────────────────────────────────────────
# TÄGLICHE PROGNOSE
# ─────────────────────────────────────────────

def _daily_lambda(lm_month: float, dom: int, year: int, month: int, dom_probs: dict[str, float]) -> float:
    if not dom_probs:
        days_in_month = monthrange(year, month)[1]
        return lm_month / days_in_month
    p = dom_probs.get(f"{dom:02d}", 0.0)
    return lm_month * p


def _build_daily_forecast(
    reference_date: date,
    horizon_days: int,
    lm_month: float,
    dom_probs: dict[str, float],
    amount_mean: float,
    amount_std: float,
    dist_type: str,
    direction: str,
    typical_time: Optional[str],
    lognorm_mu: Optional[float] = None,
    lognorm_sigma: Optional[float] = None,
) -> list[dict]:
    forecast = []
    current  = reference_date

    for _ in range(horizon_days):
        current  = current + timedelta(days=1)
        lam_day  = _daily_lambda(lm_month, current.day, current.year, current.month, dom_probs)
        p_day    = round(1.0 - math.exp(-lam_day), 4)
        exp_amt  = round(lam_day * amount_mean, 2)

        if dist_type == "lognormal" and lognorm_mu is not None and lognorm_sigma:
            ci_low  = round(max(0.0, lam_day * math.exp(lognorm_mu - CI_90_Z * lognorm_sigma)), 2)
            ci_high = round(lam_day * math.exp(lognorm_mu + CI_90_Z * lognorm_sigma), 2)
        else:
            ci_low  = round(max(0.0, exp_amt - CI_90_Z * lam_day * amount_std), 2)
            ci_high = round(exp_amt + CI_90_Z * lam_day * amount_std, 2)

        sign     = -1 if direction == "OUTFLOW" else 1
        net_flow = round(sign * exp_amt, 2)
        net_low  = round(sign * (ci_high if sign < 0 else ci_low), 2)
        net_high = round(sign * (ci_low  if sign < 0 else ci_high), 2)

        expected_dt = (
            f"{current.isoformat()}T{typical_time}:00Z"
            if typical_time
            else current.isoformat()
        )

        if p_day > 0.01:
            forecast.append({
                "date":             current.isoformat(),
                "expected_datetime":expected_dt,
                "day_of_week":      ["Mo","Di","Mi","Do","Fr","Sa","So"][current.weekday()],
                "p_transaction":    p_day,
                "p_pct":            round(p_day * 100, 1),
                "expected_amount":  exp_amt,
                "net_cashflow":     net_flow,
                "ci_90_low":        ci_low,
                "ci_90_high":       ci_high,
                "net_ci_90_low":    net_low,
                "net_ci_90_high":   net_high,
                "direction":        direction,
            })

    return forecast


def _compute_cumulative_cashflow(daily_forecast: list[dict]) -> list[dict]:
    cumulative = 0.0
    result = []
    for day in daily_forecast:
        cumulative = round(cumulative + day["net_cashflow"], 2)
        result.append({
            "date":           day["date"],
            "daily_net":      day["net_cashflow"],
            "cumulative_net": cumulative,
            "p_transaction":  day["p_transaction"],
        })
    return result


def _liquidity_risk_metrics(daily_forecast: list[dict], cum_cashflow: list[dict]) -> dict:
    if not daily_forecast:
        return {}

    high_prob = [d for d in daily_forecast if d["p_transaction"] >= 0.75]
    outflows  = [d for d in daily_forecast if d["net_cashflow"] < 0]
    inflows   = [d for d in daily_forecast if d["net_cashflow"] > 0]

    peak_out = min(outflows, key=lambda d: d["net_cashflow"])  if outflows else None
    peak_in  = max(inflows,  key=lambda d: d["net_cashflow"])  if inflows  else None

    monthly_net: defaultdict[str, float] = defaultdict(float)
    for d in daily_forecast:
        mo = d["date"][:7]
        monthly_net[mo] = round(monthly_net[mo] + d["net_cashflow"], 2)

    min_cum  = min((c["cumulative_net"] for c in cum_cashflow), default=0.0)
    min_date = next((c["date"] for c in cum_cashflow if c["cumulative_net"] == min_cum), None)

    return {
        "high_probability_days":   [d["date"] for d in high_prob[:10]],
        "high_probability_count":  len(high_prob),
        "peak_outflow_day":        peak_out["date"]              if peak_out else None,
        "peak_outflow_amount":     abs(peak_out["net_cashflow"]) if peak_out else None,
        "peak_inflow_day":         peak_in["date"]               if peak_in  else None,
        "peak_inflow_amount":      peak_in["net_cashflow"]       if peak_in  else None,
        "monthly_net_forecast":    dict(sorted(monthly_net.items())),
        "cumulative_minimum":      round(min_cum, 2),
        "cumulative_minimum_date": min_date,
        "liquidity_risk":          min_cum < 0,
    }


# ─────────────────────────────────────────────
# DOKUMENT PRO GRUPPE ZUSAMMENBAUEN
# ─────────────────────────────────────────────

def _build_distribution_doc(
    group_key: str,
    txs: list[dict],
    cutoff: date,
    ref: date,
    n_months: int,
) -> dict:
    if " :: " in group_key:
        category, counterparty = group_key.split(" :: ", 1)
        group_type = "category_counterparty"
    else:
        category, counterparty = group_key, None
        group_type = "category_only"

    direction     = _direction(txs)
    amounts       = [abs(t.get("betrag", 0.0)) for t in txs]
    amt_stats     = _amount_stats(amounts)
    amount_mean   = amt_stats.get("amount_mean", 0.0)
    amount_std    = amt_stats.get("amount_std", 0.0)
    dist_type     = amt_stats.get("dist_type", "normal")
    lognorm_mu    = amt_stats.get("lognorm_mu")
    lognorm_sigma = amt_stats.get("lognorm_sigma")

    lm_month  = _weighted_lambda(txs, ref, n_months)
    poisson   = _poisson_stats(lm_month)
    dom_probs = _dom_probability(txs, ref)
    dow_probs = _dow_probability(txs, ref)

    typical_time = _extract_times(txs)
    monthly      = _compute_monthly_breakdown(txs, cutoff, ref)

    daily_forecast = _build_daily_forecast(
        reference_date = ref,
        horizon_days   = FORECAST_HORIZON_DAYS,
        lm_month       = lm_month,
        dom_probs      = dom_probs,
        amount_mean    = amount_mean,
        amount_std     = amount_std,
        dist_type      = dist_type,
        direction      = direction,
        typical_time   = typical_time,
        lognorm_mu     = lognorm_mu,
        lognorm_sigma  = lognorm_sigma,
    )

    cum_cashflow      = _compute_cumulative_cashflow(daily_forecast)
    liquidity_metrics = _liquidity_risk_metrics(daily_forecast, cum_cashflow)

    amount_expected_monthly = round(lm_month * amount_mean, 2)
    sign_monthly            = -1 if direction == "OUTFLOW" else 1
    net_expected_monthly    = round(sign_monthly * amount_expected_monthly, 2)

    return {
        "group_key":               group_key,
        "group_type":              group_type,
        "category":                category,
        "counterparty":            counterparty,
        "direction":               direction,
        "sample_size":             len(txs),
        "lookback_months":         n_months,
        "analysis_from":           cutoff.isoformat(),
        "analysis_to":             ref.isoformat(),
        "calculated_at":           datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "typical_booking_time":    typical_time,
        **poisson,
        "dom_probability":         dom_probs,
        "dow_probability":         dow_probs,
        **amt_stats,
        "amount_expected_monthly": amount_expected_monthly,
        "net_expected_monthly":    net_expected_monthly,
        "monthly_breakdown":       monthly,
        "daily_forecast":          daily_forecast,
        "cumulative_cashflow":     cum_cashflow,
        **liquidity_metrics,
    }


# ─────────────────────────────────────────────
# FIRESTORE SCHREIBEN
# ─────────────────────────────────────────────

def _safe_doc_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)[:200]


def _write_to_firestore(db, documents: list[dict]) -> int:
    ref     = db.collection(TARGET_COLLECTION)
    written = 0
    for doc in documents:
        ref.document(_safe_doc_id(doc["group_key"])).set(doc)
        written += 1
    return written


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def calculate_unknown() -> dict:
    """Haupteinstiegspunkt für pipeline.py."""
    db = _init_firestore()

    all_txs = _load_all_distributions(db)
    if not all_txs:
        print("    ⚠️  Keine Transaktionen in distributions_db.")
        return {"groups_written": 0, "transactions_analyzed": 0}

    cutoff, ref, n_months = _determine_window(all_txs)
    print(f"    Zeitfenster  : {cutoff} → {ref}  ({n_months} Monate)")

    txs = _filter_by_window(all_txs, cutoff, ref)
    print(f"    Transaktionen: {len(txs)} im Fenster / {len(all_txs)} total")

    if not txs:
        print("    ⚠️  Keine Transaktionen im Fenster.")
        return {"groups_written": 0, "transactions_analyzed": 0}

    groups = _build_groups(txs)
    print(f"    Gruppen      : {len(groups)}")

    docs    = [_build_distribution_doc(gkey, gtxs, cutoff, ref, n_months) for gkey, gtxs in groups.items()]
    written = _write_to_firestore(db, docs)
    print(f"    Geschrieben  : {written} Dokumente → '{TARGET_COLLECTION}'")

    return {"groups_written": written, "transactions_analyzed": len(txs)}


# ─────────────────────────────────────────────
# KONSOLENAUSGABE  (unverändert)
# ─────────────────────────────────────────────

def _print_group(doc: dict):
    SEP = "." * 68
    dir_sym = {"INFLOW": "▲", "OUTFLOW": "▼", "MIXED": "◆"}.get(doc["direction"], "?")
    print(f"\n  {'─'*68}")
    print(f"  {dir_sym}  {doc['group_key']}")
    print(f"  {SEP}")
    print(f"  Richtung       : {doc['direction']}  |  "
          f"Verteilung : {doc.get('dist_type','?').upper()}  |  "
          f"Buchungszeit: {doc.get('typical_booking_time') or '–'}")
    print(f"  Stichproben    : {doc['sample_size']}  "
          f"({doc['analysis_from']} → {doc['analysis_to']})")
    print(f"  λ/Monat        : {doc['lambda_per_month']}  "
          f"→ P(≥1 Tx/Mo) = {doc['probability_any_pct']}%")
    print(f"  Betrag Ø       : EUR {doc.get('amount_mean',0):>10,.2f}  "
          f"(σ = {doc.get('amount_std',0):,.2f})")
    print(f"  P5 / P95       : EUR {doc.get('amount_p5',0):>10,.2f}  –  "
          f"EUR {doc.get('amount_p95',0):,.2f}")
    print(f"  Erw. Netto/Mo  : EUR {doc.get('net_expected_monthly',0):>+10,.2f}")

    dom_p = doc.get("dom_probability", {})
    if dom_p:
        top3    = sorted(dom_p.items(), key=lambda x: -x[1])[:3]
        dom_str = "  ".join(f"Tag {int(d)}: {p*100:.0f}%" for d, p in top3)
        print(f"  Top-Buchungst. : {dom_str}")

    if doc.get("liquidity_risk"):
        print(f"  ⚠️  LIQUIDITÄTSRISIKO  |  "
              f"Minimum: EUR {doc.get('cumulative_minimum',0):,.2f}  "
              f"am {doc.get('cumulative_minimum_date','?')}")

    hpd = doc.get("high_probability_days", [])[:5]
    if hpd:
        print(f"  Nächste Buchungen (≥75%): {', '.join(hpd)}")

    mnf = doc.get("monthly_net_forecast", {})
    if mnf:
        items  = list(mnf.items())[:3]
        fc_str = "  ".join(f"{m}: EUR {v:+,.0f}" for m, v in items)
        print(f"  Forecast Netto : {fc_str}")


def _print_summary_table(docs: list[dict]):
    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  FORECAST DISTRIBUTION  –  ÜBERSICHT  ({len(docs)} Gruppen)")
    print(f"{SEP}")
    header = f"  {'Gruppe':<38}  {'Dir':7}  {'P(≥1)':6}  {'Ø Betrag':>11}  {'Netto/Mo':>11}"
    print(header)
    print(f"  {'─'*38}  {'─'*7}  {'─'*6}  {'─'*11}  {'─'*11}")
    for doc in sorted(docs, key=lambda d: abs(d.get("net_expected_monthly", 0)), reverse=True):
        print(
            f"  {doc['group_key'][:38]:<38}  "
            f"{doc['direction']:7}  "
            f"{doc['probability_any_pct']:>5.1f}%  "
            f"EUR {doc.get('amount_mean', 0):>7,.0f}  "
            f"EUR {doc.get('net_expected_monthly', 0):>+7,.0f}"
        )
    print(f"{SEP}")

    total_monthly = sum(d.get("net_expected_monthly", 0) for d in docs)
    print(f"\n  Ø Netto-Cashflow/Monat (alle Gruppen): EUR {total_monthly:>+10,.2f}")

    risk_groups = [d for d in docs if d.get("liquidity_risk")]
    if risk_groups:
        print(f"\n  ⚠️  Liquiditätsrisiko in {len(risk_groups)} Gruppe(n):")
        for d in risk_groups:
            print(f"     • {d['group_key'][:50]}  "
                  f"Min: EUR {d.get('cumulative_minimum',0):,.2f}  "
                  f"am {d.get('cumulative_minimum_date','?')}")
    print(f"{SEP}")


def main():
    print(f"\n{'='*70}")
    print(f"  CALCULATE UNKNOWN  –  Liquiditätsprognose")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    db = _init_firestore()

    print(f"  Lade '{SOURCE_COLLECTION}' ...")
    all_txs = _load_all_distributions(db)
    print(f"  ✅  {len(all_txs)} Transaktionen geladen\n")

    if not all_txs:
        print("  ⚠️  Keine Transaktionen – Abbruch.")
        sys.exit(0)

    cutoff, ref, n_months = _determine_window(all_txs)
    txs = _filter_by_window(all_txs, cutoff, ref)

    print(f"  Zeitfenster  : {cutoff} → {ref}  ({n_months} Monate)")
    print(f"  Im Fenster   : {len(txs)} / {len(all_txs)} Transaktionen\n")

    if not txs:
        print("  ⚠️  Keine Transaktionen im Fenster – Abbruch.")
        sys.exit(0)

    groups = _build_groups(txs)
    docs   = [
        _build_distribution_doc(gkey, gtxs, cutoff, ref, n_months)
        for gkey, gtxs in groups.items()
    ]

    for doc in docs:
        _print_group(doc)

    _print_summary_table(docs)

    written = _write_to_firestore(db, docs)

    print(f"\n{'='*70}")
    print(f"  ✅  {written} Dokumente → Firestore '{TARGET_COLLECTION}'")
    print(f"  ✅  {len(txs)} Transaktionen analysiert")
    print(f"  ✅  {FORECAST_HORIZON_DAYS}-Tage-Prognose ab {ref}")
    print(f"  Abgeschlossen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
