#!/usr/bin/env python3
"""
FABO B2B Lead Scorer — Store Opener Prediction
Route: ad_account -> OUTCOME_LEADS campaigns -> ACTIVE ads -> leads
No leadgen_forms endpoint needed. Works with ads_read permission.

SELF-ADAPTING: Uses a two-stage AI pipeline.
  Stage 1 — Form Interpreter (runs once per unique form_id):
    Reads every question+answer pair on a brand-new form and produces a
    structured field-meaning map that is cached in BigQuery. This means
    any new form Meta sends — new investment amounts, new questions —
    is understood automatically without touching this code.

  Stage 2 — Lead Scorer (runs per lead):
    Uses the cached field map to build the richest possible context for
    each lead and scores 0–100 with full reasoning.
"""

import os, re, time, json, requests
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET  = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
GROQ_API_KEY      = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"
LEADS_START_DATE  = os.getenv("LEADS_START_DATE", "2026-04-01")

# Stage 1 uses a larger model — it only runs once per new form so cost is tiny
GROQ_FAST_MODEL   = "llama-3.1-8b-instant"    # Stage 2: per-lead scoring (fast + cheap)
GROQ_SMART_MODEL  = "llama-3.3-70b-versatile" # Stage 1: form interpretation (smarter)

_raw = os.getenv("META_AD_ACCOUNT_ID", "")
META_AD_ACCOUNT_ID = f"act_{_raw}" if _raw and not _raw.startswith("act_") else _raw

for var, val in [("GCP_PROJECT_ID", GCP_PROJECT_ID),
                 ("META_ACCESS_TOKEN", META_ACCESS_TOKEN),
                 ("META_AD_ACCOUNT_ID", META_AD_ACCOUNT_ID),
                 ("GROK_API_KEY / GROQ_API_KEY", GROQ_API_KEY)]:
    if not val:
        raise ValueError(f"{var} is required")

print(f"Ad Account : {META_AD_ACCOUNT_ID}")
print(f"Start date : {LEADS_START_DATE}")

# ── BigQuery ──────────────────────────────────────────────────────────────────
bq = None

def get_bq():
    global bq
    if bq is None:
        bq = bigquery.Client(project=GCP_PROJECT_ID)
    return bq

def setup_bigquery():
    get_bq()
    dataset_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}"
    try:
        bq.get_dataset(dataset_id)
    except Exception:
        bq.create_dataset(bigquery.Dataset(dataset_id))
        print(f"Created dataset {dataset_id}")

    # Table 1: lead scores (main output)
    scores_schema = [
        bigquery.SchemaField("lead_id",           "STRING"),
        bigquery.SchemaField("lead_name",          "STRING"),
        bigquery.SchemaField("phone",              "STRING"),
        bigquery.SchemaField("email",              "STRING"),
        bigquery.SchemaField("city",               "STRING"),
        bigquery.SchemaField("state",              "STRING"),
        bigquery.SchemaField("platform",           "STRING"),
        bigquery.SchemaField("investment_ready",   "STRING"),
        bigquery.SchemaField("timeline",           "STRING"),
        bigquery.SchemaField("earning_intent",     "STRING"),
        bigquery.SchemaField("campaign_id",        "STRING"),
        bigquery.SchemaField("campaign_name",      "STRING"),
        bigquery.SchemaField("ad_name",            "STRING"),
        bigquery.SchemaField("form_name",          "STRING"),
        bigquery.SchemaField("lead_created_time",  "TIMESTAMP"),
        bigquery.SchemaField("store_open_score",   "INTEGER"),
        bigquery.SchemaField("grade",              "STRING"),
        bigquery.SchemaField("score_reasoning",    "STRING"),   # NEW: AI's brief rationale
        bigquery.SchemaField("scored_at",          "TIMESTAMP"),
    ]
    scores_table = f"{dataset_id}.lead_scores"
    _ensure_table(scores_table, scores_schema)

    # Table 2: form field maps — one row per unique form_id
    # Stores the AI-interpreted meaning of every field so we never re-interpret
    forms_schema = [
        bigquery.SchemaField("form_id",         "STRING"),
        bigquery.SchemaField("form_name",       "STRING"),
        bigquery.SchemaField("field_map_json",  "STRING"),   # JSON: {field_name: meaning_dict}
        bigquery.SchemaField("interpreted_at",  "TIMESTAMP"),
    ]
    forms_table = f"{dataset_id}.form_field_maps"
    _ensure_table(forms_table, forms_schema)

    return scores_table, forms_table

def _ensure_table(table_id, schema):
    try:
        bq.get_table(table_id)
        print(f"  Table {table_id.split('.')[-1]} exists.")
    except Exception:
        bq.create_table(bigquery.Table(table_id, schema=schema))
        print(f"  Created table {table_id.split('.')[-1]}")

def get_already_scored_ids(scores_table):
    get_bq()
    try:
        ids = {row.lead_id for row in bq.query(f"SELECT lead_id FROM `{scores_table}`").result()}
        print(f"Already in BigQuery: {len(ids)} leads (will skip)")
        return ids
    except Exception as e:
        print(f"Could not query existing leads: {e}")
        return set()

def get_since_timestamp(scores_table):
    get_bq()
    seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    try:
        rows = list(bq.query(f"SELECT MAX(lead_created_time) AS latest FROM `{scores_table}`").result())
        latest = rows[0].latest if rows else None
        if latest:
            since = latest + timedelta(seconds=1)
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            if since_str < seven_days_ago:
                print(f"Incremental (capped 7d): after {seven_days_ago}")
                return seven_days_ago, True
            print(f"Incremental: after {since_str}")
            return since_str, True
    except Exception as e:
        print(f"Timestamp query failed: {e}")
    since_str = f"{LEADS_START_DATE}T00:00:00+00:00"
    print(f"Full backfill from {LEADS_START_DATE}")
    return since_str, False

# ── Form field-map cache (loaded once per run from BigQuery) ──────────────────
_form_cache: dict[str, dict] = {}   # form_id -> field_map dict

def load_form_cache(forms_table):
    """Pull all previously-interpreted form maps into memory."""
    global _form_cache
    try:
        for row in bq.query(f"SELECT form_id, field_map_json FROM `{forms_table}`").result():
            try:
                _form_cache[row.form_id] = json.loads(row.field_map_json)
            except Exception:
                pass
        print(f"  Loaded {len(_form_cache)} cached form map(s)")
    except Exception as e:
        print(f"  Could not load form cache: {e}")

def get_form_map(form_id, form_name, sample_fields: dict, forms_table) -> dict:
    """
    Return the field-meaning map for a form.
    On first encounter, ask the AI to interpret every field, cache the result.

    field_map format (one entry per custom field):
    {
      "are_you_comfortable_to_make_an_investment_around_inr_17_lakhs?": {
        "meaning":   "investment_readiness",
        "high_values":  ["yes"],
        "med_values":   ["maybe", "need_financing"],
        "low_values":   ["no"],
        "weight":       "primary"          # primary | secondary | context
      },
      ...
    }
    """
    if form_id in _form_cache:
        return _form_cache[form_id]

    print(f"  [NEW FORM] Interpreting fields for: {form_name} ({form_id})")

    # Build field list for AI — show field name + sample value
    field_lines = "\n".join(
        f'  Field: "{k}"  |  Sample answer: "{v}"'
        for k, v in sample_fields.items()
    )

    prompt = f"""You are a data analyst building a lead-scoring system for FABO, India's premium laundry franchise brand (franchises cost ₹17 Lakh; master franchises cost ₹1 Crore).

A Meta lead form called "{form_name}" has these custom question fields:
{field_lines}

For EACH field, determine:
1. meaning — one of: investment_readiness | timeline | earning_intent | location_detail | business_experience | other
2. high_values — list of answer values that indicate STRONG buying signal (invest / act soon)
3. med_values  — list of answer values that indicate MODERATE signal (interested but cautious)
4. low_values  — list of answer values that indicate WEAK signal (not ready / exploring)
5. weight      — one of: primary (most important for scoring) | secondary | context (background info only)

Return ONLY a valid JSON object like this, with one key per field name, nothing else:
{{
  "<field_name>": {{
    "meaning": "<meaning>",
    "high_values": ["<val1>", "<val2>"],
    "med_values":  ["<val1>"],
    "low_values":  ["<val1>"],
    "weight":      "<weight>"
  }}
}}"""

    result = {}
    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": GROQ_SMART_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 800, "temperature": 0.0},
                timeout=45)
            if r.status_code == 429:
                time.sleep(6 * (attempt + 1)); continue
            if not r.ok:
                print(f"  [FORM INTERP] API error {r.status_code}")
                break
            raw = r.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = _safe_json(raw)
            if parsed and isinstance(parsed, dict):
                result = parsed
                break
        except Exception as e:
            print(f"  [FORM INTERP] Exception: {e}")

    if not result:
        # Fallback: build a basic map from field names using heuristics
        result = _heuristic_field_map(sample_fields)
        print(f"  [FORM INTERP] Used heuristic fallback for {form_name}")
    else:
        print(f"  [FORM INTERP] AI interpreted {len(result)} fields for {form_name}")

    # Cache in memory and persist to BigQuery
    _form_cache[form_id] = result
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        errs = bq.insert_rows_json(forms_table, [{
            "form_id":        form_id,
            "form_name":      form_name,
            "field_map_json": json.dumps(result),
            "interpreted_at": now_str,
        }])
        if errs:
            print(f"  [BQ] Form map save error: {errs}")
    except Exception as e:
        print(f"  [BQ] Could not save form map: {e}")

    return result

def _heuristic_field_map(fields: dict) -> dict:
    """Keyword-based fallback when AI form interpretation fails."""
    result = {}
    for k, v in fields.items():
        kl = k.lower()
        if any(x in kl for x in ["invest", "capital", "fund", "lakh", "crore", "money"]):
            result[k] = {"meaning": "investment_readiness", "weight": "primary",
                         "high_values": ["yes", "ready", "comfortable"],
                         "med_values":  ["maybe", "exploring", "financing"],
                         "low_values":  ["no"]}
        elif any(x in kl for x in ["when", "start", "timeline", "time", "begin", "launch"]):
            result[k] = {"meaning": "timeline", "weight": "secondary",
                         "high_values": ["immediately", "asap", "1_month", "3_months"],
                         "med_values":  ["quarter", "6_months", "this_year"],
                         "low_values":  ["exploring", "not_sure", "later"]}
        elif any(x in kl for x in ["earn", "income", "opportunity", "business", "intent"]):
            result[k] = {"meaning": "earning_intent", "weight": "secondary",
                         "high_values": ["franchise", "own_store", "master"],
                         "med_values":  ["side_income", "commission"],
                         "low_values":  ["exploring", "not_sure"]}
        else:
            result[k] = {"meaning": "other", "weight": "context",
                         "high_values": [], "med_values": [], "low_values": []}
    return result

# ── Meta API ──────────────────────────────────────────────────────────────────
def iso_to_unix(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())

def meta_paginate(url, params):
    results = []
    while url:
        for attempt in range(4):
            r = requests.get(url, params=params if params else {}).json()
            if "error" in r:
                msg  = r["error"]["message"]
                code = r["error"].get("code", 0)
                if code in (17, 80004) or "too many calls" in msg.lower():
                    wait = 30 * (attempt + 1)
                    print(f"  [RATE LIMIT] waiting {wait}s...")
                    time.sleep(wait); continue
                return results, msg
            break
        else:
            return results, "Rate limit: retries exhausted"
        results.extend(r.get("data", []))
        url    = r.get("paging", {}).get("next")
        params = None
        time.sleep(0.3)
    return results, None

def validate_token():
    me = requests.get("https://graph.facebook.com/v19.0/me",
                      params={"access_token": META_ACCESS_TOKEN}).json()
    if "error" in me:
        print(f"[META] Token invalid: {me['error']['message']}"); return False
    print(f"[META] Token OK — {me.get('name', me.get('id'))}"); return True

def fetch_active_lead_ads(active_only=False):
    base = "https://graph.facebook.com/v19.0"
    campaigns, err = meta_paginate(
        f"{base}/{META_AD_ACCOUNT_ID}/campaigns",
        {"access_token": META_ACCESS_TOKEN, "fields": "id,name,objective,status", "limit": 100}
    )
    if err:
        print(f"[META] Campaign error: {err}"); return []

    LEAD_OBJ   = {"LEAD_GENERATION", "OUTCOME_LEADS"}
    lead_camps = [c for c in campaigns if c.get("objective") in LEAD_OBJ]
    print(f"  Campaigns: {len(campaigns)} total, {len(lead_camps)} lead gen")
    if not lead_camps:
        print("  No lead gen campaigns found."); return []

    all_ads = []
    for camp in lead_camps:
        time.sleep(0.3)
        ad_params = {"access_token": META_ACCESS_TOKEN,
                     "fields": "id,name,effective_status", "limit": 100}
        if active_only:
            ad_params["filtering"] = '[{"field":"effective_status","operator":"IN","value":["ACTIVE","PAUSED"]}]'
        ads, err = meta_paginate(f"{base}/{camp['id']}/ads", ad_params)
        if err:
            print(f"  Campaign '{camp['name']}' ads error: {err}"); continue
        for ad in ads:
            ad["_campaign_id"]   = camp["id"]
            ad["_campaign_name"] = camp["name"]
        all_ads.extend(ads)

    mode = "active/paused" if active_only else "all (backfill)"
    print(f"  Total lead ads ({mode}): {len(all_ads)}")
    return all_ads

def fetch_leads_from_ad(ad, since_ts, until_ts):
    base     = "https://graph.facebook.com/v19.0"
    until_dt = datetime.fromisoformat(until_ts.replace("Z", "+00:00"))
    url      = f"{base}/{ad['id']}/leads"
    params   = {
        "access_token": META_ACCESS_TOKEN,
        "fields": "id,created_time,field_data,platform,campaign_id,campaign_name,ad_name,form_id,form_name",
        "filtering": f'[{{"field":"time_created","operator":"GREATER_THAN","value":{iso_to_unix(since_ts)}}}]',
        "limit": 100,
    }
    leads = []
    while url:
        for attempt in range(3):
            r = requests.get(url, params=params if params else {}).json()
            if "error" in r:
                code = r["error"].get("code", 0)
                msg  = r["error"]["message"]
                if code in (17, 80004) or "too many calls" in msg.lower():
                    time.sleep(15 * (attempt + 1)); continue
                if code != 100:
                    print(f"    '{ad.get('name','?')}' error: {msg}")
                return leads
            break
        else:
            return leads
        for lead in r.get("data", []):
            try:
                if datetime.fromisoformat(lead["created_time"].replace("Z", "+00:00")) > until_dt:
                    continue
            except Exception:
                pass
            lead["_ad_name"]       = ad.get("name", "")
            lead["_campaign_id"]   = ad.get("_campaign_id", "")
            lead["_campaign_name"] = ad.get("_campaign_name", "")
            leads.append(lead)
        url    = r.get("paging", {}).get("next")
        params = None
    return leads

# ── Field extraction ──────────────────────────────────────────────────────────
# Contact fields are stable across every form; custom questions go to extra_fields
CONTACT_KEYS = {
    "full_name", "name", "first_name",
    "phone_number", "phone",
    "email",
    "city", "state", "province",
    "additional_col1_select",   # legacy state
    "additional_col2",          # legacy city
}

def _first(*keys, src):
    for k in keys:
        v = str(src.get(k, "")).strip()
        if v: return v
    return ""

def parse_lead(lead):
    raw = {f["name"]: (f["values"][0] if f.get("values") else "")
           for f in lead.get("field_data", [])}
    extra = {k: str(v) for k, v in raw.items()
             if k not in CONTACT_KEYS and str(v).strip()}
    return {
        "lead_id":           lead.get("id", ""),
        "lead_name":         _first("full_name", "name", "first_name", src=raw),
        "phone":             _first("phone_number", "phone", src=raw).replace("p:", ""),
        "email":             _first("email", src=raw),
        "city":              _first("city", "additional_col2", src=raw),
        "state":             _first("additional_col1_select", "state", "province", src=raw),
        "platform":          lead.get("platform", ""),
        "campaign_id":       lead.get("_campaign_id", "") or lead.get("campaign_id", ""),
        "campaign_name":     lead.get("_campaign_name", "") or lead.get("campaign_name", ""),
        "ad_name":           lead.get("_ad_name", "") or lead.get("ad_name", ""),
        "form_id":           lead.get("form_id", ""),
        "form_name":         lead.get("form_name", ""),
        "lead_created_time": lead.get("created_time", ""),
        "extra_fields":      extra,
        # Populated by AI scorer
        "investment_ready":  "",
        "timeline":          "",
        "earning_intent":    "",
    }

# ── Stage 1: ensure every form in this batch has a field map ─────────────────
def ensure_form_maps(leads, forms_table):
    """
    For any form_id not yet in cache, send ONE representative lead to the
    AI to interpret the form's fields. Groups by form_id and uses first lead
    as the sample.
    """
    seen = {}
    for lead in leads:
        fid = lead["form_id"]
        if fid and fid not in _form_cache and fid not in seen:
            seen[fid] = lead

    for fid, lead in seen.items():
        if lead["extra_fields"]:
            get_form_map(fid, lead["form_name"], lead["extra_fields"], forms_table)
        else:
            # No custom fields — store an empty map so we don't retry on every run
            _form_cache[fid] = {}

# ── Stage 2: AI lead scorer ───────────────────────────────────────────────────
REGION_MAP = {
    "assam":       "Northeast India — high volume, lower conversion",
    "kerala":      "Kerala — educated, cautious but committed once interested",
    "maharashtra": "Maharashtra — metro, competitive, serious investors",
    "bihar":       "Bihar — price-sensitive, high volume",
    "mp":          "Madhya Pradesh — tier 2/3, value seekers",
    "telangana":   "Telangana/Hyderabad — strong B2B ecosystem",
    "ap":          "Andhra Pradesh — similar to Telangana",
    "rajasthan":   "Rajasthan — strong family business culture",
    "gujarat":     "Gujarat — entrepreneurial culture, serious investors",
    "west bengal": "West Bengal — large market, moderate conversion",
    "jharkhand":   "Jharkhand — emerging market",
    "ncr":         "NCR/Delhi — high intent, highly competitive",
    "chandigarh":  "Chandigarh/Punjab — high income, serious buyers",
    "uttarakhand": "Uttarakhand — smaller market",
    "chattisgarh": "Chhattisgarh — tier 2 market",
    "odisha":      "Odisha — emerging B2B market",
    "punjab":      "Punjab — strong business culture",
    "bangalore":   "Bangalore/Karnataka — strong B2B ecosystem, tech-savvy",
    "karnataka":   "Karnataka — metro + tier 2, good conversion",
}

def get_ai_score(lead):
    """
    Two inputs to the scorer:
      1. Lead profile (name, city, platform, campaign, completeness)
      2. Form field map (AI-interpreted meanings) + actual answers

    Returns: (score int, investment_ready str, timeline str, earning_intent str, reasoning str)
    """
    platform = (lead["platform"] or "").lower()
    platform_note = {
        "ig": "Instagram — younger audience, impulse-driven",
        "fb": "Facebook — 30-55 age group, more deliberate",
    }.get(platform, "unknown platform")

    campaign = lead["campaign_name"] or ""
    region_note = next(
        (note for kw, note in REGION_MAP.items() if kw in campaign.lower()),
        "Region unknown"
    )

    has_city  = bool(lead["city"] and lead["city"] != "?")
    has_state = bool(lead["state"])
    has_phone = bool(lead["phone"])
    has_email = bool(lead["email"])
    has_name  = bool(lead["lead_name"])
    completeness = sum([has_city, has_state, has_phone, has_email, has_name])

    # Build the interpreted form section using the cached field map
    field_map = _form_cache.get(lead["form_id"], {})
    form_section_lines = []
    for field_name, answer in lead["extra_fields"].items():
        meta = field_map.get(field_name, {})
        meaning = meta.get("meaning", "unknown")
        weight  = meta.get("weight", "context")
        highs   = meta.get("high_values", [])
        meds    = meta.get("med_values", [])
        lows    = meta.get("low_values", [])

        # Determine signal strength for this answer
        ans_lower = answer.lower().strip()
        if any(h.lower() in ans_lower or ans_lower in h.lower() for h in highs):
            signal = "HIGH ✓✓"
        elif any(m.lower() in ans_lower or ans_lower in m.lower() for m in meds):
            signal = "MED  ✓"
        elif any(l.lower() in ans_lower or ans_lower in l.lower() for l in lows):
            signal = "LOW  ✗"
        else:
            signal = "UNKNOWN"

        form_section_lines.append(
            f'  Q: "{field_name}"\n'
            f'  A: "{answer}"  →  [{meaning.upper()}] [{weight.upper()}] signal: {signal}'
        )

    form_section = "\n".join(form_section_lines) if form_section_lines else "  (no custom fields)"

    prompt = f"""You are a senior B2B franchise sales analyst for FABO, India's premium laundry franchise brand.
Franchise cost: ₹17 Lakhs. Master Franchise cost: ₹1 Crore.

Score 0–100: the probability this specific lead will actually open a FABO store.

━━━ LEAD PROFILE ━━━
Name         : {lead['lead_name'] or 'Unknown'}
Location     : {lead['city'] or 'Unknown'}, {lead['state'] or 'Unknown'}
Platform     : {platform.upper()} — {platform_note}
Campaign     : {campaign or 'Unknown'}
Region       : {region_note}
Form         : {lead['form_name'] or 'Unknown'}
Completeness : {completeness}/5 (name={'✓' if has_name else '✗'} phone={'✓' if has_phone else '✗'} email={'✓' if has_email else '✗'} city={'✓' if has_city else '✗'} state={'✓' if has_state else '✗'})

━━━ FORM Q&A (with interpreted meaning and signal strength) ━━━
{form_section}

━━━ SCORING RULES ━━━
PRIMARY signals (investment readiness) drive 50% of the score:
  HIGH  → base 70–85
  MED   → base 40–60
  LOW   → base 5–30
  blank → base 20

SECONDARY signals (timeline) add/subtract up to 15 points:
  HIGH  (immediately / this month) → +12 to +15
  MED   (this quarter / 3–6 months) → +5 to +8
  LOW   (exploring / blank) → -10 to -15

FORM TYPE bonus (read from form name):
  "Master Franchise" or "1 Crore" → +5 (higher commitment to apply)
  "Franchise" → +0
  Unknown → +0

REGION modifier (±5):
  Strong (Gujarat, Maharastra, NCR, Punjab, Bangalore): +3 to +5
  Moderate (Kerala, Telangana, AP, Karnataka): +1 to +3
  Weak (Bihar, MP, Assam, Jharkhand, West Bengal): -3 to -5
  Unknown: -5

PLATFORM modifier:
  Facebook → +2  |  Instagram → 0  |  Unknown → -3

COMPLETENESS modifier:
  5/5 → +5  |  4/5 → +2  |  3/5 → 0  |  2/5 → -5  |  0-1/5 → -10

CALIBRATION (be honest — most leads do NOT convert):
  90–100: All signals HIGH + complete data + strong region (extremely rare)
  80–89 : HIGH investment + strong timeline + good data
  65–79 : HIGH investment but weaker timeline, or MED investment + urgent timeline
  45–64 : MED investment + moderate timeline — needs nurturing
  25–44 : LOW investment or weak signals — low priority
  0–24  : Not ready / exploring only — skip

Return ONLY this JSON, no extra text:
{{"score": <0-100>, "investment_ready": "<extracted answer>", "timeline": "<extracted answer>", "earning_intent": "<extracted answer or not_captured>", "reasoning": "<one sentence max 20 words why you gave this score>"}}"""

    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": GROQ_FAST_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 120, "temperature": 0.0},
                timeout=30)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1)); continue
            if not r.ok:
                break
            raw_resp = r.json()["choices"][0]["message"]["content"].strip()
            raw_resp = re.sub(r"```(?:json)?|```", "", raw_resp).strip()
            p = _safe_json(raw_resp)
            if p:
                score = max(0, min(100, int(p.get("score", 40))))
                return (score,
                        str(p.get("investment_ready", "")),
                        str(p.get("timeline", "")),
                        str(p.get("earning_intent", "")),
                        str(p.get("reasoning", "")))
            m = re.search(r"\b(\d{1,3})\b", raw_resp)
            return (max(0, min(100, int(m.group(1)))) if m else _fallback_score(lead),
                    "", "", "", "")
        except Exception:
            pass
    return _fallback_score(lead), "", "", "", "fallback"

def _fallback_score(lead):
    score = 30
    for v in lead.get("extra_fields", {}).values():
        vl = v.lower()
        if any(x in vl for x in ["ready", "yes", "immediately", "crore"]):
            score += 15
        elif any(x in vl for x in ["exploring", "maybe", "no"]):
            score -= 5
    if lead["phone"]: score += 5
    if lead["lead_name"]: score += 5
    return max(0, min(100, score))

def _safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return None

def grade(s):
    return "A" if s >= 85 else "B" if s >= 65 else "C" if s >= 45 else "D" if s >= 25 else "F"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    print("=" * 55)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 55)

    print("\nSetting up BigQuery...")
    scores_table, forms_table = setup_bigquery()

    print("\nValidating Meta token...")
    if not validate_token():
        return

    since_ts, is_incremental = get_since_timestamp(scores_table)
    until_ts = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    print(f"Window: {since_ts[:10]} → {until_ts[:10]}")

    already = get_already_scored_ids(scores_table)

    print("\nLoading form field maps from BigQuery...")
    load_form_cache(forms_table)

    print("\nFetching lead gen ads...")
    ads = fetch_active_lead_ads(active_only=is_incremental)
    if not ads:
        print("No active lead gen ads found."); return

    print(f"\nFetching leads from {len(ads)} ads...")
    all_leads = []
    for ad in ads:
        raw    = fetch_leads_from_ad(ad, since_ts, until_ts)
        parsed = [parse_lead(l) for l in raw]
        new    = [l for l in parsed if l["lead_id"] not in already]
        if new:
            print(f"  '{ad['name']}': {len(raw)} fetched, {len(new)} new")
        all_leads.extend(new)

    print(f"\nNew leads to score: {len(all_leads)}")
    if not all_leads:
        print("Nothing new — BigQuery is up to date."); return

    # Stage 1 — interpret any new forms before scoring begins
    print("\nStage 1: Interpreting new form structures...")
    ensure_form_maps(all_leads, forms_table)

    # Stage 2 — score every lead
    print("\nStage 2: Scoring leads with Groq AI...")
    rows = []
    total_saved = 0
    for i, lead in enumerate(all_leads, 1):
        score, inv_ready, timeline, earning, reasoning = get_ai_score(lead)
        g  = grade(score)
        ts = lead["lead_created_time"]
        try:
            ts_clean = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            ts_clean = now.strftime("%Y-%m-%dT%H:%M:%S")

        rows.append({
            "lead_id":           lead["lead_id"],
            "lead_name":         lead["lead_name"],
            "phone":             lead["phone"],
            "email":             lead["email"],
            "city":              lead["city"],
            "state":             lead["state"],
            "platform":          lead["platform"],
            "investment_ready":  inv_ready or lead["investment_ready"],
            "timeline":          timeline   or lead["timeline"],
            "earning_intent":    earning    or lead["earning_intent"],
            "campaign_id":       lead["campaign_id"],
            "campaign_name":     lead["campaign_name"],
            "ad_name":           lead["ad_name"],
            "form_name":         lead["form_name"],
            "lead_created_time": ts_clean,
            "store_open_score":  score,
            "grade":             g,
            "score_reasoning":   reasoning,
            "scored_at":         now.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[{i:3d}/{len(all_leads)}] {(lead['lead_name'] or '?'):25s} | "
              f"{(lead['city'] or '?'):12s} | {score:3d} ({g}) | {reasoning[:45]}")

        if i % 50 == 0:
            errs = bq.insert_rows_json(scores_table, rows[total_saved:i])
            if errs: print(f"  [BQ] Batch error: {errs}")
            else:    print(f"  [BQ] ✓ Saved rows {total_saved+1}–{i}")
            total_saved = i

    if len(rows) > total_saved:
        errs = bq.insert_rows_json(scores_table, rows[total_saved:])
        if errs: print(f"[BQ] Final batch error: {errs}")
        else:    print(f"[BQ] ✓ Saved final {len(rows) - total_saved} rows")

    grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in rows:
        grades[r["grade"]] += 1

    print(f"\n{'─'*55}")
    print(f"  Window : {since_ts[:10]} → {until_ts[:10]}")
    print(f"  Total  : {len(rows)} leads scored and saved")
    print(f"{'─'*55}")
    print(f"  A Hot  (85-100) : {grades['A']:3d}  ← call immediately")
    print(f"  B Warm (65-84)  : {grades['B']:3d}  ← follow up today")
    print(f"  C Cool (45-64)  : {grades['C']:3d}  ← nurture sequence")
    print(f"  D Cold (25-44)  : {grades['D']:3d}  ← low priority")
    print(f"  F Dead (0-24)   : {grades['F']:3d}  ← skip")
    print(f"{'─'*55}")

if __name__ == "__main__":
    main()
