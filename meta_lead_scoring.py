
"""
FABO B2B Lead Scorer — Store Opener Prediction
- Specify exact date range: LEADS_START_DATE + LEADS_END_DATE
- No date specified: auto last 7 days
- Fetches ALL ads (active, paused, inactive, archived)
- AI scoring with rule-based fallback
- Saves to BigQuery every 50 leads (safe against timeouts)
"""

import os, re, time, requests
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET  = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
GROQ_API_KEY      = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"

# Date range control
# Set LEADS_START_DATE + LEADS_END_DATE for an exact range (e.g. 2026-01-01 to 2026-01-31)
# Leave both blank → auto last 7 days
LEADS_START_DATE = os.getenv("LEADS_START_DATE", "")   # e.g. "2024-01-01"
LEADS_END_DATE   = os.getenv("LEADS_END_DATE", "")     # e.g. "2026-05-31" (blank = now)

_raw = os.getenv("META_AD_ACCOUNT_ID", "")
META_AD_ACCOUNT_ID = f"act_{_raw}" if _raw and not _raw.startswith("act_") else _raw

for var, val in [("GCP_PROJECT_ID", GCP_PROJECT_ID),
                 ("META_ACCESS_TOKEN", META_ACCESS_TOKEN),
                 ("META_AD_ACCOUNT_ID", META_AD_ACCOUNT_ID),
                 ("GROK_API_KEY / GROQ_API_KEY", GROQ_API_KEY)]:
    if not val:
        raise ValueError(f"{var} is required")

print(f"Ad Account      : {META_AD_ACCOUNT_ID}")
print(f"LEADS_START_DATE: {LEADS_START_DATE or '(not set — will use last 7 days)'}")
print(f"LEADS_END_DATE  : {LEADS_END_DATE   or '(not set — will use now)'}")

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

    schema = [
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
        bigquery.SchemaField("store_open_score",   "FLOAT"),
        bigquery.SchemaField("grade",              "STRING"),
        bigquery.SchemaField("scored_at",          "TIMESTAMP"),
    ]
    table_id = f"{dataset_id}.lead_scores"
    try:
        bq.get_table(table_id)
        print(f"Table {table_id} already exists.")
    except Exception:
        bq.create_table(bigquery.Table(table_id, schema=schema))
        print(f"Created table {table_id}")
    return table_id

def get_already_scored_ids(table_id):
    get_bq()
    try:
        ids = {row.lead_id for row in bq.query(f"SELECT lead_id FROM `{table_id}`").result()}
        print(f"Already in BigQuery: {len(ids)} leads (will skip duplicates)")
        return ids
    except Exception as e:
        print(f"Could not query existing leads: {e}")
        return set()

def resolve_window(now_utc):
    """
    Determine since_ts and until_ts based on env vars.

    Rules:
      LEADS_START_DATE set, LEADS_END_DATE set   → exact range [start, end]
      LEADS_START_DATE set, LEADS_END_DATE blank  → [start, now]
      LEADS_START_DATE blank                      → last 7 days [now-7d, now]
    """
    if LEADS_START_DATE:
        since_ts = f"{LEADS_START_DATE}T00:00:00+00:00"
        if LEADS_END_DATE:
            until_ts = f"{LEADS_END_DATE}T23:59:59+00:00"
            mode = f"Custom range: {LEADS_START_DATE} → {LEADS_END_DATE}"
        else:
            until_ts = now_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            mode = f"Start date to now: {LEADS_START_DATE} → today"
    else:
        seven_days_ago = now_utc - timedelta(days=7)
        since_ts = seven_days_ago.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        until_ts = now_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        mode = "Auto: last 7 days"

    return since_ts, until_ts, mode

# ── Meta API ──────────────────────────────────────────────────────────────────
def iso_to_unix(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())

def meta_paginate(url, params):
    """Paginate Meta Graph API with rate-limit retry."""
    results = []
    while url:
        for attempt in range(4):
            r = requests.get(url, params=params if params else {}).json()
            if "error" in r:
                msg  = r["error"]["message"]
                code = r["error"].get("code", 0)
                if code in (17, 80004) or "too many calls" in msg.lower():
                    wait = 30 * (attempt + 1)
                    print(f"  [RATE LIMIT] waiting {wait}s (attempt {attempt+1}/4)...")
                    time.sleep(wait)
                    continue
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
        print(f"[META] Token invalid: {me['error']['message']}")
        return False
    print(f"[META] Token OK — {me.get('name', me.get('id'))}")
    return True

def fetch_all_lead_ads():
    """
    Fetch ALL ads from ALL lead gen campaigns regardless of status.
    Covers: ACTIVE, PAUSED, INACTIVE, ARCHIVED, DELETED.
    This ensures historical leads from old campaigns are always fetched.
    """
    base = "https://graph.facebook.com/v19.0"

    campaigns, err = meta_paginate(
        f"{base}/{META_AD_ACCOUNT_ID}/campaigns",
        {"access_token": META_ACCESS_TOKEN,
         "fields": "id,name,objective,status",
         "limit": 100}
    )
    if err:
        print(f"[META] Campaign error: {err}")
        return []

    LEAD_OBJ = {"LEAD_GENERATION", "OUTCOME_LEADS"}
    lead_camps = [c for c in campaigns if c.get("objective") in LEAD_OBJ]
    all_obj    = sorted({c.get("objective", "?") for c in campaigns})
    print(f"  Campaigns: {len(campaigns)} total, {len(lead_camps)} lead gen")
    print(f"  Objectives: {all_obj}")
    print(f"  Campaign statuses: {sorted({c.get('status','?') for c in lead_camps})}")

    all_ads = []
    for camp in lead_camps:
        time.sleep(0.3)
        # No status filter — get ALL ads (active, paused, inactive, archived)
        ads, err = meta_paginate(
            f"{base}/{camp['id']}/ads",
            {"access_token": META_ACCESS_TOKEN,
             "fields": "id,name,effective_status",
             "limit": 100}
        )
        if err:
            print(f"  Campaign '{camp['name']}' ads error: {err}")
            continue
        for ad in ads:
            ad["_campaign_id"]   = camp["id"]
            ad["_campaign_name"] = camp["name"]
        all_ads.extend(ads)

    statuses = sorted({a.get("effective_status", "?") for a in all_ads})
    print(f"  Total ads fetched: {len(all_ads)}")
    print(f"  Ad statuses found: {statuses}")
    return all_ads

def fetch_leads_from_ad(ad, since_ts, until_ts):
    """Fetch leads from one ad within the date window."""
    base     = "https://graph.facebook.com/v19.0"
    until_dt = datetime.fromisoformat(until_ts.replace("Z", "+00:00"))
    url      = f"{base}/{ad['id']}/leads"
    params   = {
        "access_token": META_ACCESS_TOKEN,
        "fields": "id,created_time,field_data,platform,campaign_id,campaign_name,ad_name",
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
                    time.sleep(15 * (attempt + 1))
                    continue
                if code != 100:
                    print(f"    '{ad.get('name','?')}' error: {msg}")
                return leads
            break
        else:
            return leads

        for lead in r.get("data", []):
            try:
                if datetime.fromisoformat(
                        lead["created_time"].replace("Z", "+00:00")) > until_dt:
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

def parse_lead(lead):
    raw = {f["name"]: (f["values"][0] if f.get("values") else "")
           for f in lead.get("field_data", [])}
    def g(*keys):
        for k in keys:
            v = raw.get(k, "").strip()
            if v: return v
        return ""
    return {
        "lead_id":           lead.get("id", ""),
        "lead_name":         g("full_name","name","first_name"),
        "phone":             g("phone_number","phone").replace("p:",""),
        "email":             g("email"),
        "city":              g("city"),
        "state":             g("additional_col1_select","state","province"),
        "platform":          lead.get("platform",""),
        "investment_ready":  g("additional_col6_select","investment_readiness","are_you_ready_to_invest"),
        "timeline":          g("additional_col3_select","timeline","when_are_you_planning_to_start"),
        "earning_intent":    g("what_type_of_earning_opportunity_are_you_looking_for?","earning_type"),
        "campaign_id":       lead.get("_campaign_id","") or lead.get("campaign_id",""),
        "campaign_name":     lead.get("_campaign_name","") or lead.get("campaign_name",""),
        "ad_name":           lead.get("_ad_name","") or lead.get("ad_name",""),
        "form_name":         "",
        "lead_created_time": lead.get("created_time",""),
    }

# ── Scoring ───────────────────────────────────────────────────────────────────
INV_SCORE = {"yes, i'm ready to invest":40, "yes, i\u2019m ready to invest":40,
             "yes, i's ready to invest":40,
             "i may need financing options":20, "just exploring":5}
TL_SCORE  = {"within 1\u20133 months":35, "within 3\u20136 months":20, "just exploring":5}
ET_SCORE  = {"high_commission_per_successful_closure":25, "side_income":10, "just_exploring":2}

REGION_MAP = {
    "assam":       ("Northeast India — high volume, lower conversion", -4),
    "kerala":      ("Kerala — cautious but reliable once committed",   +2),
    "maharashtra": ("Maharashtra — metro, serious investors",          +4),
    "bihar":       ("Bihar — price-sensitive, high volume",           -3),
    "mp":          ("Madhya Pradesh — tier 2/3, value seekers",       -3),
    "telangana":   ("Telangana/Hyderabad — strong B2B ecosystem",     +3),
    "ap":          ("Andhra Pradesh — similar to Telangana",          +2),
    "rajasthan":   ("Rajasthan — family business culture",             0),
    "gujarat":     ("Gujarat — entrepreneurial, serious investors",    +5),
    "west bengal": ("West Bengal — large market, moderate conversion", -3),
    "bengal":      ("West Bengal — large market, moderate conversion", -3),
    "jharkhand":   ("Jharkhand — emerging market",                    -3),
    "ncr":         ("NCR/Delhi — metro, high intent",                 +4),
    "delhi":       ("Delhi — metro, high intent",                     +4),
    "chandigarh":  ("Chandigarh/Punjab — high income buyers",         +4),
    "punjab":      ("Punjab — strong business culture",               +3),
    "uttarakhand": ("Uttarakhand — smaller market",                   -1),
    "chattisgarh": ("Chhattisgarh — tier 2 market",                  -2),
    "chhattisgarh":("Chhattisgarh — tier 2 market",                  -2),
    "odisha":      ("Odisha — emerging B2B market",                   -1),
}

def get_score(lead):
    inv_s = INV_SCORE.get(lead["investment_ready"].lower(), 5)
    tl_s  = TL_SCORE.get(lead["timeline"].lower(), 5)
    et_s  = ET_SCORE.get(lead["earning_intent"].lower(), 2)
    rule_total = inv_s + tl_s + et_s

    has_city  = bool(lead["city"]      and lead["city"].strip()      and lead["city"] != "?")
    has_state = bool(lead["state"]     and lead["state"].strip())
    has_phone = bool(lead["phone"]     and lead["phone"].strip())
    has_email = bool(lead["email"]     and lead["email"].strip())
    has_name  = bool(lead["lead_name"] and lead["lead_name"].strip())
    completeness = sum([has_city, has_state, has_phone, has_email, has_name])

    platform = (lead["platform"] or "").lower()
    platform_note = {"ig":"Instagram — younger, impulse-driven",
                     "fb":"Facebook — deliberate, age 30–55"}.get(platform, "Platform unknown")

    campaign = (lead["campaign_name"] or "").lower()
    region_note, region_mod = "Region unknown", -5
    for kw, (note, mod) in REGION_MAP.items():
        if kw in campaign:
            region_note, region_mod = note, mod
            break

    inv_l = {40:"Ready to invest immediately",
             20:"Needs financing — interested but cautious",
             5:"Just exploring — no commitment"}.get(inv_s, "Unknown")
    tl_l  = {35:"Within 1–3 months — high urgency",
             20:"Within 3–6 months — moderate urgency",
             5:"Just exploring — no timeline"}.get(tl_s, "Unknown")
    et_l  = {25:"Wants franchise/master franchise ownership",
             10:"Wants side income — less committed",
             2:"Just exploring opportunities"}.get(et_s, "Unknown")

    comp_note = (f"{completeness}/5 complete "
                 f"(name={'✓' if has_name else '✗'} "
                 f"phone={'✓' if has_phone else '✗'} "
                 f"email={'✓' if has_email else '✗'} "
                 f"city={'✓' if has_city else '✗'} "
                 f"state={'✓' if has_state else '✗'})")

    prompt = f"""You are a B2B franchise sales analyst for FABO India (premium laundry franchise).
Score 0–100: probability this lead will open a FABO store.

LEAD:
Name: {lead['lead_name'] or 'Unknown'} | City: {lead['city'] or '?'}, {lead['state'] or '?'}
Platform: {platform.upper() or '?'} ({platform_note})
Campaign: {lead['campaign_name'] or '?'} | Region: {region_note}
Data: {comp_note}

FORM ANSWERS:
Investment: "{lead['investment_ready'] or '[blank]'}" → {inv_l} [{inv_s}/40]
Timeline:   "{lead['timeline'] or '[blank]'}" → {tl_l} [{tl_s}/35]
Intent:     "{lead['earning_intent'] or '[blank]'}" → {et_l} [{et_s}/25]
Rule total: {rule_total}/100

SCORING (apply each modifier):
Investment base:
  Ready+1-3mo+franchise=85-100 | Ready+weaker=65-84
  Financing+1-3mo+franchise=50-70 | Financing+weak=30-50
  Exploring=5-35
Region modifier: {region_mod:+d} pts ({region_note})
Platform: FB=+2, IG=-2, unknown=-3
Completeness: 5/5=+5, 4/5=+2, 3/5=0, 2/5=-5, 0-1/5=-10
Calibration: 90-100=extremely rare | 80-89=strong | 65-79=warm | 45-64=nurture | 25-44=cold | 0-24=skip

Single integer 0-100 only:"""

    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model":"llama-3.1-8b-instant",
                      "messages":[{"role":"user","content":prompt}],
                      "max_tokens":5,"temperature":0.0},
                timeout=30)
            if r.status_code == 429:
                time.sleep(4*(attempt+1)); continue
            if not r.ok:
                return rule_total
            raw_resp = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\b(\d{1,3})\b", raw_resp)
            return max(0, min(100, int(m.group(1)))) if m else rule_total
        except Exception:
            return rule_total
    return rule_total

def grade(s):
    return "A" if s>=85 else "B" if s>=65 else "C" if s>=45 else "D" if s>=25 else "F"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    print("="*60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*60)

    print("\nSetting up BigQuery...")
    table_id = setup_bigquery()

    print("\nValidating Meta token...")
    if not validate_token():
        return

    since_ts, until_ts, mode = resolve_window(now)
    print(f"\nDate window : {since_ts[:10]} → {until_ts[:10]}")
    print(f"Mode        : {mode}")

    already = get_already_scored_ids(table_id)

    print("\nFetching ALL lead gen ads (active + paused + inactive + archived)...")
    ads = fetch_all_lead_ads()
    if not ads:
        print("No lead gen ads found.")
        return

    print(f"\nFetching leads from {len(ads)} ads in window {since_ts[:10]} → {until_ts[:10]}...")
    all_leads = []
    for ad in ads:
        raw    = fetch_leads_from_ad(ad, since_ts, until_ts)
        parsed = [parse_lead(l) for l in raw]
        new    = [l for l in parsed if l["lead_id"] not in already]
        if new:
            print(f"  '{ad['name']}' [{ad.get('effective_status','?')}]: "
                  f"{len(raw)} fetched, {len(new)} new")
        all_leads.extend(new)

    print(f"\nNew leads to score : {len(all_leads)}")
    if not all_leads:
        print("Nothing new in this window — BigQuery is up to date.")
        return

    print("\nScoring with Groq AI...")
    rows        = []
    total_saved = 0

    for i, lead in enumerate(all_leads, 1):
        score = get_score(lead)
        g     = grade(score)
        ts    = lead["lead_created_time"]
        try:
            ts_clean = datetime.fromisoformat(ts.replace("Z","+00:00")).strftime("%Y-%m-%dT%H:%M:%S")
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
            "investment_ready":  lead["investment_ready"],
            "timeline":          lead["timeline"],
            "earning_intent":    lead["earning_intent"],
            "campaign_id":       lead["campaign_id"],
            "campaign_name":     lead["campaign_name"],
            "ad_name":           lead["ad_name"],
            "form_name":         lead["form_name"],
            "lead_created_time": ts_clean,
            "store_open_score":  float(score),
            "grade":             g,
            "scored_at":         now.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[{i:4d}/{len(all_leads)}] {(lead['lead_name'] or '?'):25s} | "
              f"{(lead['city'] or '?'):15s} | {score:3d} ({g}) | "
              f"{lead['investment_ready'][:28]}")

        # Flush to BigQuery every 50 leads — never lose data on timeout
        if i % 50 == 0:
            errs = bq.insert_rows_json(table_id, rows[total_saved:i])
            if errs:
                print(f"  [BQ] Batch error at row {i}: {errs}")
            else:
                print(f"  [BQ] ✓ Saved rows {total_saved+1}–{i}")
            total_saved = i

    # Final flush
    if len(rows) > total_saved:
        errs = bq.insert_rows_json(table_id, rows[total_saved:])
        if errs:
            print(f"[BQ] Final batch error: {errs}")
        else:
            print(f"[BQ] ✓ Saved final {len(rows)-total_saved} rows")

    grades = {"A":0,"B":0,"C":0,"D":0,"F":0}
    for r in rows:
        grades[r["grade"]] += 1

    print(f"\n{'─'*60}")
    print(f"  Window   : {since_ts[:10]} → {until_ts[:10]}")
    print(f"  Mode     : {mode}")
    print(f"  Total    : {len(rows)} leads scored and saved")
    print(f"{'─'*60}")
    print(f"  A Hot  (85-100) : {grades['A']:4d}  ← call immediately")
    print(f"  B Warm (65-84)  : {grades['B']:4d}  ← follow up today")
    print(f"  C Cool (45-64)  : {grades['C']:4d}  ← nurture sequence")
    print(f"  D Cold (25-44)  : {grades['D']:4d}  ← low priority")
    print(f"  F Dead (0-24)   : {grades['F']:4d}  ← skip")
    print(f"{'─'*60}")

if __name__ == "__main__":
    main()
