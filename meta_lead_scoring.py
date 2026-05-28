#!/usr/bin/env python3
"""
FABO B2B Lead Scorer — Store Opener Prediction
Route: ad_account -> OUTCOME_LEADS campaigns -> ACTIVE ads -> leads
No leadgen_forms endpoint needed. Works with ads_read permission.
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
LEADS_START_DATE  = os.getenv("LEADS_START_DATE", "2026-04-01")

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
        print(f"Already in BigQuery: {len(ids)} leads (will skip)")
        return ids
    except Exception as e:
        print(f"Could not query existing leads: {e}")
        return set()

MAX_WINDOW_DAYS = 7  # Never scan more than 7 days back on any run

def get_since_timestamp(table_id):
    """Return the since-timestamp for this run, capped at MAX_WINDOW_DAYS ago."""
    get_bq()
    now = datetime.now(timezone.utc)
    hard_floor = now - timedelta(days=MAX_WINDOW_DAYS)

    try:
        rows = list(bq.query(f"SELECT MAX(lead_created_time) AS latest FROM `{table_id}`").result())
        latest = rows[0].latest if rows else None
        if latest:
            # latest may be a naive datetime from BQ — ensure it's UTC-aware
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=timezone.utc)
            since = latest + timedelta(seconds=1)
            # Cap: never go further back than MAX_WINDOW_DAYS
            if since < hard_floor:
                print(f"Incremental: last record was {latest.date()} — capping window to {MAX_WINDOW_DAYS} days")
                since = hard_floor
            else:
                print(f"Incremental: fetching leads after {since.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
            return since.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except Exception as e:
        print(f"Timestamp query failed: {e}")

    # No data in BQ yet — still cap at 7 days (don't do a full backfill on daily runs)
    since = hard_floor
    print(f"No prior data — starting from {MAX_WINDOW_DAYS}-day window: {since.date()}")
    return since.strftime("%Y-%m-%dT%H:%M:%S+00:00")

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

def fetch_active_lead_ads():
    """
    Fetch ALL lead gen ads (active + inactive) so no leads within the 7-day
    window are missed. campaign -> ads route (no leadgen_forms needed).
    """
    base = "https://graph.facebook.com/v19.0"

    # Fetch ALL campaigns regardless of status — paused/archived campaigns still hold leads
    campaigns, err = meta_paginate(
        f"{base}/{META_AD_ACCOUNT_ID}/campaigns",
        {"access_token": META_ACCESS_TOKEN,
         "fields": "id,name,objective,status,effective_status",
         "limit": 100}
    )
    if err:
        print(f"[META] Campaign error: {err}")
        return []

    LEAD_OBJ = {"LEAD_GENERATION", "OUTCOME_LEADS"}
    lead_camps = [c for c in campaigns if c.get("objective") in LEAD_OBJ]
    all_obj    = sorted({c.get("objective", "?") for c in campaigns})
    print(f"  Campaigns: {len(campaigns)} total, {len(lead_camps)} lead gen")
    print(f"  Objectives found: {all_obj}")

    if not lead_camps:
        print("  No lead gen campaigns found.")
        return []

    all_ads = []
    for camp in lead_camps:
        time.sleep(0.3)
        # Fetch ALL ads (all statuses) — inactive ads still hold historical leads
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

    active_count   = sum(1 for a in all_ads if a.get("effective_status") == "ACTIVE")
    inactive_count = len(all_ads) - active_count
    print(f"  Total lead ads: {len(all_ads)} ({active_count} active, {inactive_count} inactive/paused)")
    return all_ads

def fetch_leads_from_ad(ad, since_ts, until_ts):
    """Fetch leads from one ad using ads_read permission."""
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
                if code != 100:  # 100 = no leads, expected
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

# ── AI Scoring ────────────────────────────────────────────────────────────────
INV_SCORE = {"yes, i'm ready to invest":40,"yes, i\u2019m ready to invest":40,
             "i may need financing options":20,"just exploring":5}
TL_SCORE  = {"within 1\u20133 months":35,"within 3\u20136 months":20,"just exploring":5}
ET_SCORE  = {"high_commission_per_successful_closure":25,"side_income":10,"just_exploring":2}

def get_ai_score(lead):
    inv_s = INV_SCORE.get(lead["investment_ready"].lower(), 5)
    tl_s  = TL_SCORE.get(lead["timeline"].lower(), 5)
    et_s  = ET_SCORE.get(lead["earning_intent"].lower(), 2)
    base  = inv_s + tl_s + et_s

    inv_l = {40:"Ready to invest",20:"Needs financing",5:"Exploring"}.get(inv_s,"?")
    tl_l  = {35:"1-3 months",20:"3-6 months",5:"Exploring"}.get(tl_s,"?")
    et_l  = {25:"Franchise owner",10:"Side income",2:"Exploring"}.get(et_s,"?")

    prompt = (
        f"You are a B2B franchise analyst for FABO India. "
        f"Score 0-100 likelihood this lead will open a FABO store.\n\n"
        f"Name: {lead['lead_name'] or 'Unknown'}\n"
        f"City: {lead['city'] or '?'}, {lead['state'] or '?'}\n"
        f"Investment: \"{lead['investment_ready']}\" → {inv_l} [{inv_s}/40]\n"
        f"Timeline: \"{lead['timeline']}\" → {tl_l} [{tl_s}/35]\n"
        f"Intent: \"{lead['earning_intent']}\" → {et_l} [{et_s}/25]\n"
        f"Rule baseline: {base}/100\n\n"
        f"Guide: ready+1-3mo+franchise=85-100 | ready+longer=70-84 | "
        f"financing+1-3mo=55-75 | financing+longer=40-60 | "
        f"exploring=5-45 | missing fields=-10\n\n"
        f"Return ONLY a single integer 0-100:\nScore:"
    )
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
                time.sleep(4 * (attempt+1)); continue
            if not r.ok:
                return base
            raw = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\b(\d{1,3})\b", raw)
            return max(0, min(100, int(m.group(1)))) if m else base
        except Exception:
            return base
    return base

def grade(s):
    return "A" if s>=85 else "B" if s>=65 else "C" if s>=45 else "D" if s>=25 else "F"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    print("="*55)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*55)

    print("\nSetting up BigQuery...")
    table_id = setup_bigquery()

    print("\nValidating Meta token...")
    if not validate_token():
        return

    since_ts = get_since_timestamp(table_id)
    until_ts = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    print(f"Window: {since_ts[:10]} → {until_ts[:10]}")

    already = get_already_scored_ids(table_id)

    print("\nFetching active lead gen ads...")
    ads = fetch_active_lead_ads()
    if not ads:
        print("No active lead gen ads found.")
        return

    print(f"\nFetching leads from {len(ads)} ads...")
    all_leads = []
    for ad in ads:
        raw = fetch_leads_from_ad(ad, since_ts, until_ts)
        parsed = [parse_lead(l) for l in raw]
        new    = [l for l in parsed if l["lead_id"] not in already]
        if new:
            print(f"  '{ad['name']}': {len(raw)} fetched, {len(new)} new")
        all_leads.extend(new)

    print(f"\nNew leads to score: {len(all_leads)}")
    if not all_leads:
        print("Nothing new — BigQuery is up to date.")
        return

    print("\nScoring with Groq AI...")
    rows = []
    total_saved = 0
    for i, lead in enumerate(all_leads, 1):
        score = get_ai_score(lead)
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
        print(f"[{i:3d}/{len(all_leads)}] {(lead['lead_name'] or '?'):25s} | "
              f"{(lead['city'] or '?'):15s} | {score:3d} ({g}) | "
              f"{lead['investment_ready'][:28]}")

        # Save every 50 leads so data is never lost on timeout
        if i % 50 == 0:
            errs = bq.insert_rows_json(table_id, rows[total_saved:i])
            if errs:
                print(f"  [BQ] Batch error: {errs}")
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

    print(f"\n{'─'*50}")
    print(f"  Window   : {since_ts[:10]} → {until_ts[:10]}")
    print(f"  Total    : {len(rows)} leads scored and saved")
    print(f"{'─'*50}")
    print(f"  A Hot  (85-100) : {grades['A']:3d}  ← call immediately")
    print(f"  B Warm (65-84)  : {grades['B']:3d}  ← follow up today")
    print(f"  C Cool (45-64)  : {grades['C']:3d}  ← nurture sequence")
    print(f"  D Cold (25-44)  : {grades['D']:3d}  ← low priority")
    print(f"  F Dead (0-24)   : {grades['F']:3d}  ← skip")
    print(f"{'─'*50}")

if __name__ == "__main__":
    main()
