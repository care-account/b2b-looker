#!/usr/bin/env python3
"""
FABO B2B Lead Scorer — Individual Store Opener Prediction
Fetches leads via leadgen_forms (with leads_retrieval permission)
Incremental: only new leads since last run. Full backfill on first run.
Scores each lead with Groq AI and stores to BigQuery.
"""

import os
import re
import time
import requests
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET  = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
GROQ_API_KEY      = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"
LEADS_START_DATE  = os.getenv("LEADS_START_DATE", "2026-01-01")
GRAPH_VER         = "v19.0"
BASE              = f"https://graph.facebook.com/{GRAPH_VER}"

_raw = os.getenv("META_AD_ACCOUNT_ID", "")
META_AD_ACCOUNT_ID = f"act_{_raw}" if _raw and not _raw.startswith("act_") else _raw

for var, val in [
    ("GCP_PROJECT_ID",   GCP_PROJECT_ID),
    ("META_ACCESS_TOKEN",META_ACCESS_TOKEN),
    ("META_AD_ACCOUNT_ID",META_AD_ACCOUNT_ID),
    ("GROK_API_KEY",     GROQ_API_KEY),
]:
    if not val:
        raise ValueError(f"{var} environment variable is required")

print(f"Ad Account      : {META_AD_ACCOUNT_ID}")
print(f"Start date      : {LEADS_START_DATE}")

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
        print(f"  Created dataset {dataset_id}")

    # FIX: added inv_score, tl_score, et_score columns for score breakdown
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
        bigquery.SchemaField("inv_score",          "INTEGER"),   # /40
        bigquery.SchemaField("tl_score",           "INTEGER"),   # /35
        bigquery.SchemaField("et_score",           "INTEGER"),   # /25
        bigquery.SchemaField("store_open_score",   "FLOAT"),     # /100
        bigquery.SchemaField("grade",              "STRING"),
        bigquery.SchemaField("scored_at",          "TIMESTAMP"),
    ]

    table_id = f"{dataset_id}.lead_scores"
    try:
        existing = bq.get_table(table_id)
        # FIX: add missing columns to existing table if needed
        existing_cols = {f.name for f in existing.schema}
        new_fields = [f for f in schema if f.name not in existing_cols]
        if new_fields:
            existing.schema = list(existing.schema) + new_fields
            bq.update_table(existing, ["schema"])
            print(f"  Added {len(new_fields)} new columns to existing table")
        else:
            print(f"  Table {table_id} up to date.")
    except Exception:
        bq.create_table(bigquery.Table(table_id, schema=schema))
        print(f"  Created table {table_id}")
    return table_id

def get_already_scored_ids(table_id):
    get_bq()
    try:
        rows = bq.query(f"SELECT lead_id FROM `{table_id}`").result()
        ids = {row.lead_id for row in rows}
        print(f"  Already in BigQuery: {len(ids)} leads (will skip duplicates)")
        return ids
    except Exception as e:
        print(f"  Could not query existing IDs: {e}")
        return set()

def get_since_timestamp(table_id):
    """Incremental: use MAX timestamp from BQ. First run: use LEADS_START_DATE."""
    get_bq()
    try:
        rows = list(bq.query(
            f"SELECT MAX(lead_created_time) AS latest FROM `{table_id}`"
        ).result())
        latest = rows[0].latest if rows else None
        if latest:
            since = latest + timedelta(seconds=1)
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            print(f"  Incremental mode — fetching leads after {since_str}")
            return since_str
    except Exception as e:
        print(f"  Timestamp query failed: {e}")

    since_str = f"{LEADS_START_DATE}T00:00:00+00:00"
    print(f"  Full backfill — fetching from {LEADS_START_DATE} to today")
    return since_str

# ── Meta helpers ──────────────────────────────────────────────────────────────
def iso_to_unix(iso_str):
    return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())

def meta_paginate(url, params):
    """
    Paginate Meta Graph API with rate-limit retry.
    Returns (results_list, error_string_or_None)
    """
    results = []
    while url:
        for attempt in range(4):
            try:
                r = requests.get(url, params=params if params else {}, timeout=30).json()
            except Exception as e:
                return results, str(e)

            if "error" in r:
                msg  = r["error"].get("message", "")
                code = r["error"].get("code", 0)
                # Rate limit codes: 17, 80004, or message contains "too many"
                if code in (17, 80004) or "too many" in msg.lower():
                    wait = 30 * (attempt + 1)
                    print(f"    Rate limited — waiting {wait}s (attempt {attempt+1}/4)")
                    time.sleep(wait)
                    continue
                return results, msg
            break
        else:
            return results, "Rate limit: all retries exhausted"

        results.extend(r.get("data", []))
        url    = r.get("paging", {}).get("next")
        params = None
        time.sleep(0.3)   # gentle inter-page pause
    return results, None

def validate_token():
    me = requests.get(f"{BASE}/me",
        params={"access_token": META_ACCESS_TOKEN}, timeout=15).json()
    if "error" in me:
        print(f"  Token INVALID: {me['error']['message']}")
        return False
    print(f"  Token OK — authenticated as: {me.get('name', me.get('id'))}")
    return True

# ── FIX: dual fetch strategy ──────────────────────────────────────────────────
# Strategy 1: /leadgen_forms  (needs leads_retrieval — token has it now)
# Strategy 2: campaigns → ads → /leads  (fallback, needs only ads_read)
# We try both and merge, deduplicated by lead id.

def fetch_via_forms(since_ts, until_ts):
    """Fetch all leads via leadgen_forms endpoint."""
    until_dt = datetime.fromisoformat(until_ts.replace("Z", "+00:00"))

    forms, err = meta_paginate(
        f"{BASE}/{META_AD_ACCOUNT_ID}/leadgen_forms",
        {"access_token": META_ACCESS_TOKEN,
         "fields": "id,name,status", "limit": 100}
    )
    if err:
        print(f"  [forms] Error: {err}")
        return {}

    active = [f for f in forms if f.get("status") != "ARCHIVED"]
    print(f"  [forms] {len(active)} active forms found")

    leads_by_id = {}
    for form in active:
        page_leads, err = meta_paginate(
            f"{BASE}/{form['id']}/leads",
            {
                "access_token": META_ACCESS_TOKEN,
                "fields": "id,created_time,field_data,platform,"
                          "ad_id,ad_name,campaign_id,campaign_name",
                "filtering": f'[{{"field":"time_created","operator":"GREATER_THAN",'
                             f'"value":{iso_to_unix(since_ts)}}}]',
                "limit": 100,
            }
        )
        if err and "100" not in str(err):
            print(f"    Form '{form['name']}' error: {err}")
            continue

        for lead in page_leads:
            try:
                if datetime.fromisoformat(
                    lead["created_time"].replace("Z", "+00:00")) > until_dt:
                    continue
            except Exception:
                pass
            lead["_form_name"] = form["name"]
            leads_by_id[lead["id"]] = lead

        print(f"    Form '{form['name']}': {len(page_leads)} leads")

    return leads_by_id

def fetch_via_ads(since_ts, until_ts):
    """Fetch leads via campaign → ads → /leads (fallback route)."""
    until_dt = datetime.fromisoformat(until_ts.replace("Z", "+00:00"))

    campaigns, err = meta_paginate(
        f"{BASE}/{META_AD_ACCOUNT_ID}/campaigns",
        {"access_token": META_ACCESS_TOKEN,
         "fields": "id,name,objective,status", "limit": 100}
    )
    if err:
        print(f"  [ads] Campaign error: {err}")
        return {}

    # Accept all objective names (Meta keeps renaming them)
    lead_objectives = {"LEAD_GENERATION", "OUTCOME_LEADS"}
    lead_camps = [c for c in campaigns if c.get("objective") in lead_objectives]
    if not lead_camps:
        lead_camps = campaigns   # last resort: try all
    print(f"  [ads] {len(lead_camps)} campaigns to check")

    leads_by_id = {}
    for camp in lead_camps:
        ads, err = meta_paginate(
            f"{BASE}/{camp['id']}/ads",
            {"access_token": META_ACCESS_TOKEN,
             "fields": "id,name,status", "limit": 100}
        )
        if err:
            continue
        for ad in ads:
            page_leads, err = meta_paginate(
                f"{BASE}/{ad['id']}/leads",
                {
                    "access_token": META_ACCESS_TOKEN,
                    "fields": "id,created_time,field_data,platform,"
                              "campaign_id,campaign_name,ad_name",
                    "filtering": f'[{{"field":"time_created","operator":"GREATER_THAN",'
                                 f'"value":{iso_to_unix(since_ts)}}}]',
                    "limit": 100,
                }
            )
            if err and "100" not in str(err):
                continue
            for lead in page_leads:
                try:
                    if datetime.fromisoformat(
                        lead["created_time"].replace("Z", "+00:00")) > until_dt:
                        continue
                except Exception:
                    pass
                lead["_form_name"]    = ""
                lead["_campaign_id"]  = camp["id"]
                lead["_campaign_name"]= camp["name"]
                lead["_ad_name"]      = ad.get("name", "")
                leads_by_id[lead["id"]] = lead

    print(f"  [ads] {len(leads_by_id)} leads found via ads route")
    return leads_by_id

def parse_lead(lead):
    """Convert Meta field_data array to clean dict."""
    raw = {}
    for f in lead.get("field_data", []):
        vals = f.get("values", [])
        raw[f["name"]] = vals[0].strip() if vals else ""

    def g(*keys):
        for k in keys:
            v = raw.get(k, "").strip()
            if v: return v
        return ""

    return {
        "lead_id":           lead.get("id", ""),
        "lead_name":         g("full_name", "name", "first_name"),
        "phone":             g("phone_number", "phone").replace("p:", ""),
        "email":             g("email"),
        "city":              g("city"),
        "state":             g("additional_col1_select", "state", "province"),
        "platform":          lead.get("platform", ""),
        "investment_ready":  g("additional_col6_select", "investment_readiness",
                               "are_you_ready_to_invest"),
        "timeline":          g("additional_col3_select", "timeline",
                               "when_are_you_planning_to_start"),
        "earning_intent":    g("what_type_of_earning_opportunity_are_you_looking_for?",
                               "earning_type", "opportunity_type"),
        "campaign_id":       lead.get("campaign_id","") or lead.get("_campaign_id",""),
        "campaign_name":     lead.get("campaign_name","") or lead.get("_campaign_name",""),
        "ad_name":           lead.get("ad_name","") or lead.get("_ad_name",""),
        "form_name":         lead.get("_form_name",""),
        "lead_created_time": lead.get("created_time",""),
    }

# ── Scoring ───────────────────────────────────────────────────────────────────
INVEST_SCORE = {
    "yes, i'm ready to invest":      40,
    "yes, i\u2019m ready to invest": 40,
    "i may need financing options":   20,
    "just exploring":                  5,
}
TIME_SCORE = {
    "within 1\u20133 months": 35,
    "within 1-3 months":      35,
    "within 3\u20136 months": 20,
    "within 3-6 months":      20,
    "just exploring":           5,
}
INTENT_SCORE = {
    "high_commission_per_successful_closure": 25,
    "side_income":                            10,
    "just_exploring":                          2,
}

def rule_scores(lead):
    inv = INVEST_SCORE.get(lead["investment_ready"].lower(), 5)
    tl  = TIME_SCORE.get(lead["timeline"].lower(), 5)
    ei  = INTENT_SCORE.get(lead["earning_intent"].lower(), 2)
    return inv, tl, ei

def groq_score(lead):
    inv_s, tl_s, ei_s = rule_scores(lead)
    baseline = inv_s + tl_s + ei_s

    prompt = f"""You are a B2B franchise sales analyst for FABO (Indian laundry franchise).
Score 0-100: likelihood this lead will open a FABO store.

Name: {lead['lead_name'] or 'Unknown'} | City: {lead['city']}, {lead['state']} | Platform: {lead['platform']}
Campaign: {lead['campaign_name']}

Form answers:
- Investment: "{lead['investment_ready']}" [{inv_s}/40]
- Timeline:   "{lead['timeline']}" [{tl_s}/35]
- Intent:     "{lead['earning_intent']}" [{ei_s}/25]
- Baseline:   {baseline}/100

Guide: Ready+1-3mo+franchise→85-100 | Ready+longer/side→70-84 | Financing+1-3mo+franchise→55-75
       Financing+weak→40-60 | Exploring investment→20-45 | All exploring→5-20

Reply with ONE integer only (0-100):"""

    for attempt in range(3):
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 5, "temperature": 0.0},
                timeout=30
            )
            if r.status_code == 429:
                wait = 2 * (attempt + 1)
                print(f"    Groq rate limit — waiting {wait}s")
                time.sleep(wait)
                continue
            if not r.ok:
                print(f"    Groq {r.status_code}: {r.text[:80]}")
                return baseline, inv_s, tl_s, ei_s
            raw = r.json()["choices"][0]["message"]["content"].strip()
            m   = re.search(r"\b(\d{1,3})\b", raw)
            score = max(0, min(100, int(m.group(1)))) if m else baseline
            return score, inv_s, tl_s, ei_s
        except Exception as e:
            print(f"    Groq error: {e}")
            return baseline, inv_s, tl_s, ei_s
    return baseline, inv_s, tl_s, ei_s

def grade(score):
    return ("A" if score >= 85 else "B" if score >= 65 else
            "C" if score >= 45 else "D" if score >= 25 else "F")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    print("=" * 56)
    print(f"  Run started: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 56)

    print("\n[1] BigQuery setup")
    table_id = setup_bigquery()

    print("\n[2] Validate Meta token")
    if not validate_token():
        raise SystemExit("Invalid token — aborting")

    since_ts = get_since_timestamp(table_id)
    until_ts = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    print(f"  Window: {since_ts[:19]}  →  {until_ts[:19]}")

    already = get_already_scored_ids(table_id)

    print("\n[3] Fetch leads from Meta")
    # Try forms route first (best data quality — includes form_name)
    print("  Trying /leadgen_forms route...")
    leads_by_id = fetch_via_forms(since_ts, until_ts)

    # Supplement with ads route (catches anything the forms route missed)
    print("  Supplementing with campaign→ads→leads route...")
    ads_leads = fetch_via_ads(since_ts, until_ts)
    before = len(leads_by_id)
    for lid, lead in ads_leads.items():
        if lid not in leads_by_id:
            leads_by_id[lid] = lead
    print(f"  Combined: {len(leads_by_id)} unique leads "
          f"({len(leads_by_id)-before} added by ads route)")

    # Parse and deduplicate against BigQuery
    all_leads = [parse_lead(l) for l in leads_by_id.values()
                 if l.get("id") not in already]
    # Sort by created_time ascending so BQ gets chronological data
    all_leads.sort(key=lambda x: x["lead_created_time"])

    print(f"\n[4] Score {len(all_leads)} new leads")
    if not all_leads:
        print("  Nothing new — BigQuery is already up to date.")
        return

    rows = []
    for i, lead in enumerate(all_leads, 1):
        score, inv_s, tl_s, ei_s = groq_score(lead)
        g = grade(score)
        time.sleep(2.1)   # Groq free tier: 30 RPM

        ts = lead["lead_created_time"]
        try:
            ts_clean = datetime.fromisoformat(
                ts.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M:%S")
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
            "inv_score":         inv_s,
            "tl_score":          tl_s,
            "et_score":          ei_s,
            "store_open_score":  float(score),
            "grade":             g,
            "scored_at":         now.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"  [{i:3d}/{len(all_leads)}] {(lead['lead_name'] or '?'):22s} | "
              f"{(lead['city'] or '?'):13s} | {score:3d} {g} | "
              f"{lead['investment_ready'][:25]}")

    print(f"\n[5] Insert {len(rows)} rows into BigQuery...")
    # FIX: insert in batches of 500 (BQ streaming limit)
    batch_size = 500
    total_errors = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        errs  = bq.insert_rows_json(table_id, batch)
        if errs:
            total_errors.extend(errs)
            print(f"  Batch {start//batch_size+1} errors: {errs[:2]}")
        else:
            print(f"  Batch {start//batch_size+1}: {len(batch)} rows inserted OK")

    if total_errors:
        print(f"\n  WARNING: {len(total_errors)} row(s) had insert errors")
    else:
        print(f"  All {len(rows)} rows inserted successfully → {table_id}")

    # Summary
    grade_counts = {"A":0,"B":0,"C":0,"D":0,"F":0}
    for row in rows:
        grade_counts[row["grade"]] += 1

    print(f"\n{'─'*54}")
    print(f"  Date range : {since_ts[:10]}  →  {until_ts[:10]}")
    print(f"  Total leads: {len(rows)}")
    print(f"{'─'*54}")
    print(f"  A Hot  85-100 : {grade_counts['A']:3d}  ← CALL IMMEDIATELY")
    print(f"  B Warm 65-84  : {grade_counts['B']:3d}  ← Follow up today")
    print(f"  C Cool 45-64  : {grade_counts['C']:3d}  ← Nurture sequence")
    print(f"  D Cold 25-44  : {grade_counts['D']:3d}  ← Low priority")
    print(f"  F Dead  0-24  : {grade_counts['F']:3d}  ← Skip")
    print(f"{'─'*54}")

if __name__ == "__main__":
    main()
