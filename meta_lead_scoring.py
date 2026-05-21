#!/usr/bin/env python3
"""
FABO B2B Lead Scorer — Individual Store Opener Prediction
- Fetches leads from all Lead Forms in the ad account (requires leads_retrieval)
- Incremental: only new leads since last scored lead (or LEADS_START_DATE)
- Scores each person with Groq AI and loads to BigQuery
"""

import os
import re
import time
import requests
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# ── Config (GitHub Secrets) ───────────────────────────────────────────────────
GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET  = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
GROQ_API_KEY      = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"

LEADS_START_DATE  = os.getenv("LEADS_START_DATE", "2024-01-01")

_raw = os.getenv("META_AD_ACCOUNT_ID", "")
META_AD_ACCOUNT_ID = f"act_{_raw}" if _raw and not _raw.startswith("act_") else _raw

for var, val in [("GCP_PROJECT_ID", GCP_PROJECT_ID), ("META_ACCESS_TOKEN", META_ACCESS_TOKEN),
                 ("META_AD_ACCOUNT_ID", META_AD_ACCOUNT_ID), ("GROQ_API_KEY", GROQ_API_KEY)]:
    if not val:
        raise ValueError(f"{var} environment variable is required")

print(f"Using Ad Account : {META_AD_ACCOUNT_ID}")
print(f"Historical start : {LEADS_START_DATE}")

# ── BigQuery helpers ──────────────────────────────────────────────────────────
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

def get_already_scored_lead_ids(table_id):
    get_bq()
    query = f"SELECT lead_id FROM `{table_id}`"
    try:
        result = bq.query(query).result()
        ids = {row.lead_id for row in result}
        print(f"Already scored in BigQuery: {len(ids)} leads (will skip these)")
        return ids
    except Exception as e:
        print(f"Could not query existing leads (table may be empty): {e}")
        return set()

def get_fetch_since_timestamp(table_id):
    """Return timestamp to fetch leads FROM (incremental or full backfill)."""
    get_bq()
    query = f"SELECT MAX(lead_created_time) AS latest FROM `{table_id}`"
    try:
        result = list(bq.query(query).result())
        latest = result[0].latest if result else None
        if latest:
            since = latest + timedelta(seconds=1)
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            print(f"Incremental mode: fetching leads submitted after {since_str} (UTC)")
            return since_str
    except Exception as e:
        print(f"Could not determine latest timestamp: {e}")

    since_str = f"{LEADS_START_DATE}T00:00:00+00:00"
    print(f"Full backfill mode: fetching all leads from {LEADS_START_DATE} to today")
    return since_str

# ── Meta API helpers ──────────────────────────────────────────────────────────
def iso_to_unix(iso_str):
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp())

def paginate(url, params):
    results = []
    while url:
        r = requests.get(url, params=params if params else {}).json()
        if "error" in r:
            return results, r["error"]["message"]
        results.extend(r.get("data", []))
        url = r.get("paging", {}).get("next")
        params = None
    return results, None

def fetch_all_lead_forms():
    """Fetch all Lead Generation forms from the ad account."""
    base = "https://graph.facebook.com/v19.0"
    url = f"{base}/{META_AD_ACCOUNT_ID}/leadgen_forms"
    params = {
        "access_token": META_ACCESS_TOKEN,
        "fields": "id,name,status,created_time",
        "limit": 100
    }
    forms, err = paginate(url, params)
    if err:
        print(f"[META] Form fetch error: {err}")
        print("[META] Tip: make sure your META_ACCESS_TOKEN secret uses the newly generated token with leads_retrieval permission.")
        return []
    active = [f for f in forms if f.get("status") != "ARCHIVED"]
    print(f"  Total lead forms: {len(forms)}, active: {len(active)}")
    return active

def fetch_leads_from_form(form_id, form_name, since_ts, until_ts):
    """Fetch leads from a specific form using /{form_id}/leads."""
    base = "https://graph.facebook.com/v19.0"
    url = f"{base}/{form_id}/leads"
    until_dt = datetime.fromisoformat(until_ts.replace("Z", "+00:00"))

    params = {
        "access_token": META_ACCESS_TOKEN,
        "fields": "id,created_time,field_data,platform,ad_name,ad_id,campaign_id,campaign_name",
        "filtering": f'[{{"field":"time_created","operator":"GREATER_THAN","value":{iso_to_unix(since_ts)}}}]',
        "limit": 100,
    }

    leads = []
    while url:
        r = requests.get(url, params=params if params else {}).json()
        if "error" in r:
            if r["error"].get("code") != 100:  # 100 = no leads
                print(f"    Form {form_id} error: {r['error']['message']}")
            break

        for lead in r.get("data", []):
            try:
                created = datetime.fromisoformat(lead["created_time"].replace("Z", "+00:00"))
                if created > until_dt:
                    continue
            except Exception:
                pass
            lead["form_id"]   = form_id
            lead["form_name"] = form_name
            leads.append(lead)

        url = r.get("paging", {}).get("next")
        params = None

    return leads

def parse_lead_fields(lead):
    """Convert Meta's field_data array into a clean dict."""
    raw = {}
    for f in lead.get("field_data", []):
        values = f.get("values", [])
        raw[f["name"]] = values[0] if values else ""

    def g(*keys):
        for k in keys:
            v = raw.get(k, "").strip()
            if v:
                return v
        return ""

    return {
        "lead_id":          lead.get("id", ""),
        "lead_name":        g("full_name", "name", "first_name"),
        "phone":            g("phone_number", "phone").replace("p:", ""),
        "email":            g("email"),
        "city":             g("city"),
        "state":            g("additional_col1_select", "state", "province"),
        "platform":         lead.get("platform", ""),
        "investment_ready": g("additional_col6_select", "investment_readiness",
                               "are_you_ready_to_invest", "ready_to_invest"),
        "timeline":         g("additional_col3_select", "timeline",
                               "when_are_you_planning_to_start"),
        "earning_intent":   g("what_type_of_earning_opportunity_are_you_looking_for?",
                               "earning_type", "opportunity_type"),
        "campaign_id":      lead.get("campaign_id", ""),
        "campaign_name":    lead.get("campaign_name", ""),
        "ad_name":          lead.get("ad_name", ""),
        "form_name":        lead.get("form_name", ""),
        "lead_created_time": lead.get("created_time", ""),
    }

# ── AI Scoring ────────────────────────────────────────────────────────────────
INVESTMENT_SCORE = {
    "yes, i'm ready to invest":      40,
    "yes, i\u2019m ready to invest": 40,
    "i may need financing options":   20,
    "just exploring":                  5,
}
TIMELINE_SCORE = {
    "within 1\u20133 months": 35,
    "within 3\u20136 months": 20,
    "just exploring":           5,
}
INTENT_SCORE = {
    "high_commission_per_successful_closure": 25,
    "side_income":                            10,
    "just_exploring":                          2,
}

def precompute_signal_scores(lead):
    inv_s = INVESTMENT_SCORE.get(lead["investment_ready"].lower(), 5)
    tl_s  = TIMELINE_SCORE.get(lead["timeline"].lower(), 5)
    ei_s  = INTENT_SCORE.get(lead["earning_intent"].lower(), 2)
    return inv_s, tl_s, ei_s

def get_ai_score(lead):
    inv_s, tl_s, ei_s = precompute_signal_scores(lead)
    rule_total = inv_s + tl_s + ei_s

    inv_label = {40:"Ready to invest",20:"Needs financing",5:"Just exploring"}.get(inv_s,"Unknown")
    tl_label  = {35:"1-3 months",20:"3-6 months",5:"Just exploring"}.get(tl_s,"Unknown")
    ei_label  = {25:"Wants franchise ownership",10:"Wants side income",2:"Just exploring"}.get(ei_s,"Unknown")

    prompt = f"""You are an expert B2B franchise sales analyst for FABO, an Indian laundry franchise brand.
Your task: predict the probability (0-100) that this lead will actually open a FABO store.

LEAD PROFILE:
Name: {lead['lead_name'] or 'Unknown'}
City: {lead['city'] or 'Unknown'}, {lead['state'] or 'Unknown'}
Platform: {lead['platform'] or 'Unknown'}
Campaign: {lead['campaign_name'] or 'Unknown'}

FORM RESPONSES:
Investment readiness : "{lead['investment_ready']}" → {inv_label} [{inv_s}/40 pts]
Decision timeline    : "{lead['timeline']}" → {tl_label} [{tl_s}/35 pts]
Earning intent       : "{lead['earning_intent']}" → {ei_label} [{ei_s}/25 pts]
Rule-based baseline  : {rule_total}/100

SCORING GUIDE:
- Ready to invest + 1-3 months + franchise ownership → 85-100
- Ready to invest + 3-6 months or side income intent → 70-84
- Needs financing + 1-3 months + franchise ownership → 55-75
- Needs financing + longer timeline or side income   → 40-60
- Just exploring on investment OR weak intent        → 20-45
- Just exploring on all three signals                → 5-20
- Empty/missing fields reduce score by 5-10 pts

Return ONLY a single integer 0-100. No text. No explanation.
Score:"""

    for attempt in range(3):
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0.0
                },
                timeout=30
            )
            if r.status_code == 429:
                wait = 2 * (attempt + 1)
                print(f"  Rate limited — retrying in {wait}s...")
                time.sleep(wait)
                continue
            if not r.ok:
                print(f"  Groq {r.status_code}: {r.text[:120]}")
                return rule_total
            raw = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\b(\d{1,3})\b", raw)
            if m:
                return max(0, min(100, int(m.group(1))))
            print(f"  Unexpected response: '{raw}' — using rule score {rule_total}")
            return rule_total
        except Exception as e:
            print(f"  Groq error: {e}")
            return rule_total
    return rule_total

def grade_from_score(score):
    if score >= 85: return "A"
    if score >= 65: return "B"
    if score >= 45: return "C"
    if score >= 25: return "D"
    return "F"

# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    now_utc = datetime.now(timezone.utc)

    print("=" * 55)
    print(f"Run started : {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 55)

    print("\nSetting up BigQuery...")
    table_id = setup_bigquery()

    since_ts = get_fetch_since_timestamp(table_id)
    until_ts = now_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    print(f"Fetch window : {since_ts}  →  {until_ts}")

    already_scored = get_already_scored_lead_ids(table_id)

    print("\nFetching lead forms from Meta...")
    forms = fetch_all_lead_forms()
    if not forms:
        print("No lead forms found. Check token permissions (need leads_retrieval).")
        return

    all_leads = []
    for form in forms:
        print(f"  Form: {form['name']} ({form['id']})")
        raw_leads = fetch_leads_from_form(form["id"], form["name"], since_ts, until_ts)
        parsed = [parse_lead_fields(l) for l in raw_leads]
        new = [l for l in parsed if l["lead_id"] not in already_scored]
        print(f"    → {len(raw_leads)} fetched, {len(new)} new")
        all_leads.extend(new)

    print(f"\nNew leads to score : {len(all_leads)}")
    if not all_leads:
        print("Nothing new to score — BigQuery is up to date.")
        return

    print("\nScoring with Groq AI...")
    rows = []
    for i, lead in enumerate(all_leads, 1):
        score = get_ai_score(lead)
        grade = grade_from_score(score)
        time.sleep(2.1)

        ts = lead["lead_created_time"]
        try:
            ts_clean = datetime.fromisoformat(
                ts.replace("Z", "+00:00")
            ).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            ts_clean = now_utc.strftime("%Y-%m-%dT%H:%M:%S")

        row = {
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
            "grade":             grade,
            "scored_at":         now_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rows.append(row)
        print(f"[{i:3d}/{len(all_leads)}] {(lead['lead_name'] or 'Unknown'):25s} | "
              f"{(lead['city'] or '?'):15s} | {score:3d} ({grade}) | "
              f"{lead['investment_ready'][:28]}")

    print(f"\nInserting {len(rows)} rows into BigQuery...")
    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        print(f"BigQuery errors: {errors}")
    else:
        print(f"Successfully loaded {len(rows)} lead scores → {table_id}")

    grades = {"A":0,"B":0,"C":0,"D":0,"F":0}
    for r in rows:
        grades[r["grade"]] += 1

    print(f"\n{'─'*50}")
    print(f"  Run date    : {now_utc.strftime('%Y-%m-%d')}")
    print(f"  Window      : {since_ts[:10]} → {until_ts[:10]}")
    print(f"  Total new   : {len(rows)}")
    print(f"{'─'*50}")
    print(f"  A Hot  (85-100) : {grades['A']:3d}  ← call immediately")
    print(f"  B Warm (65-84)  : {grades['B']:3d}  ← follow up today")
    print(f"  C Cool (45-64)  : {grades['C']:3d}  ← nurture sequence")
    print(f"  D Cold (25-44)  : {grades['D']:3d}  ← low priority")
    print(f"  F Dead (0-24)   : {grades['F']:3d}  ← skip")
    print(f"{'─'*50}")

if __name__ == "__main__":
    main()
