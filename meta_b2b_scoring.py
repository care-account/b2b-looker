#!/usr/bin/env python3
"""
Complete Meta Ads B2B Lead Campaign Scoring System
Uses Grok AI for scoring - GitHub Actions + BigQuery (Free Tier)
"""

import os
import time
import re
import requests
from datetime import datetime
from google.cloud import bigquery

# Configuration (set in GitHub Secrets)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_URL = "https://api.groq.com/openai/v1/chat/completions"  # FIX: Groq (groq.com), not xAI Grok

# FIX 1: Auto-add "act_" prefix if missing
_raw_account_id = os.getenv("META_AD_ACCOUNT_ID", "")
if _raw_account_id and not _raw_account_id.startswith("act_"):
    META_AD_ACCOUNT_ID = f"act_{_raw_account_id}"
else:
    META_AD_ACCOUNT_ID = _raw_account_id

# Validate required env vars
if not GCP_PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID environment variable is required")
if not META_ACCESS_TOKEN:
    raise ValueError("META_ACCESS_TOKEN environment variable is required")
if not META_AD_ACCOUNT_ID:
    raise ValueError("META_AD_ACCOUNT_ID environment variable is required")
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY environment variable is required")

print(f"Using Ad Account ID: {META_AD_ACCOUNT_ID}")

def validate_meta_token():
    """
    Diagnostic: verify the token is valid and list all accessible ad accounts.
    This runs before any campaign fetch so misconfigurations are caught early.
    """
    base = "https://graph.facebook.com/v19.0"

    # 1. Check token identity
    me = requests.get(f"{base}/me", params={"access_token": META_ACCESS_TOKEN}).json()
    if "error" in me:
        print(f"[META DIAGNOSTIC] Token is INVALID or EXPIRED: {me['error']['message']}")
        return False
    print(f"[META DIAGNOSTIC] Token is valid. Authenticated as: {me.get('name', me.get('id', 'unknown'))}")

    # 2. List all ad accounts this token can access
    accounts_resp = requests.get(
        f"{base}/me/adaccounts",
        params={"access_token": META_ACCESS_TOKEN, "fields": "id,name,account_status"}
    ).json()

    if "error" in accounts_resp:
        print(f"[META DIAGNOSTIC] Cannot list ad accounts: {accounts_resp['error']['message']}")
        print("[META DIAGNOSTIC] Token may be missing 'ads_read' permission.")
        return False

    accounts = accounts_resp.get("data", [])
    if not accounts:
        print("[META DIAGNOSTIC] Token has access to 0 ad accounts. Check Business Manager permissions.")
        return False

    print(f"[META DIAGNOSTIC] Token can access {len(accounts)} ad account(s):")
    found = False
    for acc in accounts:
        status_map = {1: "ACTIVE", 2: "DISABLED", 3: "UNSETTLED", 7: "PENDING_RISK_REVIEW", 9: "IN_GRACE_PERIOD"}
        status = status_map.get(acc.get("account_status"), f"STATUS_{acc.get('account_status')}")
        marker = " <-- THIS ONE" if acc["id"] == META_AD_ACCOUNT_ID else ""
        print(f"  - {acc['id']} | {acc.get('name', 'N/A')} | {status}{marker}")
        if acc["id"] == META_AD_ACCOUNT_ID:
            found = True

    if not found:
        print(f"[META DIAGNOSTIC] WARNING: '{META_AD_ACCOUNT_ID}' is NOT in the accessible accounts list above.")
        print("[META DIAGNOSTIC] Update the META_AD_ACCOUNT_ID secret to one of the IDs listed above.")
        return False

    print(f"[META DIAGNOSTIC] Account {META_AD_ACCOUNT_ID} confirmed accessible. Proceeding...")
    return True

bq = None

def get_bigquery_client():
    global bq
    if bq is None:
        bq = bigquery.Client(project=GCP_PROJECT_ID)
    return bq

def setup_bigquery():
    get_bigquery_client()
    dataset_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}"
    try:
        bq.get_dataset(dataset_id)
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        bq.create_dataset(dataset)
        print(f"Created dataset {dataset_id}.")

    schema = [
        bigquery.SchemaField("campaign_id", "STRING"),
        bigquery.SchemaField("campaign_name", "STRING"),
        bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("spend", "FLOAT"),
        bigquery.SchemaField("impressions", "INTEGER"),
        bigquery.SchemaField("clicks", "INTEGER"),
        bigquery.SchemaField("leads", "INTEGER"),
        bigquery.SchemaField("cost_per_lead", "FLOAT"),
        bigquery.SchemaField("ctr", "FLOAT"),
        bigquery.SchemaField("grok_score", "FLOAT"),
        bigquery.SchemaField("grade", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
    ]

    table_id = f"{dataset_id}.campaign_scores"
    try:
        bq.get_table(table_id)
        print(f"Table {table_id} already exists.")
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        bq.create_table(table)
        print(f"Created table {table_id}.")
    return table_id

def fetch_meta_ads():
    """Fetch all campaigns with pagination and lifetime insights."""
    all_campaigns = []
    url = f"https://graph.facebook.com/v19.0/{META_AD_ACCOUNT_ID}/campaigns"
    params = {
        "access_token": META_ACCESS_TOKEN,
        # FIX 2: "leads" is only available for Lead Generation objective campaigns.
        # Using "actions" covers all conversion types including lead form submissions.
        # "maximum" is the correct date_preset for all-time/lifetime data.
        "fields": (
            "id,name,status,"
            "insights.date_preset(maximum){"
            "spend,impressions,clicks,ctr,cpm,cpp,actions"
            "}"
        ),
        "limit": 100
    }

    while url:
        response = requests.get(url, params=params if params else {})
        data = response.json()

        if "error" in data:
            print(f"Meta API error: {data['error']}")
            return {"data": []}

        campaigns = data.get("data", [])
        all_campaigns.extend(campaigns)

        # Get next page URL
        url = data.get("paging", {}).get("next")
        if url:
            params = None  # next URL already contains all params

    return {"data": all_campaigns}

def extract_leads_from_actions(actions):
    """
    FIX 3: Extract lead count from the 'actions' array returned by Meta API.
    Lead Generation campaigns report leads under action types like
    'lead', 'onsite_conversion.lead_grouped', or 'offsite_conversion.fb_pixel_lead'.
    """
    if not actions:
        return 0
    lead_action_types = {
        "lead",
        "onsite_conversion.lead_grouped",
        "offsite_conversion.fb_pixel_lead",
    }
    total_leads = 0
    for action in actions:
        if action.get("action_type") in lead_action_types:
            total_leads += int(float(action.get("value", 0)))
    return total_leads

def get_grok_score(campaign_data):
    """Get AI-powered score from Groq (0-100) with rate-limit retry."""
    import time, re

    name = campaign_data.get('name', 'Unknown')
    leads = campaign_data.get('leads', 0)
    spend = campaign_data.get('spend', 0.0)
    impressions = campaign_data.get('impressions', 0)
    clicks = campaign_data.get('clicks', 0)
    ctr = campaign_data.get('ctr', 0.0)
    cpl = campaign_data.get('cost_per_lead', 0.0)

    # Strict prompt: no explanation, no fractions, just a plain integer
    prompt = (
        f"You are a B2B ad campaign scorer. "
        f"Reply with a SINGLE integer between 0 and 100. No text, no explanation, no fraction. "
        f"Score based on: high leads + low cost-per-lead = high score; zero leads = low score.\n\n"
        f"Campaign: {name}\n"
        f"Leads: {leads}\n"
        f"Spend: ${spend:.2f}\n"
        f"Impressions: {impressions}\n"
        f"Clicks: {clicks}\n"
        f"CTR: {ctr:.2f}%\n"
        f"CPL: ${cpl:.2f}\n\n"
        f"Score (integer 0-100):"
    )

    for attempt in range(3):
        try:
            response = requests.post(
                GROK_URL,
                headers={
                    "Authorization": f"Bearer {GROK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0.0
                },
                timeout=30
            )

            # Rate limit: wait and retry
            if response.status_code == 429:
                wait = 2 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            if not response.ok:
                print(f"Groq HTTP {response.status_code}: {response.text}")
                return 50

            raw = response.json()["choices"][0]["message"]["content"].strip()

            # Match a 1-3 digit number (whole response)
            match = re.search(r"\b(\d{1,3})\b", raw)
            if match:
                score = int(match.group(1))
                return max(0, min(100, score))
            else:
                print(f"  Unexpected Groq response: '{raw}' — defaulting to 50")
                return 50

        except Exception as e:
            print(f"Groq error: {e}")
            return 50

    print("  All retries exhausted — defaulting to 50")
    return 50

def grade_from_score(score):
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"

def load_to_bigquery(meta_data, table_id):
    """Score campaigns and load to BigQuery."""
    get_bigquery_client()
    rows = []

    for campaign in meta_data.get("data", []):
        insights_list = campaign.get("insights", {}).get("data", [])
        insights = insights_list[0] if insights_list else {}

        # FIX 4: Safe type casting — Meta API returns numeric fields as strings
        spend = float(insights.get("spend", 0) or 0)
        impressions = int(insights.get("impressions", 0) or 0)
        clicks = int(insights.get("clicks", 0) or 0)
        ctr = float(insights.get("ctr", 0) or 0)

        # FIX 3: Use actions array to count leads
        actions = insights.get("actions", [])
        leads = extract_leads_from_actions(actions)

        cpl = (spend / leads) if leads > 0 else 0.0

        campaign_data = {
            "name": campaign.get("name", ""),
            "leads": leads,
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "cost_per_lead": cpl
        }

        score = get_grok_score(campaign_data)
        time.sleep(2.1)  # Stay under 30 RPM free tier limit (2s between calls)

        row = {
            "campaign_id": campaign["id"],
            "campaign_name": campaign.get("name", ""),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "leads": leads,
            "cost_per_lead": cpl,
            "ctr": ctr,
            "grok_score": float(score),
            "grade": grade_from_score(score),
            "created_at": datetime.utcnow().isoformat()
        }
        rows.append(row)
        print(f"Processed: {campaign.get('name')} | Leads: {leads} | Spend: ${spend:.2f} | Score: {score} ({grade_from_score(score)})")

    if not rows:
        print("No campaigns to load. Skipping BigQuery insert.")
        return 0

    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        print(f"BigQuery insert errors: {errors}")
    else:
        print(f"Successfully inserted {len(rows)} rows into BigQuery.")
    return len(rows)

def main():
    print("Setting up BigQuery...")
    table_id = setup_bigquery()

    print("Validating Meta token and account access...")
    if not validate_meta_token():
        print("Aborting: fix the Meta token/account ID issues above before proceeding.")
        return

    print("Fetching Meta Ads data...")
    meta_data = fetch_meta_ads()
    campaign_count = len(meta_data.get("data", []))
    print(f"Found {campaign_count} campaigns")

    print("Scoring with Grok AI and loading to BigQuery...")
    count = load_to_bigquery(meta_data, table_id)
    print(f"Loaded {count} campaigns")

if __name__ == "__main__":
    main()
