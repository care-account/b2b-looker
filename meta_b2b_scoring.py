#!/usr/bin/env python3
"""
Complete Meta Ads B2B Lead Campaign Scoring System
Uses Grok AI for scoring - GitHub Actions + BigQuery (Free Tier)
"""

import os
import requests
from datetime import datetime
from google.cloud import bigquery

# Configuration (set in GitHub Secrets)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_URL = "https://api.x.ai/v1/chat/completions"

# Validate required env vars
if not GCP_PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID environment variable is required")
if not META_ACCESS_TOKEN:
    raise ValueError("META_ACCESS_TOKEN environment variable is required")
if not META_AD_ACCOUNT_ID:
    raise ValueError("META_AD_ACCOUNT_ID environment variable is required")
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY environment variable is required")

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
    except:
        dataset = bigquery.Dataset(dataset_id)
        bq.create_dataset(dataset)

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
    except:
        table = bigquery.Table(table_id, schema=schema)
        bq.create_table(table)
    return table_id

def fetch_meta_ads():
    """Extract all campaigns with pagination and lifetime insights."""
    all_campaigns = []
    url = f"https://graph.facebook.com/v19.0/{META_AD_ACCOUNT_ID}/campaigns"
    params = {
        "access_token": META_ACCESS_TOKEN,
        "fields": "id,name,status,insights.metric(leads,spend,impressions,clicks,ctr,cpm,cpp).period(lifetime)",
        "limit": 100,
        "filtering": '[{"field":"status","operator":"IN","value":["ACTIVE","PAUSED"]}]'
    }

    while url:
        response = requests.get(url, params=params)
        data = response.json()

        # Debug: print any API error
        if "error" in data:
            print(f"Meta API error: {data['error']}")
            return {"data": []}

        campaigns = data.get("data", [])
        all_campaigns.extend(campaigns)

        # Get next page URL from paging
        url = data.get("paging", {}).get("next")
        # For subsequent requests, params are already in the 'next' URL
        if url:
            params = None   # next URL already contains all params

    return {"data": all_campaigns}

def get_grok_score(campaign_data):
    """Get AI-powered score from Grok."""
    # Guard against missing keys
    name = campaign_data.get('name', 'Unknown')
    leads = campaign_data.get('leads', 0)
    spend = campaign_data.get('spend', 0.0)
    impressions = campaign_data.get('impressions', 0)
    clicks = campaign_data.get('clicks', 0)
    ctr = campaign_data.get('ctr', 0.0)
    cpl = campaign_data.get('cost_per_lead', 0.0)

    prompt = f"""Score this Meta Ads B2B lead campaign 0-100:
    Campaign: {name}
    Leads: {leads}
    Spend: ${spend:.2f}
    Impressions: {impressions}
    Clicks: {clicks}
    CTR: {ctr:.2%}
    CPL: ${cpl:.2f}

    B2B scoring criteria:
    - High lead count + low CPL = high score
    - Good CTR indicates relevance
    - Return only integer 0-100.
    """

    try:
        response = requests.post(
            GROK_URL,
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.1
            },
            timeout=30
        )
        return int(response.json()["choices"][0]["message"]["content"].strip())
    except Exception as e:
        print(f"Grok error: {e}")
        return 50  # Default score

def grade_from_score(score):
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"

def load_to_bigquery(meta_data, table_id):
    """Load campaign data to BigQuery with Grok scores."""
    get_bigquery_client()
    rows = []

    for campaign in meta_data.get("data", []):
        insights = campaign.get("insights", {}).get("data", [{}])[0]

        cpl = 0.0
        leads = insights.get("leads", 0)
        spend = insights.get("spend", 0.0)
        if leads > 0:
            cpl = spend / leads

        campaign_data = {
            "name": campaign.get("name", ""),
            "leads": leads,
            "spend": spend,
            "impressions": insights.get("impressions", 0),
            "clicks": insights.get("clicks", 0),
            "ctr": insights.get("ctr", 0.0),
            "cost_per_lead": cpl
        }

        score = get_grok_score(campaign_data)

        row = {
            "campaign_id": campaign["id"],
            "campaign_name": campaign.get("name", ""),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "spend": float(spend),
            "impressions": int(insights.get("impressions", 0)),
            "clicks": int(insights.get("clicks", 0)),
            "leads": int(leads),
            "cost_per_lead": float(cpl),
            "ctr": float(insights.get("ctr", 0.0)),
            "grok_score": float(score),
            "grade": grade_from_score(score),
            "created_at": datetime.utcnow().isoformat()
        }
        rows.append(row)
        print(f"Processed: {campaign.get('name')} -> Score: {score}")

    # ✅ Guard against empty rows
    if not rows:
        print("No campaigns to load. Skipping BigQuery insert.")
        return 0

    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        print(f"BigQuery errors: {errors}")
    return len(rows)

def main():
    print("Setting up BigQuery...")
    table_id = setup_bigquery()

    print("Fetching Meta Ads data...")
    meta_data = fetch_meta_ads()
    print(f"Found {len(meta_data.get('data', []))} campaigns")

    print("Scoring with Grok AI and loading to BigQuery...")
    count = load_to_bigquery(meta_data, table_id)
    print(f"Loaded {count} campaigns")

if __name__ == "__main__":
    main()
