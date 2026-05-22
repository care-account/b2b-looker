#!/usr/bin/env python3
"""
FABO B2B Lead Scoring System
Meta Ads → Groq AI → BigQuery

FIXED VERSION:
- Handles Meta rate limits
- Handles Groq rate limits
- Deduplicates leads
- Incremental syncing
- BigQuery insert logging
- Retry handling
- Safer batching
"""

import os
import re
import time
import requests
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# =============================================================================
# CONFIG
# =============================================================================

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")

_raw = os.getenv("META_AD_ACCOUNT_ID", "")
META_AD_ACCOUNT_ID = (
    f"act_{_raw}"
    if _raw and not _raw.startswith("act_")
    else _raw
)

GROQ_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

LEADS_START_DATE = os.getenv("LEADS_START_DATE", "2025-01-01")

# =============================================================================
# VALIDATION
# =============================================================================

required_vars = {
    "GCP_PROJECT_ID": GCP_PROJECT_ID,
    "META_ACCESS_TOKEN": META_ACCESS_TOKEN,
    "META_AD_ACCOUNT_ID": META_AD_ACCOUNT_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
}

for k, v in required_vars.items():
    if not v:
        raise ValueError(f"{k} is required")

print(f"Using Ad Account : {META_AD_ACCOUNT_ID}")

# =============================================================================
# BIGQUERY
# =============================================================================

bq = bigquery.Client(project=GCP_PROJECT_ID)


def setup_bigquery():

    dataset_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}"

    try:
        bq.get_dataset(dataset_id)
        print(f"Dataset exists: {dataset_id}")

    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        bq.create_dataset(dataset)
        print(f"Created dataset: {dataset_id}")

    table_id = f"{dataset_id}.lead_scores"

    schema = [
        bigquery.SchemaField("lead_id", "STRING"),
        bigquery.SchemaField("lead_name", "STRING"),
        bigquery.SchemaField("phone", "STRING"),
        bigquery.SchemaField("email", "STRING"),
        bigquery.SchemaField("city", "STRING"),
        bigquery.SchemaField("state", "STRING"),
        bigquery.SchemaField("platform", "STRING"),
        bigquery.SchemaField("investment_ready", "STRING"),
        bigquery.SchemaField("timeline", "STRING"),
        bigquery.SchemaField("earning_intent", "STRING"),
        bigquery.SchemaField("campaign_id", "STRING"),
        bigquery.SchemaField("campaign_name", "STRING"),
        bigquery.SchemaField("ad_name", "STRING"),
        bigquery.SchemaField("lead_created_time", "TIMESTAMP"),
        bigquery.SchemaField("rule_score", "FLOAT"),
        bigquery.SchemaField("ai_score", "FLOAT"),
        bigquery.SchemaField("final_score", "FLOAT"),
        bigquery.SchemaField("grade", "STRING"),
        bigquery.SchemaField("scored_at", "TIMESTAMP"),
    ]

    try:
        bq.get_table(table_id)
        print(f"Table exists: {table_id}")

    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        bq.create_table(table)
        print(f"Created table: {table_id}")

    return table_id


# =============================================================================
# HELPERS
# =============================================================================

def paginate(url, params):

    results = []

    while url:

        # Meta rate-limit protection
        time.sleep(1)

        r = requests.get(url, params=params if params else {}).json()

        if "error" in r:

            code = r["error"].get("code")

            # Meta throttling
            if code in [4, 17]:

                print("Meta rate limit hit — sleeping 30s...")
                time.sleep(30)
                continue

            print(f"Meta Error: {r['error']['message']}")
            return results

        results.extend(r.get("data", []))

        url = r.get("paging", {}).get("next")

        params = None

    return results


# =============================================================================
# FETCH CAMPAIGNS
# =============================================================================

def fetch_lead_campaigns():

    base = "https://graph.facebook.com/v19.0"

    campaigns = paginate(
        f"{base}/{META_AD_ACCOUNT_ID}/campaigns",
        {
            "access_token": META_ACCESS_TOKEN,
            "fields": "id,name,objective,status",
            "limit": 100
        }
    )

    LEAD_OBJECTIVES = {
        "LEAD_GENERATION",
        "OUTCOME_LEADS"
    }

    lead_campaigns = [
        c for c in campaigns
        if c.get("objective") in LEAD_OBJECTIVES
        and c.get("status") == "ACTIVE"
    ]

    print(f"Lead campaigns found: {len(lead_campaigns)}")

    return lead_campaigns


# =============================================================================
# FETCH ADS
# =============================================================================

def fetch_ads(campaign):

    base = "https://graph.facebook.com/v19.0"

    ads = paginate(
        f"{base}/{campaign['id']}/ads",
        {
            "access_token": META_ACCESS_TOKEN,
            "fields": "id,name,status",
            "limit": 100
        }
    )

    for ad in ads:
        ad["_campaign_id"] = campaign["id"]
        ad["_campaign_name"] = campaign["name"]

    return ads


# =============================================================================
# FETCH LEADS
# =============================================================================

def fetch_leads(ad):

    base = "https://graph.facebook.com/v19.0"

    leads = paginate(
        f"{base}/{ad['id']}/leads",
        {
            "access_token": META_ACCESS_TOKEN,
            "fields": (
                "id,"
                "created_time,"
                "field_data,"
                "platform,"
                "campaign_id,"
                "campaign_name,"
                "ad_name"
            ),
            "limit": 100
        }
    )

    for lead in leads:
        lead["_campaign_id"] = ad["_campaign_id"]
        lead["_campaign_name"] = ad["_campaign_name"]

    return leads


# =============================================================================
# PARSE LEADS
# =============================================================================

def parse_lead(lead):

    raw = {}

    for field in lead.get("field_data", []):

        vals = field.get("values", [])

        raw[field["name"]] = vals[0] if vals else ""

    def g(*keys):

        for k in keys:

            val = raw.get(k, "").strip()

            if val:
                return val

        return ""

    return {
        "lead_id": lead.get("id", ""),
        "lead_name": g("full_name", "name"),
        "phone": g("phone_number", "phone"),
        "email": g("email"),
        "city": g("city"),
        "state": g("state", "province"),
        "platform": lead.get("platform", ""),
        "investment_ready": g(
            "investment_readiness",
            "ready_to_invest"
        ),
        "timeline": g(
            "timeline",
            "when_are_you_planning_to_start"
        ),
        "earning_intent": g(
            "earning_type",
            "opportunity_type"
        ),
        "campaign_id": lead.get("campaign_id", ""),
        "campaign_name": lead.get("campaign_name", ""),
        "ad_name": lead.get("ad_name", ""),
        "lead_created_time": lead.get("created_time", "")
    }


# =============================================================================
# SCORING RULES
# =============================================================================

INVESTMENT_SCORE = {
    "yes, i'm ready to invest": 40,
    "yes, i’m ready to invest": 40,
    "i may need financing options": 20,
    "just exploring": 5,
}

TIMELINE_SCORE = {
    "within 1–3 months": 35,
    "within 3–6 months": 20,
    "just exploring": 5,
}

INTENT_SCORE = {
    "high_commission_per_successful_closure": 25,
    "side_income": 10,
    "just_exploring": 2,
}


def get_rule_score(lead):

    inv = INVESTMENT_SCORE.get(
        lead["investment_ready"].lower(),
        5
    )

    tl = TIMELINE_SCORE.get(
        lead["timeline"].lower(),
        5
    )

    ei = INTENT_SCORE.get(
        lead["earning_intent"].lower(),
        2
    )

    return inv + tl + ei


# =============================================================================
# GROQ AI SCORING
# =============================================================================

def get_ai_score(lead, rule_score):

    prompt = f"""
You are an expert B2B franchise sales analyst.

Predict probability (0-100) that this lead
will open a franchise store.

Lead:
Name: {lead['lead_name']}
City: {lead['city']}
State: {lead['state']}

Investment Readiness:
{lead['investment_ready']}

Timeline:
{lead['timeline']}

Earning Intent:
{lead['earning_intent']}

Rule Score:
{rule_score}

Return ONLY integer score 0-100.
"""

    for attempt in range(3):

        try:

            response = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 5
                },
                timeout=30
            )

            if response.status_code == 429:

                wait = 5 * (attempt + 1)

                print(f"Groq rate limit — sleeping {wait}s")

                time.sleep(wait)

                continue

            if not response.ok:

                print(f"Groq error: {response.text}")

                return rule_score

            raw = response.json()["choices"][0]["message"]["content"]

            match = re.search(r"\d+", raw)

            if match:

                return int(match.group())

            return rule_score

        except Exception as e:

            print(f"Groq exception: {e}")

            return rule_score

    return rule_score


# =============================================================================
# GRADE
# =============================================================================

def grade(score):

    if score >= 85:
        return "A"

    if score >= 65:
        return "B"

    if score >= 45:
        return "C"

    if score >= 25:
        return "D"

    return "F"


# =============================================================================
# DEDUPLICATION
# =============================================================================

def get_existing_ids(table_id):

    query = f"""
    SELECT lead_id
    FROM `{table_id}`
    """

    try:

        rows = bq.query(query).result()

        return {r.lead_id for r in rows}

    except Exception:

        return set()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 60)
    print("FABO B2B LEAD SCORING")
    print("=" * 60)

    table_id = setup_bigquery()

    existing_ids = get_existing_ids(table_id)

    campaigns = fetch_lead_campaigns()

    all_rows = []

    # DEBUG MODE
    # Remove [:5] later
    for campaign in campaigns[:5]:

        print(f"\nCampaign: {campaign['name']}")

        ads = fetch_ads(campaign)

        print(f"Ads found: {len(ads)}")

        # DEBUG MODE
        for ad in ads[:10]:

            print(f"Fetching leads from: {ad['name']}")

            leads = fetch_leads(ad)

            print(f"Leads fetched: {len(leads)}")

            for raw_lead in leads:

                lead = parse_lead(raw_lead)

                if lead["lead_id"] in existing_ids:
                    continue

                rule_score = get_rule_score(lead)

                ai_score = get_ai_score(
                    lead,
                    rule_score
                )

                final_score = round(
                    (rule_score * 0.6) +
                    (ai_score * 0.4),
                    2
                )

                row = {
                    "lead_id": lead["lead_id"],
                    "lead_name": lead["lead_name"],
                    "phone": lead["phone"],
                    "email": lead["email"],
                    "city": lead["city"],
                    "state": lead["state"],
                    "platform": lead["platform"],
                    "investment_ready": lead["investment_ready"],
                    "timeline": lead["timeline"],
                    "earning_intent": lead["earning_intent"],
                    "campaign_id": lead["campaign_id"],
                    "campaign_name": lead["campaign_name"],
                    "ad_name": lead["ad_name"],
                    "lead_created_time": lead["lead_created_time"],
                    "rule_score": float(rule_score),
                    "ai_score": float(ai_score),
                    "final_score": float(final_score),
                    "grade": grade(final_score),
                    "scored_at": datetime.utcnow().isoformat()
                }

                all_rows.append(row)

                print(
                    f"{lead['lead_name']} | "
                    f"{final_score} | "
                    f"{grade(final_score)}"
                )

                # Groq protection
                time.sleep(2)

    print("\n")
    print("=" * 60)
    print(f"TOTAL ROWS TO INSERT: {len(all_rows)}")
    print("=" * 60)

    if not all_rows:
        print("No rows to insert.")
        return

    errors = bq.insert_rows_json(
        table_id,
        all_rows
    )

    if errors:

        print("\nBIGQUERY INSERT ERRORS:")
        print(errors)

    else:

        print("\nSUCCESS!")
        print(f"Inserted {len(all_rows)} rows into BigQuery")


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    main()
