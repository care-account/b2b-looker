#!/usr/bin/env python3

"""
FABO B2B Lead Scoring Pipeline
Meta Ads -> Groq AI -> BigQuery

FINAL PRODUCTION VERSION

FEATURES
--------
✓ Incremental sync
✓ BigQuery batching
✓ Meta rate-limit handling
✓ Groq retry handling
✓ Deduplication
✓ Fast GitHub Actions execution
✓ Campaign filtering
✓ Ad limiting
✓ AI scoring only for quality leads
✓ Safe for daily cron jobs
"""

import os
import re
import time
import json
import requests

from datetime import datetime, timedelta, timezone

from google.cloud import bigquery

# =============================================================================
# ENV CONFIG
# =============================================================================

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

BIGQUERY_DATASET = os.getenv(
    "BIGQUERY_DATASET",
    "meta_ads_b2b"
)

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")

RAW_AD_ACCOUNT = os.getenv(
    "META_AD_ACCOUNT_ID",
    ""
)

META_AD_ACCOUNT_ID = (
    f"act_{RAW_AD_ACCOUNT}"
    if RAW_AD_ACCOUNT
    and not RAW_AD_ACCOUNT.startswith("act_")
    else RAW_AD_ACCOUNT
)

GROQ_API_KEY = (
    os.getenv("GROK_API_KEY")
    or os.getenv("GROQ_API_KEY")
)

SYNC_DAYS = int(
    os.getenv("SYNC_DAYS", "1")
)

MAX_ADS_PER_CAMPAIGN = int(
    os.getenv("MAX_ADS_PER_CAMPAIGN", "10")
)

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "50")
)

GROQ_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)

# =============================================================================
# VALIDATION
# =============================================================================

required = {
    "GCP_PROJECT_ID": GCP_PROJECT_ID,
    "META_ACCESS_TOKEN": META_ACCESS_TOKEN,
    "META_AD_ACCOUNT_ID": META_AD_ACCOUNT_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
}

for k, v in required.items():

    if not v:
        raise ValueError(f"{k} is missing")

# =============================================================================
# BIGQUERY
# =============================================================================

bq = bigquery.Client(project=GCP_PROJECT_ID)

# =============================================================================
# BIGQUERY SETUP
# =============================================================================


def setup_bigquery():

    dataset_id = (
        f"{GCP_PROJECT_ID}."
        f"{BIGQUERY_DATASET}"
    )

    try:

        bq.get_dataset(dataset_id)

        print(f"Dataset exists: {dataset_id}")

    except Exception:

        dataset = bigquery.Dataset(dataset_id)

        bq.create_dataset(dataset)

        print(f"Created dataset: {dataset_id}")

    table_id = f"{dataset_id}.lead_scores"

    schema = [

        bigquery.SchemaField(
            "lead_id",
            "STRING"
        ),

        bigquery.SchemaField(
            "lead_name",
            "STRING"
        ),

        bigquery.SchemaField(
            "phone",
            "STRING"
        ),

        bigquery.SchemaField(
            "email",
            "STRING"
        ),

        bigquery.SchemaField(
            "city",
            "STRING"
        ),

        bigquery.SchemaField(
            "state",
            "STRING"
        ),

        bigquery.SchemaField(
            "platform",
            "STRING"
        ),

        bigquery.SchemaField(
            "investment_ready",
            "STRING"
        ),

        bigquery.SchemaField(
            "timeline",
            "STRING"
        ),

        bigquery.SchemaField(
            "earning_intent",
            "STRING"
        ),

        bigquery.SchemaField(
            "campaign_id",
            "STRING"
        ),

        bigquery.SchemaField(
            "campaign_name",
            "STRING"
        ),

        bigquery.SchemaField(
            "ad_name",
            "STRING"
        ),

        bigquery.SchemaField(
            "lead_created_time",
            "TIMESTAMP"
        ),

        bigquery.SchemaField(
            "rule_score",
            "FLOAT"
        ),

        bigquery.SchemaField(
            "ai_score",
            "FLOAT"
        ),

        bigquery.SchemaField(
            "final_score",
            "FLOAT"
        ),

        bigquery.SchemaField(
            "grade",
            "STRING"
        ),

        bigquery.SchemaField(
            "scored_at",
            "TIMESTAMP"
        ),
    ]

    try:

        bq.get_table(table_id)

        print(f"Table exists: {table_id}")

    except Exception:

        table = bigquery.Table(
            table_id,
            schema=schema
        )

        bq.create_table(table)

        print(f"Created table: {table_id}")

    return table_id

# =============================================================================
# EXISTING IDS
# =============================================================================


def get_existing_ids(table_id):

    query = f"""
    SELECT lead_id
    FROM `{table_id}`
    """

    try:

        rows = bq.query(query).result()

        return {
            r.lead_id
            for r in rows
        }

    except Exception as e:

        print(
            f"Could not fetch existing IDs: {e}"
        )

        return set()

# =============================================================================
# BIGQUERY INSERT
# =============================================================================


def insert_batch(table_id, rows):

    if not rows:
        return

    try:

        errors = bq.insert_rows_json(
            table_id,
            rows
        )

        if errors:

            print("\nBigQuery insert errors:")
            print(errors)

        else:

            print(
                f"Inserted "
                f"{len(rows)} rows"
            )

    except Exception as e:

        print(f"BigQuery error: {e}")

# =============================================================================
# META PAGINATION
# =============================================================================


def paginate(url, params=None):

    results = []

    while url:

        try:

            # Meta protection
            time.sleep(0.15)

            r = requests.get(
                url,
                params=params if params else {},
                timeout=60
            )

            data = r.json()

            if "error" in data:

                code = data["error"].get("code")

                # Meta rate limit
                if code in [4, 17]:

                    print(
                        "Meta rate limit..."
                        "sleeping 20s"
                    )

                    time.sleep(20)

                    continue

                print(
                    f"Meta error: "
                    f"{data['error']['message']}"
                )

                break

            results.extend(
                data.get("data", [])
            )

            url = (
                data
                .get("paging", {})
                .get("next")
            )

            params = None

        except Exception as e:

            print(f"Pagination error: {e}")

            break

    return results

# =============================================================================
# FETCH CAMPAIGNS
# =============================================================================


def fetch_campaigns():

    base = "https://graph.facebook.com/v19.0"

    since = (
        datetime.now(timezone.utc)
        - timedelta(days=7)
    ).strftime("%Y-%m-%d")

    campaigns = paginate(
        f"{base}/{META_AD_ACCOUNT_ID}/campaigns",
        {
            "access_token":
            META_ACCESS_TOKEN,

            "fields":
            "id,name,objective,status,updated_time",

            "limit": 100,

            "filtering": json.dumps([
                {
                    "field": "updated_time",
                    "operator": "GREATER_THAN",
                    "value": since
                }
            ])
        }
    )

    lead_objectives = {
        "LEAD_GENERATION",
        "OUTCOME_LEADS"
    }

    filtered = [

        c for c in campaigns

        if c.get("objective")
        in lead_objectives
    ]

    print(
        f"Lead campaigns found: "
        f"{len(filtered)}"
    )

    return filtered

# =============================================================================
# FETCH ADS
# =============================================================================


def fetch_ads(campaign):

    base = "https://graph.facebook.com/v19.0"

    ads = paginate(
        f"{base}/{campaign['id']}/ads",
        {
            "access_token":
            META_ACCESS_TOKEN,

            "fields":
            "id,name,status,updated_time",

            "limit":
            MAX_ADS_PER_CAMPAIGN
        }
    )

    for ad in ads:

        ad["_campaign_name"] = (
            campaign["name"]
        )

    return ads

# =============================================================================
# FETCH LEADS
# =============================================================================


def fetch_leads(ad):

    base = "https://graph.facebook.com/v19.0"

    since = (
        datetime.now(timezone.utc)
        - timedelta(days=SYNC_DAYS)
    )

    leads = paginate(
        f"{base}/{ad['id']}/leads",
        {
            "access_token":
            META_ACCESS_TOKEN,

            "fields":
            (
                "id,"
                "created_time,"
                "field_data,"
                "platform,"
                "campaign_id,"
                "campaign_name,"
                "ad_name"
            ),

            "limit": 100,

            "filtering": json.dumps([
                {
                    "field": "time_created",
                    "operator": "GREATER_THAN",
                    "value": int(
                        since.timestamp()
                    )
                }
            ])
        }
    )

    return leads

# =============================================================================
# PARSE LEAD
# =============================================================================


def parse_lead(lead):

    raw = {}

    for field in lead.get("field_data", []):

        values = field.get("values", [])

        raw[field["name"]] = (
            values[0]
            if values else ""
        )

    def get_value(*keys):

        for k in keys:

            val = raw.get(k, "").strip()

            if val:
                return val

        return ""

    return {

        "lead_id":
        lead.get("id", ""),

        "lead_name":
        get_value("full_name", "name"),

        "phone":
        get_value(
            "phone_number",
            "phone"
        ),

        "email":
        get_value("email"),

        "city":
        get_value("city"),

        "state":
        get_value(
            "state",
            "province"
        ),

        "platform":
        lead.get("platform", ""),

        "investment_ready":
        get_value(
            "investment_readiness",
            "ready_to_invest"
        ),

        "timeline":
        get_value(
            "timeline",
            "when_are_you_planning_to_start"
        ),

        "earning_intent":
        get_value(
            "earning_type",
            "opportunity_type"
        ),

        "campaign_id":
        lead.get("campaign_id", ""),

        "campaign_name":
        lead.get("campaign_name", ""),

        "ad_name":
        lead.get("ad_name", ""),

        "lead_created_time":
        lead.get("created_time", ""),
    }

# =============================================================================
# RULE SCORING
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


def rule_score(lead):

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
# AI SCORE
# =============================================================================


def ai_score(lead, base_score):

    # Skip AI for weak leads
    if base_score < 40:
        return base_score

    prompt = f"""
You are a B2B franchise sales expert.

Predict probability (0-100)
that this lead will become
a serious franchise buyer.

Lead:

Name:
{lead['lead_name']}

City:
{lead['city']}

State:
{lead['state']}

Investment Readiness:
{lead['investment_ready']}

Timeline:
{lead['timeline']}

Earning Intent:
{lead['earning_intent']}

Rule Score:
{base_score}

Return ONLY integer score.
"""

    for attempt in range(3):

        try:

            r = requests.post(
                GROQ_URL,

                headers={
                    "Authorization":
                    f"Bearer {GROQ_API_KEY}",

                    "Content-Type":
                    "application/json"
                },

                json={
                    "model":
                    "llama-3.1-8b-instant",

                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    "temperature": 0,

                    "max_tokens": 5
                },

                timeout=60
            )

            if r.status_code == 429:

                wait = 5 * (attempt + 1)

                print(
                    f"Groq rate limit..."
                    f"sleeping {wait}s"
                )

                time.sleep(wait)

                continue

            if not r.ok:

                print(
                    f"Groq error: {r.text}"
                )

                return base_score

            content = (
                r.json()
                ["choices"][0]
                ["message"]["content"]
            )

            match = re.search(
                r"\d+",
                content
            )

            if match:
                return int(match.group())

            return base_score

        except Exception as e:

            print(f"Groq exception: {e}")

            return base_score

    return base_score

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
# MAIN
# =============================================================================


def main():

    print("=" * 60)
    print("FABO B2B LEAD SCORING")
    print("=" * 60)

    table_id = setup_bigquery()

    existing_ids = get_existing_ids(
        table_id
    )

    campaigns = fetch_campaigns()

    batch_rows = []

    total_processed = 0

    for campaign in campaigns:

        print(
            f"\nCampaign: "
            f"{campaign['name']}"
        )

        ads = fetch_ads(campaign)

        print(f"Ads found: {len(ads)}")

        for ad in ads:

            print(
                f"Fetching: {ad['name']}"
            )

            leads = fetch_leads(ad)

            # Skip empty ads fast
            if not leads:
                continue

            print(
                f"Leads fetched: "
                f"{len(leads)}"
            )

            for raw in leads:

                lead = parse_lead(raw)

                if (
                    lead["lead_id"]
                    in existing_ids
                ):
                    continue

                base = rule_score(lead)

                ai = ai_score(
                    lead,
                    base
                )

                final = round(
                    (base * 0.6)
                    + (ai * 0.4),
                    2
                )

                row = {

                    "lead_id":
                    lead["lead_id"],

                    "lead_name":
                    lead["lead_name"],

                    "phone":
                    lead["phone"],

                    "email":
                    lead["email"],

                    "city":
                    lead["city"],

                    "state":
                    lead["state"],

                    "platform":
                    lead["platform"],

                    "investment_ready":
                    lead["investment_ready"],

                    "timeline":
                    lead["timeline"],

                    "earning_intent":
                    lead["earning_intent"],

                    "campaign_id":
                    lead["campaign_id"],

                    "campaign_name":
                    lead["campaign_name"],

                    "ad_name":
                    lead["ad_name"],

                    "lead_created_time":
                    lead["lead_created_time"],

                    "rule_score":
                    float(base),

                    "ai_score":
                    float(ai),

                    "final_score":
                    float(final),

                    "grade":
                    grade(final),

                    "scored_at":
                    datetime.utcnow().isoformat()
                }

                batch_rows.append(row)

                total_processed += 1

                print(
                    f"{lead['lead_name']} | "
                    f"{final} | "
                    f"{grade(final)}"
                )

                # Batch insert
                if len(batch_rows) >= BATCH_SIZE:

                    insert_batch(
                        table_id,
                        batch_rows
                    )

                    batch_rows = []

                # Small delay
                time.sleep(0.1)

    # Final insert
    if batch_rows:

        insert_batch(
            table_id,
            batch_rows
        )

    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"Total processed: "
        f"{total_processed}"
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    main()
