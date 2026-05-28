#!/usr/bin/env python3
"""
FABO B2B Lead Scorer — Store Opener Prediction
Strategy: fetch ALL leads from the ad account in one paginated call filtered
by time window (max 7 days), then score in parallel with Groq.

Why this is fast:
  - OLD: 1,127 ads × 1 API call each = ~1,127 requests + 338 s forced sleep
  - NEW: 1 account-level /leads call paginated → all leads in ~5–20 requests
  - Groq scoring: parallel threads instead of sequential
"""

import os, re, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET  = os.getenv("BIGQUERY_DATASET", "meta_ads_b2b")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
GROQ_API_KEY      = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"

MAX_WINDOW_DAYS   = 7   # Never scan more than 7 days back on any run
GROQ_WORKERS      = 8   # Parallel threads for Groq scoring
GRAPH_VERSION     = "v19.0"
BASE              = f"https://graph.facebook.com/{GRAPH_VERSION}"

_raw = os.getenv("META_AD_ACCOUNT_ID", "")
META_AD_ACCOUNT_ID = f"act_{_raw}" if _raw and not _raw.startswith("act_") else _raw

for var, val in [("GCP_PROJECT_ID", GCP_PROJECT_ID),
                 ("META_ACCESS_TOKEN", META_ACCESS_TOKEN),
                 ("META_AD_ACCOUNT_ID", META_AD_ACCOUNT_ID),
                 ("GROK_API_KEY / GROQ_API_KEY", GROQ_API_KEY)]:
    if not val:
        raise ValueError(f"{var} is required")

print(f"Ad Account : {META_AD_ACCOUNT_ID}")

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
    """Return set of lead_ids already in BigQuery — used to skip duplicates."""
    get_bq()
    try:
        ids = {row.lead_id for row in bq.query(f"SELECT lead_id FROM `{table_id}`").result()}
        print(f"Already in BigQuery: {len(ids)} leads (will skip duplicates)")
        return ids
    except Exception as e:
        print(f"Could not query existing leads: {e}")
        return set()

def get_window(table_id):
    """
    Return (since_ts, until_ts) strings for this run.
    since = MAX(lead_created_time) + 1s, but never more than MAX_WINDOW_DAYS ago.
    until = now UTC.
    """
    get_bq()
    now        = datetime.now(timezone.utc)
    hard_floor = now - timedelta(days=MAX_WINDOW_DAYS)
    until_ts   = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    try:
        rows   = list(bq.query(f"SELECT MAX(lead_created_time) AS latest FROM `{table_id}`").result())
        latest = rows[0].latest if rows else None
        if latest:
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=timezone.utc)
            since = latest + timedelta(seconds=1)
            if since < hard_floor:
                print(f"Last BQ record was {latest.date()} — capping window to {MAX_WINDOW_DAYS} days")
                since = hard_floor
            else:
                print(f"Incremental: fetching leads after {since.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
            return since.strftime("%Y-%m-%dT%H:%M:%S+00:00"), until_ts
    except Exception as e:
        print(f"Timestamp query failed: {e}")

    since = hard_floor
    print(f"No prior data — using {MAX_WINDOW_DAYS}-day window from {since.date()}")
    return since.strftime("%Y-%m-%dT%H:%M:%S+00:00"), until_ts

# ── Meta API ──────────────────────────────────────────────────────────────────
def iso_to_unix(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())

def meta_get(url, params, retries=4):
    """Single Meta Graph API GET with rate-limit retry. Returns (data_list, error)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30).json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return [], str(e)
        if "error" in r:
            msg  = r["error"]["message"]
            code = r["error"].get("code", 0)
            if code in (17, 80004) or "too many calls" in msg.lower():
                wait = 20 * (attempt + 1)
                print(f"  [RATE LIMIT] waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            return [], msg
        return r.get("data", []), r.get("paging", {}).get("next"), None
    return [], None, "Rate limit: retries exhausted"

def validate_token():
    r = requests.get(f"{BASE}/me", params={"access_token": META_ACCESS_TOKEN}, timeout=15).json()
    if "error" in r:
        print(f"[META] Token invalid: {r['error']['message']}")
        return False
    print(f"[META] Token OK — {r.get('name', r.get('id'))}")
    return True

def fetch_all_leads_for_account(since_ts, until_ts, already_scored):
    """
    Fetch ALL new leads for the ad account in a single paginated stream,
    filtered server-side by time_created > since_ts.

    This replaces the old 1,127-ad loop:
      - 1 endpoint: /{ad_account}/leads
      - Server-side time filter (GREATER_THAN unix timestamp)
      - Paginated — typically 5–30 pages for a 7-day window
      - No per-ad sleep needed
    """
    since_unix = iso_to_unix(since_ts)
    until_dt   = datetime.fromisoformat(until_ts.replace("Z", "+00:00"))

    url    = f"{BASE}/{META_AD_ACCOUNT_ID}/leads"
    params = {
        "access_token": META_ACCESS_TOKEN,
        "fields": "id,created_time,field_data,platform,ad_id,ad_name,campaign_id,campaign_name",
        "filtering": f'[{{"field":"time_created","operator":"GREATER_THAN","value":{since_unix}}}]',
        "limit": 200,   # max page size — fewer round-trips
    }

    all_leads  = []
    page_count = 0
    skipped    = 0

    print(f"  Fetching from account endpoint (since={since_ts[:10]}, until={until_ts[:10]})...")

    while url:
        page_count += 1
        r = requests.get(url, params=params if params else {}, timeout=30)
        if not r.ok:
            print(f"  [HTTP {r.status_code}] {r.text[:200]}")
            break
        data = r.json()

        if "error" in data:
            msg  = data["error"]["message"]
            code = data["error"].get("code", 0)
            if code in (17, 80004) or "too many calls" in msg.lower():
                print(f"  [RATE LIMIT] page {page_count} — waiting 30s...")
                time.sleep(30)
                continue  # retry same URL
            print(f"  [API ERROR] {msg}")
            break

        for lead in data.get("data", []):
            # Client-side until_ts guard (server filter is only GREATER_THAN)
            try:
                created = datetime.fromisoformat(lead["created_time"].replace("Z", "+00:00"))
                if created > until_dt:
                    skipped += 1
                    continue
            except Exception:
                pass
            # Skip if already in BigQuery
            if lead.get("id") in already_scored:
                skipped += 1
                continue
            all_leads.append(lead)

        next_url = data.get("paging", {}).get("next")
        url      = next_url
        params   = None  # next URL has params baked in

        if page_count % 5 == 0:
            print(f"  ... page {page_count}, {len(all_leads)} new leads so far")

        # Tiny pause only every 10 pages to avoid burst rate limits
        if page_count % 10 == 0:
            time.sleep(1)

    print(f"  Done: {page_count} pages, {len(all_leads)} new leads ({skipped} skipped/duplicate)")
    return all_leads

# ── Lead Parsing ──────────────────────────────────────────────────────────────
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
        "lead_name":         g("full_name", "name", "first_name"),
        "phone":             g("phone_number", "phone").replace("p:", ""),
        "email":             g("email"),
        "city":              g("city"),
        "state":             g("additional_col1_select", "state", "province"),
        "platform":          lead.get("platform", ""),
        "investment_ready":  g("additional_col6_select", "investment_readiness", "are_you_ready_to_invest"),
        "timeline":          g("additional_col3_select", "timeline", "when_are_you_planning_to_start"),
        "earning_intent":    g("what_type_of_earning_opportunity_are_you_looking_for?", "earning_type"),
        "campaign_id":       lead.get("campaign_id", ""),
        "campaign_name":     lead.get("campaign_name", ""),
        "ad_name":           lead.get("ad_name", ""),
        "form_name":         "",
        "lead_created_time": lead.get("created_time", ""),
    }

# ── AI Scoring ────────────────────────────────────────────────────────────────
INV_SCORE = {"yes, i'm ready to invest": 40, "yes, i\u2019m ready to invest": 40,
             "i may need financing options": 20, "just exploring": 5}
TL_SCORE  = {"within 1\u20133 months": 35, "within 3\u20136 months": 20, "just exploring": 5}
ET_SCORE  = {"high_commission_per_successful_closure": 25, "side_income": 10, "just_exploring": 2}

def get_ai_score(lead):
    inv_s = INV_SCORE.get(lead["investment_ready"].lower(), 5)
    tl_s  = TL_SCORE.get(lead["timeline"].lower(), 5)
    et_s  = ET_SCORE.get(lead["earning_intent"].lower(), 2)
    base  = inv_s + tl_s + et_s

    inv_l = {40: "Ready to invest", 20: "Needs financing", 5: "Exploring"}.get(inv_s, "?")
    tl_l  = {35: "1-3 months", 20: "3-6 months", 5: "Exploring"}.get(tl_s, "?")
    et_l  = {25: "Franchise owner", 10: "Side income", 2: "Exploring"}.get(et_s, "?")

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
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 5, "temperature": 0.0},
                timeout=30)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            if not r.ok:
                return base
            raw = r.json()["choices"][0]["message"]["content"].strip()
            m   = re.search(r"\b(\d{1,3})\b", raw)
            return max(0, min(100, int(m.group(1)))) if m else base
        except Exception:
            return base
    return base

def score_lead(args):
    """Worker function for ThreadPoolExecutor."""
    i, total, lead, now = args
    score    = get_ai_score(lead)
    g        = grade(score)
    ts       = lead["lead_created_time"]
    try:
        ts_clean = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        ts_clean = now.strftime("%Y-%m-%dT%H:%M:%S")
    return i, {
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
    }

def grade(s):
    return "A" if s >= 85 else "B" if s >= 65 else "C" if s >= 45 else "D" if s >= 25 else "F"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    print("=" * 55)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 55)

    print("\nSetting up BigQuery...")
    table_id = setup_bigquery()

    print("\nValidating Meta token...")
    if not validate_token():
        return

    since_ts, until_ts = get_window(table_id)
    print(f"Window: {since_ts[:10]} → {until_ts[:10]} (max {MAX_WINDOW_DAYS} days)")

    # Load existing IDs once — used for dedup inside the fetch loop
    already = get_already_scored_ids(table_id)

    # ── Fetch leads (single account-level call, not per-ad) ──────────────────
    print("\nFetching leads from account endpoint...")
    t0         = time.time()
    raw_leads  = fetch_all_leads_for_account(since_ts, until_ts, already)
    fetch_secs = time.time() - t0
    print(f"  Fetch completed in {fetch_secs:.1f}s")

    parsed_leads = [parse_lead(l) for l in raw_leads]

    print(f"\nNew leads to score: {len(parsed_leads)}")
    if not parsed_leads:
        print("Nothing new — BigQuery is up to date.")
        return

    # ── Score in parallel ─────────────────────────────────────────────────────
    print(f"\nScoring {len(parsed_leads)} leads with Groq ({GROQ_WORKERS} parallel workers)...")
    t1   = time.time()
    rows = [None] * len(parsed_leads)
    args = [(i + 1, len(parsed_leads), lead, now) for i, lead in enumerate(parsed_leads)]

    with ThreadPoolExecutor(max_workers=GROQ_WORKERS) as pool:
        futures = {pool.submit(score_lead, a): a[0] for a in args}
        for fut in as_completed(futures):
            try:
                i, row = fut.result()
                rows[i - 1] = row
                g = row["grade"]
                s = int(row["store_open_score"])
                print(f"  [{i:3d}/{len(parsed_leads)}] "
                      f"{(row['lead_name'] or '?'):25s} | "
                      f"{(row['city'] or '?'):15s} | "
                      f"{s:3d} ({g}) | "
                      f"{row['investment_ready'][:28]}")
            except Exception as e:
                print(f"  [SCORE ERROR] {e}")

    score_secs = time.time() - t1
    print(f"  Scoring completed in {score_secs:.1f}s")

    # Filter out any None rows (shouldn't happen, but be safe)
    rows = [r for r in rows if r is not None]

    # ── Save to BigQuery in batches ───────────────────────────────────────────
    BATCH = 200
    saved = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        errs  = bq.insert_rows_json(table_id, batch)
        if errs:
            print(f"  [BQ] Batch {start}–{start+len(batch)} error: {errs[:2]}")
        else:
            saved += len(batch)
            print(f"  [BQ] ✓ Saved rows {start+1}–{start+len(batch)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in rows:
        grade_counts[r["grade"]] += 1

    total_secs = time.time() - t0
    print(f"\n{'─'*50}")
    print(f"  Window   : {since_ts[:10]} → {until_ts[:10]}")
    print(f"  Saved    : {saved} leads  |  Total time: {total_secs:.0f}s")
    print(f"{'─'*50}")
    print(f"  A Hot  (85-100) : {grade_counts['A']:3d}  ← call immediately")
    print(f"  B Warm (65-84)  : {grade_counts['B']:3d}  ← follow up today")
    print(f"  C Cool (45-64)  : {grade_counts['C']:3d}  ← nurture sequence")
    print(f"  D Cold (25-44)  : {grade_counts['D']:3d}  ← low priority")
    print(f"  F Dead (0-24)   : {grade_counts['F']:3d}  ← skip")
    print(f"{'─'*50}")

if __name__ == "__main__":
    main()
