#!/usr/bin/env python3
"""
FABO B2B Lead Scorer — Store Opener Prediction

Speed strategy:
  - Fetch ad list once (campaigns → ads, all statuses)
  - Fetch leads from all ads IN PARALLEL (20 threads) with 7-day time filter
  - Skip ads that returned 0 leads (no sleep wasted)
  - Score new leads IN PARALLEL (8 Groq threads)

Expected runtime: ~2–4 min for 1,100+ ads with 0–50 new leads/day
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

MAX_WINDOW_DAYS   = 7    # Never look back more than 7 days
META_FETCH_WORKERS = 5   # Keep under Meta rate limits  # Parallel threads for fetching leads from ads
GROQ_WORKERS      = 8    # Parallel threads for Groq AI scoring
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
    get_bq()
    try:
        ids = {row.lead_id for row in bq.query(f"SELECT lead_id FROM `{table_id}`").result()}
        print(f"Already in BigQuery: {len(ids)} leads (will skip duplicates)")
        return ids
    except Exception as e:
        print(f"Could not query existing leads: {e}")
        return set()

def get_window(table_id):
    """Return (since_ts, until_ts), since capped at MAX_WINDOW_DAYS ago."""
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
                print(f"Last BQ record {latest.date()} — capping window to {MAX_WINDOW_DAYS} days")
                since = hard_floor
            else:
                print(f"Incremental: fetching leads after {since.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
            return since.strftime("%Y-%m-%dT%H:%M:%S+00:00"), until_ts
    except Exception as e:
        print(f"Timestamp query failed: {e}")

    since = hard_floor
    print(f"No prior data — using {MAX_WINDOW_DAYS}-day window from {since.date()}")
    return since.strftime("%Y-%m-%dT%H:%M:%S+00:00"), until_ts

# ── Meta API helpers ──────────────────────────────────────────────────────────
def iso_to_unix(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())

def meta_paginate_serial(url, params):
    """Paginate a single Meta endpoint serially with exponential backoff."""
    results = []
    while url:
        for attempt in range(6):  # waits: 30, 60, 120, 240, 300, 300s
            try:
                r = requests.get(url, params=params if params else {}, timeout=30).json()
            except Exception as e:
                if attempt < 5:
                    time.sleep(10)
                    continue
                return results, str(e)
            if "error" in r:
                msg  = r["error"]["message"]
                code = r["error"].get("code", 0)
                if code in (17, 80004) or "too many calls" in msg.lower():
                    wait = min(300, 30 * (2 ** attempt))  # 30,60,120,240,300,300
                    print(f"  [RATE LIMIT] waiting {wait}s (attempt {attempt+1}/6)...")
                    time.sleep(wait)
                    continue
                return results, msg
            break
        else:
            return results, "Rate limit exhausted after 6 attempts"
        results.extend(r.get("data", []))
        url    = r.get("paging", {}).get("next")
        params = None
        time.sleep(0.5)  # gentle inter-page pause
    return results, None

def validate_token():
    r = requests.get(f"{BASE}/me",
                     params={"access_token": META_ACCESS_TOKEN}, timeout=15).json()
    if "error" in r:
        print(f"[META] Token invalid: {r['error']['message']}")
        return False
    print(f"[META] Token OK — {r.get('name', r.get('id'))}")
    return True

# ── Ad discovery ──────────────────────────────────────────────────────────────
def fetch_all_lead_ads():
    """
    Fetch ALL lead-gen ads (all statuses) across all campaigns.
    Returns list of ad dicts with _campaign_id / _campaign_name attached.
    """
    campaigns, err = meta_paginate_serial(
        f"{BASE}/{META_AD_ACCOUNT_ID}/campaigns",
        {"access_token": META_ACCESS_TOKEN,
         "fields": "id,name,objective,effective_status",
         "limit": 100}
    )
    if err:
        print(f"[META] Campaign fetch error: {err}")
        return []

    LEAD_OBJ   = {"LEAD_GENERATION", "OUTCOME_LEADS"}
    lead_camps = [c for c in campaigns if c.get("objective") in LEAD_OBJ]
    all_obj    = sorted({c.get("objective", "?") for c in campaigns})
    print(f"  Campaigns: {len(campaigns)} total, {len(lead_camps)} lead gen")
    print(f"  Objectives: {all_obj}")

    if not lead_camps:
        return []

    all_ads = []
    for camp in lead_camps:
        ads, err = meta_paginate_serial(
            f"{BASE}/{camp['id']}/ads",
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

    active   = sum(1 for a in all_ads if a.get("effective_status") == "ACTIVE")
    inactive = len(all_ads) - active
    print(f"  Total ads: {len(all_ads)} ({active} active, {inactive} inactive/paused)")
    return all_ads

# ── Parallel lead fetching ────────────────────────────────────────────────────
def fetch_leads_for_one_ad(ad, since_unix, until_dt, already_scored):
    """
    Fetch leads from a single ad filtered by time. Designed to run in a thread.
    Returns list of raw lead dicts (new only, not in already_scored).
    """
    url    = f"{BASE}/{ad['id']}/leads"
    params = {
        "access_token": META_ACCESS_TOKEN,
        "fields": "id,created_time,field_data,platform,ad_name,campaign_id,campaign_name",
        "filtering": f'[{{"field":"time_created","operator":"GREATER_THAN","value":{since_unix}}}]',
        "limit": 100,
    }
    leads = []
    while url:
        for attempt in range(3):
            try:
                r = requests.get(url, params=params if params else {}, timeout=30).json()
            except Exception:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return leads
            if "error" in r:
                code = r["error"].get("code", 0)
                msg  = r["error"]["message"]
                if code in (17, 80004) or "too many calls" in msg.lower():
                    wait = min(120, 15 * (2 ** attempt))  # 15, 30, 60
                    time.sleep(wait)
                    continue
                # code 100 = no leads on this ad — expected, not an error
                return leads
            break
        else:
            return leads

        for lead in r.get("data", []):
            # Client-side upper bound guard
            try:
                created = datetime.fromisoformat(lead["created_time"].replace("Z", "+00:00"))
                if created > until_dt:
                    continue
            except Exception:
                pass
            if lead.get("id") in already_scored:
                continue
            lead["_ad_name"]       = ad.get("name", "")
            lead["_campaign_id"]   = ad.get("_campaign_id", "")
            lead["_campaign_name"] = ad.get("_campaign_name", "")
            leads.append(lead)

        url    = r.get("paging", {}).get("next")
        params = None
    return leads

def fetch_all_leads_parallel(ads, since_ts, until_ts, already_scored):
    """
    Fetch leads from all ads using a thread pool.
    1,127 ads with 20 workers ≈ 57 rounds × ~0.5s = ~30s total vs 18 min serial.

    Dedup note: already_scored covers leads from previous runs.
    seen_this_run deduplicates within this run (same lead_id on 2 ads).
    Both sets are read-only inside threads; seen_this_run is updated only
    in the main thread as futures complete, so no lock needed.
    """
    since_unix    = iso_to_unix(since_ts)
    until_dt      = datetime.fromisoformat(until_ts.replace("Z", "+00:00"))
    seen_this_run = set()  # tracks lead_ids collected so far this run

    all_leads      = []
    ads_with_leads = 0
    done           = 0
    total          = len(ads)

    print(f"  Fetching from {total} ads in parallel ({META_FETCH_WORKERS} workers)...")

    with ThreadPoolExecutor(max_workers=META_FETCH_WORKERS) as pool:
        future_to_ad = {
            pool.submit(fetch_leads_for_one_ad, ad, since_unix, until_dt, already_scored): ad
            for ad in ads
        }
        for fut in as_completed(future_to_ad):
            done += 1
            try:
                leads = fut.result()
                # Deduplicate within this run (main thread only — no lock needed)
                new_leads = [l for l in leads if l.get("id") not in seen_this_run]
                for l in new_leads:
                    seen_this_run.add(l.get("id"))
                if new_leads:
                    ads_with_leads += 1
                    all_leads.extend(new_leads)
                    ad = future_to_ad[fut]
                    print(f"    ✓ '{ad.get('name','?')}' — {len(new_leads)} new leads  "
                          f"[{done}/{total}]")
            except Exception as e:
                print(f"    [FETCH ERROR] {e}")

            if done % 100 == 0:
                print(f"  ... {done}/{total} ads checked, {len(all_leads)} new leads so far")

    print(f"  Done: {total} ads checked, {ads_with_leads} had new leads, "
          f"{len(all_leads)} total new leads")
    return all_leads

# ── Lead parsing ──────────────────────────────────────────────────────────────
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
        "investment_ready":  g("additional_col6_select", "investment_readiness",
                               "are_you_ready_to_invest"),
        "timeline":          g("additional_col3_select", "timeline",
                               "when_are_you_planning_to_start"),
        "earning_intent":    g("what_type_of_earning_opportunity_are_you_looking_for?",
                               "earning_type"),
        "campaign_id":       lead.get("_campaign_id", "") or lead.get("campaign_id", ""),
        "campaign_name":     lead.get("_campaign_name", "") or lead.get("campaign_name", ""),
        "ad_name":           lead.get("_ad_name", "") or lead.get("ad_name", ""),
        "form_name":         "",
        "lead_created_time": lead.get("created_time", ""),
    }

# ── AI scoring ────────────────────────────────────────────────────────────────
INV_SCORE = {"yes, i'm ready to invest": 40, "yes, i\u2019m ready to invest": 40,
             "i may need financing options": 20, "just exploring": 5}
TL_SCORE  = {"within 1\u20133 months": 35, "within 3\u20136 months": 20, "just exploring": 5}
ET_SCORE  = {"high_commission_per_successful_closure": 25, "side_income": 10,
             "just_exploring": 2}

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
            text = r.json()["choices"][0]["message"]["content"].strip()
            m    = re.search(r"\b(\d{1,3})\b", text)
            return max(0, min(100, int(m.group(1)))) if m else base
        except Exception:
            return base
    return base

def grade(s):
    return "A" if s >= 85 else "B" if s >= 65 else "C" if s >= 45 else "D" if s >= 25 else "F"

def score_lead_worker(args):
    """Worker for parallel Groq scoring."""
    i, lead, now = args
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

    already = get_already_scored_ids(table_id)

    # ── 1. Discover all lead-gen ads ─────────────────────────────────────────
    print("\nFetching lead gen ads...")
    t0  = time.time()
    time.sleep(2)  # brief pause after token validation before campaign fetch
    ads = fetch_all_lead_ads()
    if not ads:
        print("No lead gen ads found.")
        return

    # ── 2. Fetch leads from all ads in parallel ───────────────────────────────
    print(f"\nFetching leads ({META_FETCH_WORKERS} parallel workers)...")
    raw_leads  = fetch_all_leads_parallel(ads, since_ts, until_ts, already)
    fetch_secs = time.time() - t0
    print(f"  Ad discovery + lead fetch: {fetch_secs:.1f}s")

    parsed = [parse_lead(l) for l in raw_leads]

    print(f"\nNew leads to score: {len(parsed)}")
    if not parsed:
        print(f"Nothing new — BigQuery is up to date. Total time: {time.time()-t0:.0f}s")
        return

    # ── 3. Score in parallel ──────────────────────────────────────────────────
    print(f"\nScoring {len(parsed)} leads with Groq ({GROQ_WORKERS} workers)...")
    t1   = time.time()
    rows = [None] * len(parsed)

    with ThreadPoolExecutor(max_workers=GROQ_WORKERS) as pool:
        futures = {pool.submit(score_lead_worker, (i + 1, lead, now)): i
                   for i, lead in enumerate(parsed)}
        for fut in as_completed(futures):
            try:
                i, row = fut.result()
                rows[i - 1] = row
                s = int(row["store_open_score"])
                print(f"  [{i:3d}/{len(parsed)}] "
                      f"{(row['lead_name'] or '?'):25s} | "
                      f"{(row['city'] or '?'):15s} | "
                      f"{s:3d} ({row['grade']}) | "
                      f"{row['investment_ready'][:30]}")
            except Exception as e:
                print(f"  [SCORE ERROR] {e}")

    score_secs = time.time() - t1
    rows = [r for r in rows if r is not None]

    # ── 4. Save to BigQuery ───────────────────────────────────────────────────
    BATCH = 200
    saved = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        errs  = bq.insert_rows_json(table_id, batch)
        if errs:
            print(f"  [BQ] Batch error: {errs[:2]}")
        else:
            saved += len(batch)
            print(f"  [BQ] ✓ Saved rows {start+1}–{start+len(batch)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    gc = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in rows:
        gc[r["grade"]] += 1

    total_secs = time.time() - t0
    print(f"\n{'─'*50}")
    print(f"  Window     : {since_ts[:10]} → {until_ts[:10]}")
    print(f"  Fetch      : {fetch_secs:.0f}s  |  Score: {score_secs:.0f}s  |  Total: {total_secs:.0f}s")
    print(f"  Saved      : {saved} leads")
    print(f"{'─'*50}")
    print(f"  A Hot  (85-100) : {gc['A']:3d}  ← call immediately")
    print(f"  B Warm (65-84)  : {gc['B']:3d}  ← follow up today")
    print(f"  C Cool (45-64)  : {gc['C']:3d}  ← nurture sequence")
    print(f"  D Cold (25-44)  : {gc['D']:3d}  ← low priority")
    print(f"  F Dead (0-24)   : {gc['F']:3d}  ← skip")
    print(f"{'─'*50}")

if __name__ == "__main__":
    main()
