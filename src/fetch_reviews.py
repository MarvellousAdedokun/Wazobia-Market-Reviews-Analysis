"""
fetch_reviews.py
-----------------
Pulls Google Maps reviews for Wazobia African Market via the official
SerpApi Google Maps Reviews endpoint (a licensed, ToS-compliant proxy
onto Google's data — this does NOT scrape google.com directly).

SECURITY NOTE:
Your API key is read from the environment variable SERPAPI_KEY.
Never hardcode it here, never commit it, never paste it into chat.
Set it before running:

    export SERPAPI_KEY="your_key_here"      # macOS/Linux
    setx SERPAPI_KEY "your_key_here"         # Windows (new terminal after)

USAGE:
    python fetch_reviews.py --max-reviews 2000 --out reviews_raw.csv

If the listing has fewer reviews than --max-reviews, the script will
simply stop when it runs out (this is expected for smaller businesses).
"""

import os
import sys
import csv
import time
import argparse
import logging
from datetime import datetime

try:
    from serpapi import GoogleSearch
except ImportError:
    sys.exit(
        "Missing dependency. Install with:\n"
        "    pip install google-search-results --break-system-packages"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_reviews")

# Wazobia African Market & Kitchen, 16203 Westheimer Rd, Houston, TX
# (Google Place ID, found via a normal public Maps search — not scraped)
DEFAULT_PLACE_ID = "ChIJQa2divPeQIYRfq5t1NFP4iY"

FIELDS = [
    "review_id",
    "reviewer_name",
    "rating",
    "review_date_raw",
    "review_text",
    "owner_response",
    "likes",
    "local_guide",
]


def get_api_key() -> str:
    key = os.environ.get("SERPAPI_KEY")
    if not key:
        sys.exit(
            "SERPAPI_KEY environment variable not set.\n"
            "Set it first, e.g.: export SERPAPI_KEY='your_key_here'\n"
            "(Get a free key, no card required, at serpapi.com)"
        )
    return key


def fetch_all_reviews(place_id: str, api_key: str, max_reviews: int, sleep_s: float = 1.0):
    """Paginate through Google Maps reviews for a place via SerpApi."""
    all_reviews = []
    params = {
        "engine": "google_maps_reviews",
        "place_id": place_id,
        "api_key": api_key,
        "sort_by": "newestFirst",
    }

    page = 1
    while len(all_reviews) < max_reviews:
        log.info(f"Fetching page {page} (have {len(all_reviews)} reviews so far)...")
        search = GoogleSearch(params)
        result = search.get_dict()

        if "error" in result:
            log.error(f"SerpApi error: {result['error']}")
            break

        reviews = result.get("reviews", [])
        if not reviews:
            log.info("No more reviews returned. Stopping.")
            break

        for r in reviews:
            all_reviews.append({
                "review_id": r.get("review_id", ""),
                "reviewer_name": r.get("user", {}).get("name", ""),
                "rating": r.get("rating", ""),
                "review_date_raw": r.get("date", ""),
                "review_text": (r.get("snippet", "") or "").replace("\n", " ").strip(),
                "owner_response": (r.get("response", {}) or {}).get("snippet", ""),
                "likes": r.get("likes", 0),
                "local_guide": r.get("user", {}).get("local_guide", False),
            })

        next_page_token = result.get("serpapi_pagination", {}).get("next_page_token")
        if not next_page_token or len(reviews) == 0:
            log.info("Reached end of available reviews.")
            break

        params["next_page_token"] = next_page_token
        page += 1
        time.sleep(sleep_s)  # be a good API citizen, don't hammer the endpoint

    return all_reviews[:max_reviews]


def save_csv(reviews, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(reviews)
    log.info(f"Saved {len(reviews)} reviews to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Google Maps reviews via SerpApi.")
    parser.add_argument("--place-id", default=DEFAULT_PLACE_ID, help="Google Place ID")
    parser.add_argument("--max-reviews", type=int, default=2000, help="Max reviews to pull")
    parser.add_argument("--out", default="reviews_raw.csv", help="Output CSV path")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between page requests")
    args = parser.parse_args()

    api_key = get_api_key()
    log.info(f"Starting fetch for place_id={args.place_id}, target={args.max_reviews} reviews")

    reviews = fetch_all_reviews(args.place_id, api_key, args.max_reviews, args.sleep)

    if not reviews:
        log.warning(
            "No reviews retrieved. This can happen if the free-tier quota is "
            "used up, the place_id is wrong, or the listing has no reviews."
        )
        sys.exit(1)

    save_csv(reviews, args.out)
    log.info(f"Done. Retrieved {len(reviews)} reviews (listing may not have more than this).")


if __name__ == "__main__":
    main()
