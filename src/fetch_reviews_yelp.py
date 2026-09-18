"""
fetch_reviews_yelp.py
----------------------
Pulls Yelp reviews for Wazobia African Market via SerpApi's licensed
Yelp engines (yelp -> business lookup, yelp_reviews -> paginated reviews).
Same compliance logic as fetch_reviews.py: this is a licensed API, not a
scraper hitting yelp.com directly.

SECURITY: reads your key from the SERPAPI_KEY environment variable.
Never hardcode it, never commit it.

USAGE:
    python fetch_reviews_yelp.py --business "Wazobia African Market" \
        --location "Houston, TX" --out reviews_yelp_raw.csv
"""

import os
import sys
import csv
import time
import argparse
import logging

try:
    from serpapi import GoogleSearch
except ImportError:
    sys.exit(
        "Missing dependency. Install with:\n"
        "    pip install google-search-results --break-system-packages"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("fetch_reviews_yelp")

FIELDS = ["reviewer_name", "rating", "review_date_raw", "review_text", "useful", "funny", "cool", "source"]


def get_api_key() -> str:
    key = os.environ.get("SERPAPI_KEY")
    if not key:
        sys.exit("SERPAPI_KEY environment variable not set. See fetch_reviews.py header for how to set it.")
    return key


def find_place_id(business: str, location: str, api_key: str) -> str:
    """Look up the Yelp place_id for a business by name + location."""
    log.info(f"Looking up Yelp listing for '{business}' in '{location}'...")
    params = {
        "engine": "yelp",
        "find_desc": business,
        "find_loc": location,
        "api_key": api_key,
    }
    search = GoogleSearch(params)
    result = search.get_dict()

    if "error" in result:
        sys.exit(f"SerpApi error during business lookup: {result['error']}")

    organic = result.get("organic_results", [])
    if not organic:
        sys.exit(
            "No Yelp listing found for that business/location. "
            "Try adjusting --business or --location to match the Yelp page title exactly."
        )

    # Prefer an exact-ish name match, else take the first result
    for r in organic:
        if business.lower() in r.get("title", "").lower():
            log.info(f"Matched: {r.get('title')} (place_id={r.get('place_ids', [None])[0]})")
            return r.get("place_ids", [None])[0]

    first = organic[0]
    log.info(f"No exact match; using first result: {first.get('title')}")
    return first.get("place_ids", [None])[0]


def fetch_all_yelp_reviews(place_id: str, api_key: str, max_reviews: int, sleep_s: float = 1.0):
    all_reviews = []
    params = {
        "engine": "yelp_reviews",
        "place_id": place_id,
        "api_key": api_key,
        "sortby": "date_desc",
    }

    start = 0
    while len(all_reviews) < max_reviews:
        log.info(f"Fetching Yelp reviews (offset {start}, have {len(all_reviews)} so far)...")
        search = GoogleSearch(params)
        result = search.get_dict()

        if "error" in result:
            log.error(f"SerpApi error: {result['error']}")
            break

        reviews = result.get("reviews", [])
        if not reviews:
            log.info("No more Yelp reviews returned. Stopping.")
            break

        for r in reviews:
            all_reviews.append({
                "reviewer_name": r.get("user", {}).get("name", ""),
                "rating": r.get("rating", ""),
                "review_date_raw": r.get("date", ""),
                "review_text": (r.get("comment", {}).get("text", "") or "").replace("\n", " ").strip(),
                "useful": r.get("feedback", {}).get("useful", 0),
                "funny": r.get("feedback", {}).get("funny", 0),
                "cool": r.get("feedback", {}).get("cool", 0),
                "source": "yelp",
            })

        start += len(reviews)
        params["start"] = start
        time.sleep(sleep_s)

        # Yelp listings here are small; stop once results dry up naturally
        if len(reviews) < 5:
            break

    return all_reviews[:max_reviews]


def save_csv(reviews, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(reviews)
    log.info(f"Saved {len(reviews)} Yelp reviews to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--business", default="Wazobia African Market")
    parser.add_argument("--location", default="Houston, TX")
    parser.add_argument("--max-reviews", type=int, default=200)
    parser.add_argument("--out", default="reviews_yelp_raw.csv")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    api_key = get_api_key()
    place_id = find_place_id(args.business, args.location, api_key)
    if not place_id:
        sys.exit("Could not resolve a Yelp place_id for that business.")

    reviews = fetch_all_yelp_reviews(place_id, api_key, args.max_reviews, args.sleep)

    if not reviews:
        log.warning("No Yelp reviews retrieved.")
        sys.exit(1)

    save_csv(reviews, args.out)
    log.info(f"Done. Retrieved {len(reviews)} Yelp reviews.")


if __name__ == "__main__":
    main()
