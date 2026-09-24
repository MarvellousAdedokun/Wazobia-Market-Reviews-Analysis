# Wazobia African Market — Multi-Platform Review Analysis

An end-to-end analysis project: pull public reviews for Wazobia African
Market (Houston, TX) from Google and Yelp via licensed APIs, clean and
score them, and mine them for a cross-platform, data-backed story about
the customer experience.

Built as a business/data analyst practice project.

## Pipeline

1. **`src/fetch_reviews.py`** — pulls Google Maps reviews via
   [SerpApi](https://serpapi.com)'s Google Maps Reviews endpoint.
2. **`src/fetch_reviews_yelp.py`** — pulls Yelp reviews via SerpApi's
   Yelp engines (business lookup + paginated reviews).
   Both are licensed, ToS-compliant data providers — neither scrapes
   google.com or yelp.com directly.
3. **`src/analyze_reviews.py`** — cleans raw Google data, scores
   sentiment (VADER), tags reviews by theme, flags rating/sentiment
   mismatches, produces a summary report. The Yelp CSV is cleaned the
   same way (see `docs/` for the Yelp-side output).
4. **`src/analyze_reviews_extended.py`** — deeper pass: automated topic
   discovery (TF-IDF + NMF), fine-grained sub-themes (e.g. splitting
   "food quality" into taste/temperature/freshness/portion), and a
   Local Guide vs. regular reviewer comparison.
5. **`src/combine_platforms.py`** — merges the cleaned Google and Yelp
   datasets and compares theme negativity rates side by side, auto-
   flagging which issues are cross-validated (real on both platforms)
   vs. platform-specific.

## Headline finding

Two issues hold up across **both** Google and Yelp, at similar negative
rates on each platform — these are the real, cross-validated problems:

- **Wait time**: ~25-26% of mentions are negative on both platforms
- **Portion size**: ~17-20% of mentions are negative on both platforms

One issue looked like a red herring until checked across platforms:

- **Pricing**: only 14% of Google mentions are negative, but **43% of
  Yelp mentions are negative**. This isn't noise — Yelp's listing skews
  toward the prepared-food Kitchen side, where portion-for-price
  perception is a live issue; Google's skews toward the grocery Market
  side, where it barely comes up. Pricing is a real complaint, but it's
  concentrated, not universal.

**The story**: Wazobia isn't losing customers over what they sell —
they're losing them over how long it takes to get it, and (specifically
on the prepared-food side) whether the portion feels worth the price.

Full detail in [`docs/cross_platform_comparison.txt`](docs/cross_platform_comparison.txt),
[`docs/summary_report.txt`](docs/summary_report.txt), and
[`docs/extended_analysis.txt`](docs/extended_analysis.txt).

## Setup

```bash
pip install -r requirements.txt
```

Get a free SerpApi key (no credit card required) at serpapi.com, then set
it as an environment variable — **never commit it, never hardcode it**:

```bash
# macOS/Linux
export SERPAPI_KEY="your_key_here"

# Windows (Command Prompt)
set SERPAPI_KEY=your_key_here

# Windows (PowerShell)
$env:SERPAPI_KEY="your_key_here"
```

## Usage

```bash
# 1. Fetch reviews from both platforms
python src/fetch_reviews.py --max-reviews 2000 --out data/raw/reviews_raw.csv
python src/fetch_reviews_yelp.py --out data/raw/reviews_yelp_raw.csv

# 2. Clean + analyze each platform
python src/analyze_reviews.py --in data/raw/reviews_raw.csv --out-dir data/processed
# (Yelp cleaning follows the same clean/sentiment/theme steps - see
#  analyze_reviews.py for the reusable functions)

# 3. Deeper pass on the Google data
python src/analyze_reviews_extended.py --in data/processed/reviews_cleaned.csv --out-dir data/processed

# 4. Cross-platform comparison
python src/combine_platforms.py \
    --google data/processed/reviews_cleaned.csv \
    --yelp data/processed/reviews_yelp_cleaned.csv \
    --out-dir data/processed
```

## Data note

`data/raw/` is gitignored — raw scraped data includes real reviewers'
names and isn't ours to publish. The `*_anonymized.csv` files in
`data/processed/` strip reviewer names/IDs so the analysis is shareable
without republishing anyone's identity.

## Known limitations

- Google listing: 365 total reviews, 192 with text. Yelp listing: 55
  reviews, all with text. Small-business scale, not the 2,000 originally
  assumed — volume was reality-checked against the actual listings.
- Relative Google dates ("3 weeks ago") are bucketed into rough recency
  windows; Yelp dates are exact timestamps.
- Local Guide comparison has only 2 Google Local Guide reviews — too
  small a sample to draw conclusions from; reported as inconclusive
  rather than forced into a finding.
- Automated topic modeling (NMF) found one topic that looked coherent by
  vocabulary but was actually bimodal by sentiment (grouped both extreme
  praise and detailed complaints that happened to share generic
  "store"-related words). Included as a documented limitation of
  keyword/vocabulary-based topic modeling, not presented as a finding.
- Themes are keyword + sub-theme dictionary based, not fully unsupervised.
