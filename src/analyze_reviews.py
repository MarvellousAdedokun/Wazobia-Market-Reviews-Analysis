"""
analyze_reviews.py
-------------------
Full analysis pipeline for Wazobia African Market Google reviews.

Stages:
  1. Clean & normalize raw review data
  2. Sentiment scoring (VADER - tuned for short, informal text like reviews)
  3. Rating-vs-sentiment tension detection (the "story" signal)
  4. Theme/keyword extraction across common review topics
  5. Owner-response analysis
  6. Recency bucketing (relative dates -> rough time buckets)
  7. Export: cleaned dataset + summary stats + flagged "story" reviews

Run:
    python analyze_reviews.py --in reviews_raw.csv --out-dir analysis_output
"""

import re
import argparse
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# ---- Theme keyword dictionary (expand as needed) ----
THEMES = {
    "pricing": ["price", "expensive", "cheap", "cost", "overpriced", "affordable", "value", "worth"],
    "food_quality": ["delicious", "fresh", "taste", "flavor", "food", "meal", "dish", "spoiled", "stale"],
    "staff_service": ["staff", "service", "employee", "cashier", "rude", "friendly", "kind", "patient", "helpful"],
    "cleanliness": ["clean", "dirty", "hygiene", "smell", "organized", "neat", "messy"],
    "authenticity": ["authentic", "african", "real", "genuine", "imported", "original"],
    "stock_availability": ["stock", "available", "out of stock", "selection", "variety", "empty shelves"],
    "wait_time": ["wait", "slow", "quick", "fast", "line", "queue", "long time"],
    "portion_size": ["portion", "small", "big", "size", "amount", "plate"],
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def bucket_recency(raw_date: str) -> str:
    """Very rough bucketing of Google's relative date strings."""
    if not isinstance(raw_date, str):
        return "unknown"
    raw_date = raw_date.lower()
    if "day" in raw_date or "hour" in raw_date:
        return "last_week"
    if "week" in raw_date:
        return "last_month"
    if "month" in raw_date:
        if any(str(n) in raw_date for n in range(4, 12)):
            return "3-12_months_ago"
        return "1-3_months_ago"
    if "year" in raw_date:
        return "over_a_year_ago"
    return "unknown"


def score_sentiment(text: str) -> float:
    if not text:
        return 0.0
    return analyzer.polarity_scores(text)["compound"]


def tag_themes(text: str) -> list:
    text_l = text.lower()
    hits = []
    for theme, keywords in THEMES.items():
        if any(kw in text_l for kw in keywords):
            hits.append(theme)
    return hits


def classify_tension(rating: float, sentiment: float) -> str:
    """Flag cases where star rating and text sentiment disagree - these are the story goldmine."""
    if pd.isna(rating):
        return "unknown"
    if rating >= 4 and sentiment < -0.2:
        return "high_rating_negative_text"   # praised with stars, vented in words
    if rating <= 2 and sentiment > 0.3:
        return "low_rating_positive_text"    # rated harshly but text reads positive
    return "aligned"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="reviews_raw.csv")
    parser.add_argument("--out-dir", default="analysis_output")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.infile)
    df["review_text"] = df["review_text"].apply(clean_text)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["review_text"].str.len() > 0].reset_index(drop=True)

    # Sentiment
    df["sentiment_score"] = df["review_text"].apply(score_sentiment)
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"], bins=[-1.01, -0.2, 0.2, 1.01],
        labels=["negative", "neutral", "positive"]
    )

    # Tension (rating vs text mismatch)
    df["tension_flag"] = df.apply(lambda r: classify_tension(r["rating"], r["sentiment_score"]), axis=1)

    # Themes
    df["themes"] = df["review_text"].apply(tag_themes)
    df["theme_count"] = df["themes"].apply(len)

    # Recency
    df["recency_bucket"] = df["review_date_raw"].apply(bucket_recency)

    # Owner response
    df["has_owner_response"] = df["owner_response"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)

    # ---- Save cleaned dataset ----
    clean_path = f"{args.out_dir}/reviews_cleaned.csv"
    df.to_csv(clean_path, index=False)

    # ---- Summary stats ----
    summary = {}
    summary["total_reviews"] = len(df)
    summary["avg_rating"] = round(df["rating"].mean(), 2)
    summary["rating_distribution"] = df["rating"].value_counts().sort_index().to_dict()
    summary["sentiment_distribution"] = df["sentiment_label"].value_counts().to_dict()
    summary["avg_sentiment"] = round(df["sentiment_score"].mean(), 3)
    summary["owner_response_rate"] = round(df["has_owner_response"].mean() * 100, 1)
    summary["tension_counts"] = df["tension_flag"].value_counts().to_dict()

    theme_totals = {}
    for themes in df["themes"]:
        for t in themes:
            theme_totals[t] = theme_totals.get(t, 0) + 1
    summary["theme_mentions"] = dict(sorted(theme_totals.items(), key=lambda x: -x[1]))

    # theme sentiment (which topics correlate with negativity/positivity)
    theme_sentiment = {}
    for theme in THEMES:
        mask = df["themes"].apply(lambda ts: theme in ts)
        if mask.sum() > 0:
            theme_sentiment[theme] = {
                "mentions": int(mask.sum()),
                "avg_sentiment": round(df.loc[mask, "sentiment_score"].mean(), 3),
                "avg_rating": round(df.loc[mask, "rating"].mean(), 2),
            }
    summary["theme_sentiment_breakdown"] = theme_sentiment

    recency_rating = df.groupby("recency_bucket")["rating"].mean().round(2).to_dict()
    summary["avg_rating_by_recency"] = recency_rating

    # ---- Flag the most "story-worthy" reviews ----
    story_reviews = df[df["tension_flag"] != "aligned"].sort_values(
        "sentiment_score", key=lambda s: s.abs(), ascending=False
    )
    story_path = f"{args.out_dir}/story_flagged_reviews.csv"
    story_reviews[["rating", "sentiment_score", "tension_flag", "themes", "review_text"]].to_csv(
        story_path, index=False
    )

    # ---- Write human-readable summary report ----
    report_path = f"{args.out_dir}/summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("WAZOBIA AFRICAN MARKET — REVIEW ANALYSIS SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total reviews analyzed: {summary['total_reviews']}\n")
        f.write(f"Average star rating: {summary['avg_rating']}\n")
        f.write(f"Average sentiment score (VADER compound, -1 to 1): {summary['avg_sentiment']}\n")
        f.write(f"Owner response rate: {summary['owner_response_rate']}%\n\n")

        f.write("Rating distribution:\n")
        for k, v in summary["rating_distribution"].items():
            f.write(f"  {k} stars: {v}\n")
        f.write("\n")

        f.write("Sentiment distribution:\n")
        for k, v in summary["sentiment_distribution"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Rating/sentiment tension (the story signal):\n")
        for k, v in summary["tension_counts"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Theme mentions (how many reviews touch each topic):\n")
        for k, v in summary["theme_mentions"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Theme sentiment breakdown:\n")
        for theme, stats in summary["theme_sentiment_breakdown"].items():
            f.write(f"  {theme}: {stats['mentions']} mentions, "
                    f"avg sentiment {stats['avg_sentiment']}, avg rating {stats['avg_rating']}\n")
        f.write("\n")

        f.write("Average rating by recency bucket:\n")
        for k, v in summary["avg_rating_by_recency"].items():
            f.write(f"  {k}: {v}\n")

    print(f"Cleaned dataset: {clean_path}")
    print(f"Story-flagged reviews: {story_path}")
    print(f"Summary report: {report_path}")
    print("\n--- QUICK PREVIEW ---")
    print(open(report_path, encoding="utf-8").read())


if __name__ == "__main__":
    main()
