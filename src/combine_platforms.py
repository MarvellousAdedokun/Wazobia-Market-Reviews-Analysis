"""
combine_platforms.py
---------------------
Merges the cleaned Google and Yelp datasets and produces a cross-platform
comparison: does each theme show up at similar rates, and is it similarly
negative, on both platforms? This is where a single-source analysis can
mislead - a theme that looks fine on one platform may be a real problem
on another with a different reviewer base.

Run AFTER analyze_reviews.py (for Google) and the equivalent Yelp cleaning
step. Expects two cleaned CSVs with matching columns:
  rating, review_text, sentiment_score, themes (stringified list)

Usage:
    python combine_platforms.py \
        --google analysis_output/reviews_cleaned.csv \
        --yelp analysis_output/reviews_yelp_cleaned.csv \
        --out-dir analysis_output
"""

import argparse
import ast
import pandas as pd


def load(path, source_label):
    df = pd.read_csv(path)
    df["themes"] = df["themes"].apply(ast.literal_eval)
    df["source"] = source_label
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--google", default="analysis_output/reviews_cleaned.csv")
    parser.add_argument("--yelp", default="analysis_output/reviews_yelp_cleaned.csv")
    parser.add_argument("--out-dir", default="analysis_output")
    args = parser.parse_args()

    g = load(args.google, "google")
    y = load(args.yelp, "yelp")

    keep_cols = ["rating", "review_text", "sentiment_score", "themes", "source"]
    combined = pd.concat([g[keep_cols], y[keep_cols]], ignore_index=True)
    combined.to_csv(f"{args.out_dir}/combined_reviews.csv", index=False)

    lines = ["CROSS-PLATFORM COMPARISON: GOOGLE vs. YELP", "=" * 50, ""]
    lines.append(f"Google: n={len(g)}, avg rating={g['rating'].mean():.2f}, avg sentiment={g['sentiment_score'].mean():.3f}")
    lines.append(f"Yelp:   n={len(y)}, avg rating={y['rating'].mean():.2f}, avg sentiment={y['sentiment_score'].mean():.3f}")
    lines.append("")
    lines.append("Note: Yelp's listing covers the prepared-food Kitchen side more heavily;")
    lines.append("Google's covers the grocery Market side more heavily. Different customer")
    lines.append("journeys, which is itself part of why the two platforms diverge.")
    lines.append("")

    all_themes = set()
    for t in list(g["themes"]) + list(y["themes"]):
        all_themes.update(t)

    lines.append(f"{'theme':<20}{'google %':<12}{'google neg%':<14}{'yelp %':<10}{'yelp neg%':<12}")
    rows = []
    for theme in sorted(all_themes):
        g_mask = g["themes"].apply(lambda t: theme in t)
        y_mask = y["themes"].apply(lambda t: theme in t)
        g_pct = g_mask.mean() * 100
        y_pct = y_mask.mean() * 100
        g_neg = (g.loc[g_mask, "sentiment_score"] < -0.2).mean() * 100 if g_mask.sum() > 0 else 0.0
        y_neg = (y.loc[y_mask, "sentiment_score"] < -0.2).mean() * 100 if y_mask.sum() > 0 else 0.0
        rows.append((theme, g_pct, g_neg, y_pct, y_neg))
        lines.append(f"{theme:<20}{g_pct:<12.1f}{g_neg:<14.1f}{y_pct:<10.1f}{y_neg:<12.1f}")

    lines.append("")
    lines.append("READING GUIDE:")
    lines.append("- 'neg%' = share of that theme's mentions that were sentiment-negative")
    lines.append("- A theme with similar neg% on both platforms = a real, cross-validated issue")
    lines.append("- A theme with high neg% on one platform only = may be platform-specific")
    lines.append("  (different customer base or different part of the business being reviewed)")
    lines.append("")

    # Auto-flag the two comparison types
    lines.append("AUTO-FLAGGED PATTERNS:")
    for theme, g_pct, g_neg, y_pct, y_neg in rows:
        if g_neg > 15 and y_neg > 15:
            lines.append(f"  [CROSS-VALIDATED ISSUE] {theme}: negative on both platforms (google {g_neg:.0f}%, yelp {y_neg:.0f}%)")
        elif abs(g_neg - y_neg) > 20:
            lines.append(f"  [PLATFORM-SPECIFIC] {theme}: google {g_neg:.0f}% neg vs yelp {y_neg:.0f}% neg — big gap")

    report = "\n".join(lines)
    out_path = f"{args.out_dir}/cross_platform_comparison.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {args.out_dir}/combined_reviews.csv")


if __name__ == "__main__":
    main()
