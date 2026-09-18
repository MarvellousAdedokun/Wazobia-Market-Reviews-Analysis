"""
analyze_reviews_extended.py
----------------------------
Extends analyze_reviews.py with three deeper passes:

  1. Automated topic discovery (TF-IDF + NMF) - lets the data surface its
     own clusters instead of relying only on a hand-picked keyword list.
  2. Fine-grained sub-themes - splits broad buckets (food_quality,
     staff_service) into specific dimensions (taste, temperature,
     freshness, rudeness, speed, competence, etc.)
  3. Local Guide vs. regular reviewer comparison - do power-reviewers
     flag different issues than casual customers?

Run AFTER analyze_reviews.py, using its cleaned output:
    python analyze_reviews_extended.py --in analysis_output/reviews_cleaned.csv --out-dir analysis_output
"""

import re
import argparse
import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ---- Fine-grained sub-theme keyword dictionary ----
SUBTHEMES = {
    "food_quality": {
        "taste": ["delicious", "taste", "flavor", "bland", "seasoned", "spicy", "salt", "maggi"],
        "temperature": ["cold", "lukewarm", "not hot", "reheat", "warm"],
        "freshness": ["fresh", "spoiled", "stale", "rotten", "moldy", "expired"],
        "portion": ["portion", "small", "amount", "plate", "protein"],
    },
    "staff_service": {
        "friendliness": ["kind", "friendly", "nice", "welcoming", "patient", "helpful"],
        "rudeness": ["rude", "disrespect", "attitude", "unprofessional", "yell"],
        "speed": ["slow", "fast", "quick"],
        "competence": ["mistake", "wrong order", "error", "incompetent"],
    },
}

TOPIC_N = 6  # number of topics to discover
TOPIC_TOP_WORDS = 8


def run_topic_model(texts, n_topics=TOPIC_N, top_words=TOPIC_TOP_WORDS):
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=3, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)
    W = model.fit_transform(X)  # doc-topic matrix
    H = model.components_       # topic-term matrix

    topics = []
    for i, topic in enumerate(H):
        top_indices = topic.argsort()[-top_words:][::-1]
        top_terms = [feature_names[j] for j in top_indices]
        topics.append({"topic_id": i, "top_terms": top_terms})

    dominant_topic = W.argmax(axis=1)
    return topics, dominant_topic


def tag_subthemes(text: str, parent_theme: str) -> list:
    if parent_theme not in SUBTHEMES:
        return []
    text_l = text.lower()
    hits = []
    for sub, keywords in SUBTHEMES[parent_theme].items():
        if any(kw in text_l for kw in keywords):
            hits.append(f"{parent_theme}::{sub}")
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="analysis_output/reviews_cleaned.csv")
    parser.add_argument("--out-dir", default="analysis_output")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    df["themes"] = df["themes"].apply(ast.literal_eval)

    # ---- 1. Automated topic discovery ----
    topics, dominant_topic = run_topic_model(df["review_text"].tolist())
    df["auto_topic_id"] = dominant_topic

    topic_report_lines = ["AUTOMATED TOPIC DISCOVERY (TF-IDF + NMF)", "=" * 45, ""]
    for t in topics:
        n_docs = (df["auto_topic_id"] == t["topic_id"]).sum()
        avg_rating = df.loc[df["auto_topic_id"] == t["topic_id"], "rating"].mean()
        topic_report_lines.append(
            f"Topic {t['topic_id']} ({n_docs} reviews, avg rating {avg_rating:.2f}): "
            + ", ".join(t["top_terms"])
        )
    topic_report = "\n".join(topic_report_lines)

    # ---- 2. Fine-grained sub-themes ----
    def get_subthemes(row):
        subs = []
        for parent in ["food_quality", "staff_service"]:
            if parent in row["themes"]:
                subs.extend(tag_subthemes(row["review_text"], parent))
        return subs

    df["subthemes"] = df.apply(get_subthemes, axis=1)

    subtheme_counts = {}
    subtheme_sentiment = {}
    for _, row in df.iterrows():
        for sub in row["subthemes"]:
            subtheme_counts[sub] = subtheme_counts.get(sub, 0) + 1
            subtheme_sentiment.setdefault(sub, []).append(row["sentiment_score"])

    subtheme_lines = ["FINE-GRAINED SUB-THEMES", "=" * 45, ""]
    for sub, count in sorted(subtheme_counts.items(), key=lambda x: -x[1]):
        avg_sent = np.mean(subtheme_sentiment[sub])
        subtheme_lines.append(f"{sub}: {count} mentions, avg sentiment {avg_sent:.3f}")
    subtheme_report = "\n".join(subtheme_lines)

    # ---- 3. Local Guide vs. regular reviewer ----
    df["local_guide"] = df["local_guide"].astype(str).str.lower().map({"true": True, "false": False})
    lg = df[df["local_guide"] == True]
    reg = df[df["local_guide"] == False]

    lg_lines = ["LOCAL GUIDE vs. REGULAR REVIEWER", "=" * 45, ""]
    lg_lines.append(f"Local Guides: {len(lg)} reviews | avg rating {lg['rating'].mean():.2f} | avg sentiment {lg['sentiment_score'].mean():.3f}")
    lg_lines.append(f"Regular reviewers: {len(reg)} reviews | avg rating {reg['rating'].mean():.2f} | avg sentiment {reg['sentiment_score'].mean():.3f}")
    lg_lines.append("")
    lg_lines.append("Theme mention rate by group (% of group's reviews mentioning theme):")

    all_themes = set()
    for t in df["themes"]:
        all_themes.update(t)

    for theme in sorted(all_themes):
        lg_pct = lg["themes"].apply(lambda ts: theme in ts).mean() * 100
        reg_pct = reg["themes"].apply(lambda ts: theme in ts).mean() * 100
        lg_lines.append(f"  {theme}: Local Guides {lg_pct:.1f}% vs. Regular {reg_pct:.1f}%")

    lg_report = "\n".join(lg_lines)

    # ---- Write combined report ----
    full_report = topic_report + "\n\n" + subtheme_report + "\n\n" + lg_report
    out_path = f"{args.out_dir}/extended_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    df.to_csv(f"{args.out_dir}/reviews_cleaned_extended.csv", index=False)

    print(full_report)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {args.out_dir}/reviews_cleaned_extended.csv")


if __name__ == "__main__":
    main()
