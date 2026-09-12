import pandas as pd
from sklearn.metrics import cohen_kappa_score

HUMAN_PATH = "eval/results/human_judge_scores.csv"
JUDGE_PATH = "eval/results/judge_scores.csv"

DIMENSIONS = ["groundedness", "tone_appropriateness", "resolves_the_ask", "overall"]

human = pd.read_csv(HUMAN_PATH)
human = human.dropna(subset=["overall"])
judge = pd.read_csv(JUDGE_PATH)
judge = judge.dropna(subset=["overall"])

merged = human.merge(judge, on="tweet_id", how="inner", suffixes=("_human", "_judge"))
print(f"Human-scored: {len(human)}")
print(f"Judge-scored: {len(judge)}")
print(f"Overlap (both scored, usable for agreement): {len(merged)}\n")

if len(merged) == 0:
    print("No overlap yet - the judge script likely hasn't reached these specific tweet_ids.")
    print("Either wait for judge_replies.py to finish more rows, or re-run it to cover the rest.")
else:
    print("=" * 60)
    print("JUDGE vs HUMAN AGREEMENT (quadratic weighted Cohen's kappa)")
    print("=" * 60)

    for dim in DIMENSIONS:
        h_col = f"{dim}_human"
        j_col = f"{dim}_judge"
        h_scores = merged[h_col].astype(int)
        j_scores = merged[j_col].astype(int)

        kappa = cohen_kappa_score(h_scores, j_scores, weights="quadratic")
        raw_agreement = (h_scores == j_scores).mean()
        avg_diff = (h_scores - j_scores).abs().mean()

        print(f"\n{dim}:")
        print(f"  Quadratic weighted kappa: {kappa:.3f}")
        print(f"  Raw exact-match agreement: {raw_agreement:.1%}")
        print(f"  Mean absolute difference: {avg_diff:.2f} points")

    merged["overall_diff"] = (merged["overall_human"].astype(int) - merged["overall_judge"].astype(int)).abs()
    big_disagreements = merged[merged["overall_diff"] >= 2].sort_values("overall_diff", ascending=False)

    print(f"\n{'=' * 60}")
    print(f"BIG DISAGREEMENTS (overall score differs by 2+) - n={len(big_disagreements)}")
    print(f"{'=' * 60}")
    for _, row in big_disagreements.iterrows():
        print(f"\ntweet_id {row['tweet_id']}:")
        print(f"  Customer: {row['customer_text'][:100]}")
        print(f"  Draft: {row['draft_reply'][:100]}")
        print(f"  Human overall: {row['overall_human']} | Judge overall: {row['overall_judge']}")
        print(f"  Judge's justification: {row['justification']}")

    merged.to_csv("eval/results/judge_human_comparison.csv", index=False)
    print(f"\nFull comparison saved to eval/results/judge_human_comparison.csv")