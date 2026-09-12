import pandas as pd

golden = pd.read_csv("eval/golden/golden_labeled.csv")
predictions = pd.read_csv("eval/results/llm_classification_raw.csv").drop_duplicates(subset="tweet_id", keep="last")

merged = golden.merge(predictions, on="tweet_id", how="inner").dropna(subset=["predicted_intent"])
misclassified = merged[merged["intent_label"] != merged["predicted_intent"]]

print("=== billing_fee_dispute missed (actual=billing, predicted=something else) ===")
billing_missed = misclassified[misclassified["intent_label"] == "billing_fee_dispute"]
for _, row in billing_missed.head(3).iterrows():
    print(f"\nTweet: {row['customer_text'][:150]}")
    print(f"Actual: billing_fee_dispute | Predicted: {row['predicted_intent']}")
    print(f"Model's reasoning: {row['reasoning']}")

print("\n\n=== praise_non_actionable over-predicted (predicted=praise, actual=something else) ===")
praise_overpredict = misclassified[misclassified["predicted_intent"] == "praise_non_actionable"]
for _, row in praise_overpredict.head(3).iterrows():
    print(f"\nTweet: {row['customer_text'][:150]}")
    print(f"Actual: {row['intent_label']} | Predicted: praise_non_actionable")
    print(f"Model's reasoning: {row['reasoning']}")