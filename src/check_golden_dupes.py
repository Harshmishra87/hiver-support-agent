import pandas as pd

golden = pd.read_csv("eval/golden/golden_labeled.csv")
print("Total golden rows:", len(golden))
print("Unique tweet_ids:", golden["tweet_id"].nunique())

dupes = golden[golden.duplicated("tweet_id", keep=False)].sort_values("tweet_id")
print("\nDuplicate tweet_ids in golden set:")
print(dupes[["tweet_id", "customer_text", "intent_label"]].to_string())