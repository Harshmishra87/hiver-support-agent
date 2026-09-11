import pandas as pd

golden = pd.read_csv("eval/golden/golden_labeled.csv")
print("golden tweet_id dtype:", golden["tweet_id"].dtype)
print("golden tweet_id sample:", golden["tweet_id"].head(5).tolist())

RAW_PATH = "data/raw/twcs/twcs.csv"
BRAND = "AmericanAir"
CHUNK_SIZE = 200_000

usecols = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
dtypes = {"tweet_id": "int64", "author_id": "string", "inbound": "bool", "text": "string"}

brand_reply_chunks = []
for chunk in pd.read_csv(RAW_PATH, usecols=usecols, dtype=dtypes, chunksize=CHUNK_SIZE):
    brand_reply_chunks.append(chunk[chunk["author_id"] == BRAND])
brand_replies = pd.concat(brand_reply_chunks, ignore_index=True)

print("\nbrand_replies tweet_id dtype:", brand_replies["tweet_id"].dtype)
print("brand_replies row count:", len(brand_replies))

golden_ids = set(golden["tweet_id"].tolist())
overlap = brand_replies["tweet_id"].isin(golden_ids).sum()
print(f"\nDirect overlap check: {overlap} of brand_replies match a golden tweet_id")

sample_id = golden["tweet_id"].iloc[0]
print(f"\nLooking for golden tweet_id {sample_id} in brand_replies directly:")
match = brand_replies[brand_replies["tweet_id"] == sample_id]
print(match[["tweet_id", "text"]].to_string() if len(match) else "NOT FOUND in brand_replies")