import pandas as pd

RAW_PATH = "data/raw/twcs/twcs.csv"
BRAND = "AmericanAir"
CHUNK_SIZE = 200_000

usecols = ["tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"]
dtypes = {
    "tweet_id": "int64",
    "author_id": "string",
    "inbound": "bool",
    "text": "string",
    # response_tweet_id / in_response_to_tweet_id left as default - contain NaNs, can't be int64 directly
}

# --- Pass 1: collect AmericanAir's own replies ---
brand_reply_chunks = []
for chunk in pd.read_csv(RAW_PATH, usecols=usecols, dtype=dtypes, chunksize=CHUNK_SIZE):
    brand_reply_chunks.append(chunk[chunk["author_id"] == BRAND])

brand_replies = pd.concat(brand_reply_chunks, ignore_index=True)
brand_replies["in_response_to_tweet_id"] = pd.to_numeric(
    brand_replies["in_response_to_tweet_id"], errors="coerce"
).astype("Int64")

print(f"Total {BRAND} replies found: {len(brand_replies):,}")

needed_ids = set(brand_replies["in_response_to_tweet_id"].dropna().tolist())
print(f"Unique customer tweet_ids to look up: {len(needed_ids):,}")

# --- Pass 2: find those specific customer tweets ---
customer_text_chunks = []
for chunk in pd.read_csv(RAW_PATH, usecols=["tweet_id", "inbound", "text"], dtype={"tweet_id": "int64", "inbound": "bool", "text": "string"}, chunksize=CHUNK_SIZE):
    matched = chunk[chunk["tweet_id"].isin(needed_ids) & (chunk["inbound"] == True)]
    if len(matched) > 0:
        customer_text_chunks.append(matched)

customer_tweets = pd.concat(customer_text_chunks, ignore_index=True)
customer_tweets = customer_tweets.rename(columns={"tweet_id": "in_response_to_tweet_id", "text": "customer_text"})[
    ["in_response_to_tweet_id", "customer_text"]
]

print(f"Matched customer tweets: {len(customer_tweets):,}")

# --- Join and sample ---
pairs = brand_replies.merge(customer_tweets, on="in_response_to_tweet_id", how="left")
sample = pairs[["customer_text", "text"]].dropna().sample(30, random_state=42)
sample.columns = ["customer_message", "americanair_reply"]
sample.to_csv("data/processed/americanair_sample_pairs.csv", index=False)

print(f"\nSuccessfully paired: {pairs['customer_text'].notna().sum():,}")
print("Saved 30 random pairs to data/processed/americanair_sample_pairs.csv")