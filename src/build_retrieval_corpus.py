"""
Builds the full historical customer->reply pair corpus for retrieval grounding.
Excludes golden set tweet_ids so retrieval can never find the exact answer key
for an example we're evaluating against - that would be leakage, not grounding.
"""
import pandas as pd

RAW_PATH = "data/raw/twcs/twcs.csv"
GOLDEN_PATH = "eval/golden/golden_labeled.csv"
OUTPUT_PATH = "data/processed/retrieval_corpus.csv"
BRAND = "AmericanAir"
CHUNK_SIZE = 200_000

golden = pd.read_csv(GOLDEN_PATH)
golden_ids = set(golden["tweet_id"].tolist())
print(f"Excluding {len(golden_ids)} golden set tweet_ids from retrieval corpus")

usecols = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
dtypes = {"tweet_id": "int64", "author_id": "string", "inbound": "bool", "text": "string"}

brand_reply_chunks = []
for chunk in pd.read_csv(RAW_PATH, usecols=usecols, dtype=dtypes, chunksize=CHUNK_SIZE):
    brand_reply_chunks.append(chunk[chunk["author_id"] == BRAND])

brand_replies = pd.concat(brand_reply_chunks, ignore_index=True)
brand_replies["in_response_to_tweet_id"] = pd.to_numeric(
    brand_replies["in_response_to_tweet_id"], errors="coerce"
).astype("Int64")

needed_ids = set(brand_replies["in_response_to_tweet_id"].dropna().tolist())

customer_text_chunks = []
for chunk in pd.read_csv(RAW_PATH, usecols=["tweet_id", "inbound", "text"], dtype={"tweet_id": "int64", "inbound": "bool", "text": "string"}, chunksize=CHUNK_SIZE):
    matched = chunk[chunk["tweet_id"].isin(needed_ids) & (chunk["inbound"] == True)]
    if len(matched) > 0:
        customer_text_chunks.append(matched)

customer_tweets = pd.concat(customer_text_chunks, ignore_index=True)
customer_tweets = customer_tweets.rename(columns={"tweet_id": "in_response_to_tweet_id", "text": "customer_text"})[
    ["in_response_to_tweet_id", "customer_text"]
]

pairs = brand_replies.merge(customer_tweets, on="in_response_to_tweet_id", how="left")
pairs = pairs.dropna(subset=["customer_text"])

# Exclude golden set examples. golden_labeled.csv's "tweet_id" is the AmericanAir REPLY's
# own tweet_id (see build_golden_candidates.py), which matches pairs["tweet_id"] here -
# NOT pairs["in_response_to_tweet_id"] (that's the customer's tweet_id, a different ID space).
before = len(pairs)
pairs = pairs[~pairs["tweet_id"].isin(golden_ids)]
print(f"Excluded {before - len(pairs)} pairs matching golden set tweet_ids")

output = pairs[["in_response_to_tweet_id", "customer_text", "text"]].rename(
    columns={"in_response_to_tweet_id": "tweet_id", "text": "americanair_reply"}
)
output = output.drop_duplicates(subset="tweet_id")

output.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(output)} historical pairs to {OUTPUT_PATH}")