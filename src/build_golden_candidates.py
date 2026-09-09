import pandas as pd

RAW_PATH = "data/raw/twcs/twcs.csv"
BRAND = "AmericanAir"
CHUNK_SIZE = 200_000
N_CANDIDATES = 300  # oversample above the 150-250 target for labeling attrition

usecols = ["tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"]
dtypes = {
    "tweet_id": "int64",
    "author_id": "string",
    "inbound": "bool",
    "text": "string",
}

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

pairs["text_len"] = pairs["customer_text"].str.len()
pairs = pairs[pairs["text_len"] >= 15]  # drop near-empty tweets

candidates = pairs.sample(n=min(N_CANDIDATES, len(pairs)), random_state=2024)

output = candidates[["tweet_id", "customer_text", "text"]].rename(
    columns={"text": "americanair_reply"}
)
output["intent_label"] = ""
output["escalate"] = ""
output["escalate_reason"] = ""

output.to_csv("eval/golden/golden_candidates.csv", index=False)
print(f"Saved {len(output)} candidates to eval/golden/golden_candidates.csv")
print("Columns: tweet_id, customer_text, americanair_reply, intent_label, escalate, escalate_reason")