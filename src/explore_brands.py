import pandas as pd

RAW_PATH = "data/raw/twcs/twcs.csv"

CANDIDATES = [
    "AmericanAir", "Delta", "British_Airways", "SouthwestAir",  # airline
    "TMobileHelp", "Ask_Spectrum", "VerizonSupport",            # telecom
    "XboxSupport", "AskPlayStation",                            # gaming
    "SpotifyCares",                                             # streaming
]

# Load only what we need: full text isn't required for this check
df = pd.read_csv(
    RAW_PATH,
    usecols=["tweet_id", "author_id", "inbound", "response_tweet_id", "in_response_to_tweet_id"],
)

print(f"Total rows: {len(df):,}\n")

brand_rows = df[df["author_id"].isin(CANDIDATES)]

print(f"{'Brand':<18} {'Brand replies':>14} {'Has response_tweet_id':>22} {'% answered further':>20}")
for brand in CANDIDATES:
    sub = brand_rows[brand_rows["author_id"] == brand]
    total = len(sub)
    if total == 0:
        print(f"{brand:<18} {'N/A - not found':>14}")
        continue
    # response_tweet_id non-null means this brand tweet got a further reply from the customer
    has_further_reply = sub["response_tweet_id"].notna().sum()
    pct = 100 * has_further_reply / total
    print(f"{brand:<18} {total:>14,} {has_further_reply:>22,} {pct:>19.1f}%")