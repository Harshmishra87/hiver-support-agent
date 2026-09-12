import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

from prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE

load_dotenv()

GOLDEN_PATH = "eval/golden/golden_labeled.csv"
CLASSIFICATION_RESULTS_PATH = "eval/results/llm_classification_raw.csv"
DRAFTS_PATH = "eval/results/reply_drafts.csv"
OUTPUT_PATH = "eval/results/judge_scores.csv"
MODEL = "gemini-3.1-flash-lite"
DELAY_SECONDS = 5
MAX_RETRIES = 3

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def strip_markdown_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def judge_reply(customer_text, predicted_intent, draft_reply, retries=0):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=JUDGE_USER_TEMPLATE.format(
                customer_text=customer_text, predicted_intent=predicted_intent, draft_reply=draft_reply
            ),
            config={"system_instruction": JUDGE_SYSTEM_PROMPT, "http_options": {"timeout": 30000}},
        )
        raw = strip_markdown_fences(response.text)
        parsed = json.loads(raw)
        return parsed, None
    except Exception as e:
        if retries < MAX_RETRIES:
            wait = (retries + 1) * 10
            print(f"  Error, retrying in {wait}s: {e}")
            time.sleep(wait)
            return judge_reply(customer_text, predicted_intent, draft_reply, retries=retries + 1)
        return None, f"error: {e}"


def main():
    golden = pd.read_csv(GOLDEN_PATH)
    classifications = pd.read_csv(CLASSIFICATION_RESULTS_PATH).drop_duplicates(subset="tweet_id", keep="last")
    drafts = pd.read_csv(DRAFTS_PATH).drop_duplicates(subset="tweet_id", keep="last")

    df = golden.merge(classifications, on="tweet_id", how="inner")
    df = df.merge(drafts, on="tweet_id", how="inner")
    df = df.dropna(subset=["predicted_intent", "draft_reply"])
    print(f"Judging {len(df)} drafted replies")

    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["tweet_id"])
        print(f"Resuming: {len(done_ids)} already judged")
    else:
        existing = pd.DataFrame(columns=[
            "tweet_id", "groundedness", "tone_appropriateness", "resolves_the_ask", "overall", "justification", "error"
        ])
        done_ids = set()

    results = existing.to_dict("records")
    remaining = df[~df["tweet_id"].isin(done_ids)]
    est_minutes = (len(remaining) * DELAY_SECONDS) / 60
    print(f"Estimated time if uninterrupted: ~{est_minutes:.1f} minutes")
    print("If daily quota runs out partway through, this is safe to stop and re-run later - it will resume.\n")

    for i, (_, row) in enumerate(remaining.iterrows()):
        scores, error = judge_reply(row["customer_text"], row["predicted_intent"], row["draft_reply"])
        record = {"tweet_id": row["tweet_id"], "error": error}
        if scores:
            record.update(scores)
        results.append(record)

        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"  [{i + 1}/{len(remaining)}] saved checkpoint")

        time.sleep(DELAY_SECONDS)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone (or stopped). Scores saved to {OUTPUT_PATH}")

    final = pd.DataFrame(results)
    scored = final.dropna(subset=["overall"])
    if len(scored) > 0:
        print(f"\n{len(scored)} replies scored so far:")
        print(f"  Mean groundedness: {scored['groundedness'].mean():.2f}")
        print(f"  Mean tone_appropriateness: {scored['tone_appropriateness'].mean():.2f}")
        print(f"  Mean resolves_the_ask: {scored['resolves_the_ask'].mean():.2f}")
        print(f"  Mean overall: {scored['overall'].mean():.2f}")


if __name__ == "__main__":
    main()