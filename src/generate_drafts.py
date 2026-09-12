import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

from prompts import DRAFT_REPLY_SYSTEM_PROMPT, DRAFT_REPLY_USER_TEMPLATE
from retrieval import Retriever

load_dotenv()

GOLDEN_PATH = "eval/golden/golden_labeled.csv"
CLASSIFICATION_RESULTS_PATH = "eval/results/llm_classification_raw.csv"
OUTPUT_PATH = "eval/results/reply_drafts.csv"
MODEL = "gemini-3.1-flash-lite"
DELAY_SECONDS = 5
MAX_RETRIES = 3

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
retriever = Retriever()


def strip_markdown_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def format_historical_examples(matches):
    lines = []
    for i, m in enumerate(matches, 1):
        lines.append(f"Example {i} (similarity {m['similarity']:.2f}):")
        lines.append(f'  Customer: "{m["customer_text"]}"')
        lines.append(f'  AmericanAir: "{m["americanair_reply"]}"')
    return "\n".join(lines)


def draft_reply(customer_text, predicted_intent, retries=0):
    try:
        matches = retriever.retrieve(customer_text, k=3)
        historical_examples = format_historical_examples(matches)

        response = client.models.generate_content(
            model=MODEL,
            contents=DRAFT_REPLY_USER_TEMPLATE.format(
                customer_text=customer_text,
                predicted_intent=predicted_intent,
                historical_examples=historical_examples,
            ),
            config={"system_instruction": DRAFT_REPLY_SYSTEM_PROMPT},
        )
        raw = strip_markdown_fences(response.text)
        parsed = json.loads(raw)
        return parsed["draft_reply"], parsed["grounding_note"], None
    except Exception as e:
        if retries < MAX_RETRIES:
            wait = (retries + 1) * 10
            print(f"  Error, retrying in {wait}s: {e}")
            time.sleep(wait)
            return draft_reply(customer_text, predicted_intent, retries=retries + 1)
        return None, None, f"error: {e}"


def main():
    golden = pd.read_csv(GOLDEN_PATH)
    classifications = pd.read_csv(CLASSIFICATION_RESULTS_PATH)
    classifications = classifications.drop_duplicates(subset="tweet_id", keep="last")

    df = golden.merge(classifications, on="tweet_id", how="inner")
    df = df.dropna(subset=["predicted_intent"])
    print(f"Drafting replies for {len(df)} examples (using real classifier predictions)")

    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["tweet_id"])
        print(f"Resuming: {len(done_ids)} already done")
    else:
        existing = pd.DataFrame(columns=["tweet_id", "draft_reply", "grounding_note", "error"])
        done_ids = set()

    results = existing.to_dict("records")
    remaining = df[~df["tweet_id"].isin(done_ids)]
    est_minutes = (len(remaining) * DELAY_SECONDS) / 60
    print(f"Estimated time: ~{est_minutes:.1f} minutes\n")

    for i, (_, row) in enumerate(remaining.iterrows()):
        draft, grounding, error = draft_reply(row["customer_text"], row["predicted_intent"])
        results.append({
            "tweet_id": row["tweet_id"],
            "draft_reply": draft,
            "grounding_note": grounding,
            "error": error,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"  [{i + 1}/{len(remaining)}] saved checkpoint")
        time.sleep(DELAY_SECONDS)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. Drafts saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()