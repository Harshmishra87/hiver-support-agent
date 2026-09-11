import os
import json
import time
from typing import Literal
import pandas as pd
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from google import genai

from prompts import CLASSIFY_SYSTEM_PROMPT, CLASSIFY_USER_TEMPLATE
from eval_utils import evaluate_predictions, INTENTS

load_dotenv()

GOLDEN_PATH = "eval/golden/golden_labeled.csv"
RESULTS_PATH = "eval/results/llm_classification_raw.csv"
MODEL = "gemini-3.1-flash-lite"
DELAY_SECONDS = 5  # conservative pacing for flash-lite free tier
MAX_RETRIES = 3

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found - check your .env file is in the project root")

client = genai.Client(api_key=api_key)


class IntentClassification(BaseModel):
    intent: Literal[
        "flight_disruption", "baggage_issue", "billing_fee_dispute",
        "booking_seating_assistance", "loyalty_aadvantage_inquiry",
        "onboard_technical_issue", "service_staff_complaint", "praise_non_actionable",
    ]
    confidence: float
    reasoning: str


def strip_markdown_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def classify_intent(customer_text, retries=0):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=CLASSIFY_USER_TEMPLATE.format(customer_text=customer_text),
            config={"system_instruction": CLASSIFY_SYSTEM_PROMPT},
        )
        raw = strip_markdown_fences(response.text)
        parsed = json.loads(raw)
        result = IntentClassification(**parsed)
        return result.intent, result.confidence, result.reasoning, None
    except (json.JSONDecodeError, ValidationError) as e:
        return None, None, None, f"parse_error: {e}"
    except Exception as e:
        if retries < MAX_RETRIES:
            wait = (retries + 1) * 10
            print(f"  Error, retrying in {wait}s: {e}")
            time.sleep(wait)
            return classify_intent(customer_text, retries=retries + 1)
        return None, None, None, f"api_error: {e}"


def main():
    df = pd.read_csv(GOLDEN_PATH)
    df = df.dropna(subset=["intent_label"])

    if os.path.exists(RESULTS_PATH):
        existing = pd.read_csv(RESULTS_PATH)
        done_ids = set(existing["tweet_id"])
        print(f"Resuming: {len(done_ids)} already classified")
    else:
        existing = pd.DataFrame(columns=["tweet_id", "predicted_intent", "confidence", "reasoning", "error"])
        done_ids = set()

    results = existing.to_dict("records")
    total = len(df)
    remaining = df[~df["tweet_id"].isin(done_ids)]
    print(f"Classifying {len(remaining)} of {total} examples ({DELAY_SECONDS}s delay between calls)...")
    est_minutes = (len(remaining) * DELAY_SECONDS) / 60
    print(f"Estimated time: ~{est_minutes:.1f} minutes\n")

    for i, (_, row) in enumerate(remaining.iterrows()):
        intent, confidence, reasoning, error = classify_intent(row["customer_text"])
        results.append({
            "tweet_id": row["tweet_id"],
            "predicted_intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "error": error,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            print(f"  [{i + 1}/{len(remaining)}] saved checkpoint")

        time.sleep(DELAY_SECONDS)

        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nDone. Raw predictions saved to {RESULTS_PATH}")

    # Merge with golden labels and evaluate
    results_df = pd.DataFrame(results)
    results_df = results_df.drop_duplicates(subset="tweet_id", keep="last")  # guard against resume-related dupes
    merged = df.merge(results_df, on="tweet_id", how="inner")
    merged = merged.drop_duplicates(subset="tweet_id", keep="last")
    merged = merged.dropna(subset=["predicted_intent"])

    print(f"\n{len(merged)}/{total} examples had valid predictions (errors: {total - len(merged)})")

    y_true = merged["intent_label"].tolist()
    y_pred = merged["predicted_intent"].tolist()
    evaluate_predictions(y_true, y_pred, method_name="llm_intent_classification")


if __name__ == "__main__":
    main()