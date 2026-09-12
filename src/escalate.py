import os
import json
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from dotenv import load_dotenv
from google import genai

from prompts import ESCALATION_SYSTEM_PROMPT, ESCALATION_USER_TEMPLATE

load_dotenv()

GOLDEN_PATH = "eval/golden/golden_labeled.csv"
CLASSIFICATION_RESULTS_PATH = "eval/results/llm_classification_raw.csv"
OUTPUT_PATH = "eval/results/escalation_raw.csv"
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


def decide_escalation(customer_text, predicted_intent, confidence, retries=0):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=ESCALATION_USER_TEMPLATE.format(
                customer_text=customer_text, predicted_intent=predicted_intent, confidence=confidence
            ),
            config={"system_instruction": ESCALATION_SYSTEM_PROMPT},
        )
        raw = strip_markdown_fences(response.text)
        parsed = json.loads(raw)
        escalate = parsed["escalate"].lower().strip()
        if escalate not in ("yes", "no"):
            raise ValueError(f"Invalid escalate value: {escalate}")
        return escalate, parsed["reason"], None
    except Exception as e:
        if retries < MAX_RETRIES:
            wait = (retries + 1) * 10
            print(f"  Error, retrying in {wait}s: {e}")
            time.sleep(wait)
            return decide_escalation(customer_text, predicted_intent, confidence, retries=retries + 1)
        return None, None, f"error: {e}"


def main():
    golden = pd.read_csv(GOLDEN_PATH)
    golden = golden.dropna(subset=["intent_label", "escalate"])

    classifications = pd.read_csv(CLASSIFICATION_RESULTS_PATH)
    classifications = classifications.drop_duplicates(subset="tweet_id", keep="last")
    df = golden.merge(classifications, on="tweet_id", how="inner")
    df = df.dropna(subset=["predicted_intent"])
    print(f"Running escalation decisions on {len(df)} examples (using real classifier predictions)")

    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["tweet_id"])
        print(f"Resuming: {len(done_ids)} already done")
    else:
        existing = pd.DataFrame(columns=["tweet_id", "predicted_escalate", "reason", "error"])
        done_ids = set()

    results = existing.to_dict("records")
    remaining = df[~df["tweet_id"].isin(done_ids)]
    est_minutes = (len(remaining) * DELAY_SECONDS) / 60
    print(f"Estimated time: ~{est_minutes:.1f} minutes\n")

    for i, (_, row) in enumerate(remaining.iterrows()):
        escalate, reason, error = decide_escalation(
            row["customer_text"], row["predicted_intent"], row["confidence"]
        )
        results.append({
            "tweet_id": row["tweet_id"],
            "predicted_escalate": escalate,
            "reason": reason,
            "error": error,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"  [{i + 1}/{len(remaining)}] saved checkpoint")
        time.sleep(DELAY_SECONDS)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    results_df = pd.DataFrame(results).drop_duplicates(subset="tweet_id", keep="last")
    merged = df.merge(results_df, on="tweet_id", how="inner").dropna(subset=["predicted_escalate"])

    y_true = merged["escalate"].str.lower().str.strip()
    y_pred = merged["predicted_escalate"].str.lower().str.strip()

    print(f"\n{'=' * 60}")
    print(f"Escalation decision evaluation (n={len(merged)})")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred, labels=["yes", "no"], zero_division=0))

    false_negatives = merged[(y_true == "yes") & (y_pred == "no")]
    print(f"\nFALSE NEGATIVES (should have escalated, didn't) - n={len(false_negatives)}:")
    for _, row in false_negatives.iterrows():
        print(f"  - {row['customer_text'][:100]}")

    merged.to_csv("eval/results/escalation_evaluated.csv", index=False)


if __name__ == "__main__":
    main()