import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

from prompts import DRAFT_REPLY_SYSTEM_PROMPT, DRAFT_REPLY_USER_TEMPLATE
from retrieval import Retriever

load_dotenv()

MODEL = "gemini-3.1-flash-lite"
GOLDEN_PATH = "eval/golden/golden_labeled.csv"
N_TEST = 5

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
retriever = Retriever()


def format_historical_examples(matches):
    lines = []
    for i, m in enumerate(matches, 1):
        lines.append(f"Example {i} (similarity {m['similarity']:.2f}):")
        lines.append(f'  Customer: "{m["customer_text"]}"')
        lines.append(f'  AmericanAir: "{m["americanair_reply"]}"')
    return "\n".join(lines)


def draft_reply(customer_text, predicted_intent):
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

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    parsed = json.loads(raw)
    return parsed, matches


golden = pd.read_csv(GOLDEN_PATH)
sample = golden.sample(N_TEST, random_state=99)
regression_check = golden[golden["tweet_id"] == 1871367]
sample = pd.concat([regression_check, sample]).drop_duplicates(subset="tweet_id")
for _, row in sample.iterrows():
    print("=" * 70)
    print(f"CUSTOMER: {row['customer_text']}")
    print(f"(actual intent label: {row['intent_label']})")
    print()

    result, matches = draft_reply(row["customer_text"], row["intent_label"])
    print(f"DRAFTED REPLY: {result['draft_reply']}")
    print(f"GROUNDING: {result['grounding_note']}")
    print()
    print(f"(historical AA reply for comparison: {row['americanair_reply']})")
    print()

    time.sleep(5)