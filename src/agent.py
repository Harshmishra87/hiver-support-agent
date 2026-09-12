"""
Consolidated agent logic: classify -> draft -> escalate.
Used by the interactive demo app. Reuses the same prompts and retrieval
as the batch evaluation scripts, so the demo reflects the actual evaluated pipeline.
"""
import os
import json
from dotenv import load_dotenv
from google import genai

from prompts import (
    CLASSIFY_SYSTEM_PROMPT, CLASSIFY_USER_TEMPLATE,
    DRAFT_REPLY_SYSTEM_PROMPT, DRAFT_REPLY_USER_TEMPLATE,
    ESCALATION_SYSTEM_PROMPT, ESCALATION_USER_TEMPLATE,
)
from retrieval import Retriever

load_dotenv()

MODEL = "gemini-3.1-flash-lite"
TIMEOUT_CONFIG = {"http_options": {"timeout": 30000}}

_api_key = os.environ.get("GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key)
_retriever = None  # lazy-loaded, since fitting TF-IDF takes a few seconds


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _strip_markdown_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def classify(customer_text):
    response = _client.models.generate_content(
        model=MODEL,
        contents=CLASSIFY_USER_TEMPLATE.format(customer_text=customer_text),
        config={"system_instruction": CLASSIFY_SYSTEM_PROMPT, **TIMEOUT_CONFIG},
    )
    parsed = json.loads(_strip_markdown_fences(response.text))
    return parsed["intent"], parsed["confidence"], parsed["reasoning"]


def draft(customer_text, predicted_intent):
    retriever = get_retriever()
    matches = retriever.retrieve(customer_text, k=3)
    lines = []
    for i, m in enumerate(matches, 1):
        lines.append(f"Example {i} (similarity {m['similarity']:.2f}):")
        lines.append(f'  Customer: "{m["customer_text"]}"')
        lines.append(f'  AmericanAir: "{m["americanair_reply"]}"')
    historical_examples = "\n".join(lines)

    response = _client.models.generate_content(
        model=MODEL,
        contents=DRAFT_REPLY_USER_TEMPLATE.format(
            customer_text=customer_text,
            predicted_intent=predicted_intent,
            historical_examples=historical_examples,
        ),
        config={"system_instruction": DRAFT_REPLY_SYSTEM_PROMPT, **TIMEOUT_CONFIG},
    )
    parsed = json.loads(_strip_markdown_fences(response.text))
    return parsed["draft_reply"], parsed["grounding_note"], matches


def decide_escalation(customer_text, predicted_intent, confidence):
    response = _client.models.generate_content(
        model=MODEL,
        contents=ESCALATION_USER_TEMPLATE.format(
            customer_text=customer_text, predicted_intent=predicted_intent, confidence=confidence
        ),
        config={"system_instruction": ESCALATION_SYSTEM_PROMPT, **TIMEOUT_CONFIG},
    )
    parsed = json.loads(_strip_markdown_fences(response.text))
    return parsed["escalate"], parsed["reason"]


def run_agent(customer_text):
    """Runs the full pipeline: classify -> draft -> escalate. Returns a dict of all results."""
    intent, confidence, classify_reasoning = classify(customer_text)
    draft_reply, grounding_note, matches = draft(customer_text, intent)
    escalate, escalate_reason = decide_escalation(customer_text, intent, confidence)

    return {
        "intent": intent,
        "confidence": confidence,
        "classify_reasoning": classify_reasoning,
        "draft_reply": draft_reply,
        "grounding_note": grounding_note,
        "retrieved_examples": matches,
        "escalate": escalate,
        "escalate_reason": escalate_reason,
    }