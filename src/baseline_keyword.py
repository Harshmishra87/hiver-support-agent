"""
Simple baseline: keyword/rule-based classifier.
No training data, no LLM calls - just pattern matching derived from the taxonomy.
Checked in priority order (top rule wins) to reduce keyword-overlap conflicts.
"""
import re
import pandas as pd
from eval_utils import evaluate_predictions

GOLDEN_PATH = "eval/golden/golden_labeled.csv"

RULES = [
    ("flight_disruption", [
        r"\bdelay(ed|s)?\b", r"\bcancel(led|lation)?\b", r"\bdivert(ed)?\b",
        r"\btarmac\b", r"\bmissed (my |our )?connection\b", r"\bstranded\b", r"\bstuck\b",
    ]),
    ("baggage_issue", [
        r"\bbag(gage)?\b", r"\bluggage\b", r"\bsuitcase\b", r"\bcheck(ed)? bag\b",
    ]),
    ("billing_fee_dispute", [
        r"\bcharge(d)?\b", r"\brefund\b", r"\bfee(s)?\b", r"\bovercharg", r"\bbill(ing)?\b",
        r"\bcompensat", r"\bmoney back\b",
    ]),
    ("onboard_technical_issue", [
        r"\bwi-?fi\b", r"\bentertainment\b", r"\bscreen (broken|not working)\b",
        r"\boutlet\b", r"\bpower\b", r"\bapp (won'?t|bug|crash|issue)\b", r"\bbroken\b", r"\bmalfunction\b",
    ]),
    ("loyalty_aadvantage_inquiry", [
        r"\bmiles?\b", r"\baadvantage\b", r"\bpoints\b", r"\bstatus match\b",
        r"\bgold status\b", r"\bplatinum\b", r"\belite\b", r"\bupgrade list\b",
    ]),
    ("booking_seating_assistance", [
        r"\bstandby\b", r"\bseat(ing)? assign", r"\brebook\b", r"\breservation\b",
        r"\bsit together\b", r"\bboarding pass\b",
    ]),
    ("service_staff_complaint", [
        r"\brude\b", r"\bstaff\b", r"\bemployee(s)?\b", r"\bagent(s)?\b",
        r"\bunprofessional\b", r"\bdisrespect", r"\bcustomer service\b",
    ]),
]

DEFAULT_INTENT = "praise_non_actionable"


def classify_keyword(text):
    text_lower = str(text).lower()
    for intent, patterns in RULES:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    return DEFAULT_INTENT


df = pd.read_csv(GOLDEN_PATH)
df = df.dropna(subset=["intent_label"])

y_true = df["intent_label"].tolist()
y_pred = [classify_keyword(text) for text in df["customer_text"]]

evaluate_predictions(y_true, y_pred, method_name="baseline_simple_keyword")