"""
Shared evaluation harness for intent classification.
Used by every method (baselines, LLM agent) so results are directly comparable.
"""
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

INTENTS = [
    "flight_disruption",
    "baggage_issue",
    "billing_fee_dispute",
    "booking_seating_assistance",
    "loyalty_aadvantage_inquiry",
    "onboard_technical_issue",
    "service_staff_complaint",
    "praise_non_actionable",
]


def evaluate_predictions(y_true, y_pred, method_name, save_dir="eval/results"):
    """
    Compares predictions against golden labels, prints a report,
    and saves results to eval/results/{method_name}.json for later comparison.
    """
    os.makedirs(save_dir, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=INTENTS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=INTENTS)

    print(f"\n{'=' * 60}")
    print(f"Method: {method_name}")
    print(f"{'=' * 60}")
    print(f"Overall accuracy: {accuracy:.3f} ({int(accuracy * len(y_true))}/{len(y_true)})")
    print(f"\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=INTENTS, zero_division=0))

    results = {
        "method": method_name,
        "n_examples": len(y_true),
        "accuracy": accuracy,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": INTENTS,
    }

    out_path = os.path.join(save_dir, f"{method_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    return results