"""
Trivial baseline: always predict the single most common intent in the golden set.
This is the floor every other method must beat.
"""
import pandas as pd
from eval_utils import evaluate_predictions

GOLDEN_PATH = "eval/golden/golden_labeled.csv"

df = pd.read_csv(GOLDEN_PATH)
df = df.dropna(subset=["intent_label"])

majority_class = df["intent_label"].value_counts().idxmax()
print(f"Majority class: {majority_class} ({df['intent_label'].value_counts().max()}/{len(df)} examples)")

y_true = df["intent_label"].tolist()
y_pred = [majority_class] * len(df)

evaluate_predictions(y_true, y_pred, method_name="baseline_trivial_majority_class")