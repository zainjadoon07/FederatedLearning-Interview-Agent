"""
==========================================================================
  RIGOROUS ADVERSARIAL EVALUATION — Centralized Baseline Model
==========================================================================
Problem with previous evaluation:
  The original test used the SAME synthetic generation strategy (random answers,
  half-truncated answers) that the model was trained on. The model had
  essentially "memorized" those specific patterns → inflated 99.5% result.

This script tests the model with 7 DIFFERENT adversarial strategies:
  1. Excellent  → Original correct answer (true positive)
  2. Poor       → Completely irrelevant random answer (true negative)
  3. Average    → Answer truncated to FIRST QUARTER only (harder than half)
  4. Tricky     → Correct answer with key facts shuffled/scrambled
  5. Keyword    → Answer that only copies keywords from the question (gibberish logic)
  6. Verbose    → Correct answer padded with irrelevant filler (word soup)
  7. Opposite   → Correct answer from a VERY DIFFERENT domain/topic (cross-domain)

The model should score HIGH on 1, LOW on 2+5, and MID on 3+4+6+7.
If it still gets 99.5%, something is wrong.
==========================================================================
"""

import os
import random
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import time

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Utils", "FYP_Baseline_Model_v2")
CSV_PATH   = os.path.join(BASE_DIR, "FYP1Data_cleaned.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLES_PER_STRATEGY = 300  # 300 × 7 strategies = 2,100 total test cases
BATCH_SIZE = 16             # Smaller batch to be safer
RANDOM_SEED = 777           # Different seed than training (42) AND old eval (99)
random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# STEP 1: LOAD MODEL
# ─────────────────────────────────────────────────────────
print(f"[1/4] Loading model from: {MODEL_PATH}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"      Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# ─────────────────────────────────────────────────────────
# STEP 2: LOAD DATA
# ─────────────────────────────────────────────────────────
print("\n[2/4] Loading dataset...")
df = pd.read_csv(CSV_PATH).dropna().sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
questions     = df["Questions"].tolist()
ideal_answers = df["Answers"].tolist()

N = min(SAMPLES_PER_STRATEGY, len(questions))
print(f"      Dataset size: {len(df):,} rows. Using {N} unique Q&A pairs per strategy.")

# ─────────────────────────────────────────────────────────
# STEP 3: ADVERSARIAL STRATEGY HELPERS
# ─────────────────────────────────────────────────────────
def strategy_excellent(q, ans, all_answers, idx):
    """The actual correct answer. Should score EXCELLENT (2)."""
    return q, ans, 2

def strategy_poor_random(q, ans, all_answers, idx):
    """Completely random answer from another question. Should score POOR (0)."""
    rand_idx = random.choice([j for j in range(len(all_answers)) if j != idx])
    return q, str(all_answers[rand_idx]), 0

def strategy_avg_quarter(q, ans, all_answers, idx):
    """Only the first QUARTER of the answer. Very incomplete. Should score AVERAGE (1)."""
    quarter_len = max(10, len(ans) // 4)
    return q, ans[:quarter_len] + "...", 1

def strategy_tricky_scrambled(q, ans, all_answers, idx):
    """Scrambles all the words in the correct answer. Same words, wrong order.
       Tests if the model actually understands meaning vs just word-matching.
       Expected: POOR or AVERAGE (0 or 1)."""
    words = ans.split()
    random.shuffle(words)
    scrambled = " ".join(words)
    return q, scrambled, 0  # Shuffled = should be poor

def strategy_keyword_stuffing(q, ans, all_answers, idx):
    """Takes keywords from the QUESTION and repeats them.
       No real info — just looks related. Should be POOR (0)."""
    keywords = [w for w in q.split() if len(w) > 4][:8]  # take longer words
    stuffed = " ".join(keywords * 3) if keywords else "relevant important key concept"
    return q, stuffed, 0  # Keyword stuffing = poor

def strategy_verbose_filler(q, ans, all_answers, idx):
    """The correct answer padded with large amounts of irrelevant filler.
       Tests if the model punishes noise. Expected: AVERAGE (1)."""
    filler = " lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt"
    verbose = ans + filler * 3
    return q, verbose, 1  # Filler diluted = average

def strategy_cross_domain(q, ans, all_answers, idx):
    """Correct answer, but from a COMPLETELY DIFFERENT part of the dataset
       (far from idx to maximize domain difference). Should be POOR (0)."""
    # Grab an answer from the opposite end of the dataset
    opposite_idx = (idx + len(all_answers) // 2) % len(all_answers)
    return q, str(all_answers[opposite_idx]), 0

STRATEGIES = [
    ("Excellent (original answer)",       strategy_excellent,      2),
    ("Poor (random other answer)",         strategy_poor_random,    0),
    ("Average (first quarter only)",       strategy_avg_quarter,    1),
    ("Poor (word-scrambled answer)",       strategy_tricky_scrambled, 0),
    ("Poor (keyword stuffing)",            strategy_keyword_stuffing, 0),
    ("Average (filler-padded answer)",     strategy_verbose_filler,   1),
    ("Poor (cross-domain swap)",           strategy_cross_domain,     0),
]

# ─────────────────────────────────────────────────────────
# STEP 4: BUILD THE ADVERSARIAL TEST SET
# ─────────────────────────────────────────────────────────
print("\n[3/4] Building adversarial test set...")

all_test_q      = []
all_test_a      = []
all_true_labels = []
all_strategy_tags = []

for strat_name, strat_fn, _ in STRATEGIES:
    count = 0
    for i in range(N):
        q, a, label = strat_fn(questions[i], str(ideal_answers[i]), ideal_answers, i)
        all_test_q.append(q)
        all_test_a.append(a)
        all_true_labels.append(label)
        all_strategy_tags.append(strat_name)
        count += 1
    print(f"      ✓ '{strat_name}': {count} samples generated")

print(f"\n      Total adversarial test cases: {len(all_test_q)}")

# ─────────────────────────────────────────────────────────
# STEP 5: RUN INFERENCE
# ─────────────────────────────────────────────────────────
print("\n[4/4] Running predictions...")
all_preds = []
total_batches = (len(all_test_q) + BATCH_SIZE - 1) // BATCH_SIZE
start_time = time.time()

with torch.no_grad():
    for i in range(0, len(all_test_q), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        if batch_num % 20 == 0 or batch_num == 1:
            print(f"      Batch {batch_num}/{total_batches}...")

        bq = all_test_q[i : i + BATCH_SIZE]
        ba = all_test_a[i : i + BATCH_SIZE]

        inputs = tokenizer(bq, ba, padding=True, truncation=True, max_length=128, return_tensors="pt")
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)

end_time = time.time()

# ─────────────────────────────────────────────────────────
# STEP 6: DETAILED RESULTS
# ─────────────────────────────────────────────────────────
# Overall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc  = accuracy_score(all_true_labels, all_preds)
prec = precision_score(all_true_labels, all_preds, average="weighted", zero_division=0)
rec  = recall_score(all_true_labels, all_preds, average="weighted", zero_division=0)
f1   = f1_score(all_true_labels, all_preds, average="weighted", zero_division=0)

label_names = {0: "Poor", 1: "Average", 2: "Excellent"}

print("\n" + "="*65)
print("  ADVERSARIAL EVALUATION RESULTS — CENTRALIZED BASELINE MODEL")
print("="*65)
print(f"  Total test cases        : {len(all_test_q)}")
print(f"  Inference time          : {end_time - start_time:.2f}s  ({(end_time-start_time)/len(all_test_q)*1000:.1f}ms/sample)")
print(f"\n  🎯 Overall Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  🎯 Weighted Precision: {prec:.4f}  ({prec*100:.2f}%)")
print(f"  🎯 Weighted Recall   : {rec:.4f}  ({rec*100:.2f}%)")
print(f"  🎯 Weighted F1-Score : {f1:.4f}  ({f1*100:.2f}%)")
print("="*65)

# Per strategy breakdown — the REAL insight
print("\n  📊 RESULTS BY ADVERSARIAL STRATEGY:")
print(f"  {'Strategy':<38} {'Expected':>8}  {'Predicted':>10}  {'Acc%':>6}")
print(f"  {'-'*38} {'-'*8}  {'-'*10}  {'-'*6}")

results_df = pd.DataFrame({
    "strategy": all_strategy_tags,
    "true_label": all_true_labels,
    "pred_label": all_preds,
})

report_lines = []
for strat_name, _, expected_label in STRATEGIES:
    subset = results_df[results_df["strategy"] == strat_name]
    n = len(subset)
    n_correct = (subset["true_label"] == subset["pred_label"]).sum()
    strat_acc = n_correct / n * 100
    pred_counts = subset["pred_label"].value_counts().to_dict()
    pred_summary = ", ".join([f"{label_names[k]}:{v}" for k,v in sorted(pred_counts.items())])
    line = f"  {strat_name:<38} {label_names[expected_label]:>8}  {pred_summary:>30}  {strat_acc:>5.1f}%"
    print(line)
    report_lines.append(line)

print("\n  Full Classification Report:")
target_names = ['Poor (0)', 'Average (1)', 'Excellent (2)']
report = classification_report(all_true_labels, all_preds, target_names=target_names)
print(report)

print("  Confusion Matrix (rows=actual, cols=predicted):")
print("                Poor  Avg  Exc")
cm = confusion_matrix(all_true_labels, all_preds)
for i, row_name in enumerate(["  Poor    ", "  Average ", "  Excellent"]):
    print(f"  {row_name}  {cm[i]}")

# Save full results
save_path = os.path.join(RESULTS_DIR, "adversarial_baseline_metrics.txt")
with open(save_path, "w") as f:
    f.write("ADVERSARIAL BASELINE EVALUATION\n")
    f.write("="*65 + "\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall   : {rec:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n\n")
    f.write("Per-Strategy Results:\n")
    for l in report_lines:
        f.write(l + "\n")
    f.write("\nFull Report:\n")
    f.write(report)

print(f"\n✅ Full adversarial results saved to: {save_path}")
