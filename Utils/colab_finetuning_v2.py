# ==============================================================================
# 🚀 IMPROVED COLAB FINE-TUNING — v2 (Target: ≥85% Adversarial Accuracy)
# ==============================================================================
# What changed from v1:
#  v1 only trained on 3 strategies: correct answer, half-truncation, random swap.
#  The model completely failed on keyword-stuffing, filler-padding, and scrambling.
#
#  v2 trains on ALL 8 strategies so it learns to handle every adversarial case.
# ==============================================================================
import os
import random
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from google.colab import drive

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_EPOCHS = 4               # 1 extra epoch to absorb harder patterns
BATCH_SIZE = 32
TRAIN_SAMPLE_SIZE = 40000   # 40k base rows × 8 strategies = 320k pairs

CSV_PATH = "/content/FYP1Data_cleaned.csv"
DRIVE_SAVE_PATH = "/content/drive/MyDrive/FYP_Baseline_Model_v2"

# ─────────────────────────────────────────────
# STEP 1: Mount Drive
# ─────────────────────────────────────────────
print("Mounting Google Drive...")
drive.mount('/content/drive')
os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 2: Load data
# ─────────────────────────────────────────────
print("\n[1/4] Loading dataset...")
df = pd.read_csv(CSV_PATH).dropna()
df = df.sample(n=min(TRAIN_SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
questions     = df["Questions"].tolist()
ideal_answers = df["Answers"].tolist()
print(f"      Loaded {len(df):,} Q&A pairs for training")

# ─────────────────────────────────────────────
# STEP 3: Generate AUGMENTED training data
#         8 strategies — covers all failure modes
# ─────────────────────────────────────────────
print("[2/4] Generating augmented training pairs (8 strategies)...")

train_q, train_a, train_labels = [], [], []

for i in range(len(questions)):
    q   = questions[i]
    ans = str(ideal_answers[i])
    words = ans.split()

    # 1. EXCELLENT → Original correct answer
    train_q.append(q); train_a.append(ans); train_labels.append(2)

    # 2. EXCELLENT → Same answer with minor whitespace noise
    noisy = "  ".join(words)  # double spaces — same meaning
    train_q.append(q); train_a.append(noisy); train_labels.append(2)

    # 3. AVERAGE → First 50% of answer
    half = max(10, len(ans) // 2)
    train_q.append(q); train_a.append(ans[:half] + "..."); train_labels.append(1)

    # 4. AVERAGE → First 25% of answer (quarter)
    quarter = max(10, len(ans) // 4)
    train_q.append(q); train_a.append(ans[:quarter] + "..."); train_labels.append(1)

    # 5. AVERAGE → Filler-padded correct answer
    #    Model previously predicted Excellent for this — teach it AVERAGE
    filler = " lorem ipsum dolor sit amet consectetur adipiscing elit"
    train_q.append(q); train_a.append(ans + filler * 2); train_labels.append(1)

    # 6. POOR → Completely random answer from another question
    rand_idx = random.randint(0, len(questions) - 1)
    while rand_idx == i:
        rand_idx = random.randint(0, len(questions) - 1)
    train_q.append(q); train_a.append(str(ideal_answers[rand_idx])); train_labels.append(0)

    # 7. POOR → Word-scrambled answer (same words, wrong order)
    shuffled_words = words.copy()
    random.shuffle(shuffled_words)
    train_q.append(q); train_a.append(" ".join(shuffled_words)); train_labels.append(0)

    # 8. POOR → Keyword stuffing (only the long words from question, repeated)
    keywords = [w for w in q.split() if len(w) > 4][:8]
    if not keywords:
        keywords = q.split()[:4]
    stuffed = " ".join(keywords * 4)
    train_q.append(q); train_a.append(stuffed); train_labels.append(0)

train_df = pd.DataFrame({"Question": train_q, "Answer": train_a, "Label": train_labels})

# Print class balance
counts = train_df["Label"].value_counts().sort_index()
print(f"      Total training pairs: {len(train_df):,}")
print(f"      Label distribution → Poor(0): {counts[0]:,} | Average(1): {counts[1]:,} | Excellent(2): {counts[2]:,}")

# ─────────────────────────────────────────────
# STEP 4: Tokenize
# ─────────────────────────────────────────────
print("[3/4] Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_dataset = Dataset.from_pandas(train_df)

def tokenize_fn(examples):
    return tokenizer(
        examples["Question"],
        examples["Answer"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized = hf_dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.rename_column("Label", "labels")

split = tokenized.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
eval_data  = split["test"]
print(f"      Train: {len(train_data):,}  |  Eval: {len(eval_data):,}")

# ─────────────────────────────────────────────
# STEP 5: Train on GPU
# ─────────────────────────────────────────────
print("[4/4] Training on GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"      Device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

training_args = TrainingArguments(
    output_dir="/content/results_v2",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,     # automatically keeps the best checkpoint
    metric_for_best_model="eval_loss",
    fp16=True,
    logging_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()

# ─────────────────────────────────────────────
# STEP 6: Save to Google Drive
# ─────────────────────────────────────────────
print(f"\n✅ Training complete! Saving to Google Drive...")
model.save_pretrained(DRIVE_SAVE_PATH)
tokenizer.save_pretrained(DRIVE_SAVE_PATH)
print(f"🎉 Saved to: {DRIVE_SAVE_PATH}")
print("Download the 'FYP_Baseline_Model_v2' folder from Google Drive to your local PC.")
