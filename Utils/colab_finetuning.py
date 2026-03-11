

import os
import random
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from google.colab import drive

# --- Settings ---
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 32      
TRAIN_SAMPLE_SIZE = 50000  

CSV_PATH = "/content/FYP1Data_cleaned.csv"  
DRIVE_SAVE_PATH = "/content/drive/MyDrive/FYP_Baseline_Model" 


print("Mounting Google Drive to save the model...")
drive.mount('/content/drive')
os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

print("\n[1/4] Loading and preparing data with synthetic labels...")
df = pd.read_csv(CSV_PATH).dropna()

if TRAIN_SAMPLE_SIZE:
    # Shuffle and sample
    df = df.sample(n=min(TRAIN_SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

questions = df["Questions"].tolist()
ideal_answers = df["Answers"].tolist()

training_questions = []
training_answers = []
training_labels = []

for i in range(len(questions)):
    q = questions[i]
    ans = str(ideal_answers[i])
    
    # 1. Excellent (Label 2) 
    training_questions.append(q)
    training_answers.append(ans)
    training_labels.append(2)
    
    # 2. Average (Label 1) -> Half the context
    half_length = len(ans) // 2
    avg_ans = ans[:half_length] + "..." if half_length > 10 else ans
    training_questions.append(q)
    training_answers.append(avg_ans)
    training_labels.append(1)
    
    # 3. Poor (Label 0) -> Random answer from another question
    random_idx = random.randint(0, len(questions) - 1)
    while random_idx == i:
        random_idx = random.randint(0, len(questions) - 1)
    
    poor_ans = str(ideal_answers[random_idx])
    training_questions.append(q)
    training_answers.append(poor_ans)
    training_labels.append(0)

train_df = pd.DataFrame({
    "Question": training_questions,
    "Answer": training_answers,
    "Label": training_labels
})

print(f"      Total training pairs generated: {len(train_df)}")


print("[2/4] Tokenizing the dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_dataset = Dataset.from_pandas(train_df)

def tokenize_function(examples):
    return tokenizer(
        examples["Question"], 
        examples["Answer"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("Label", "labels")

split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_data = split_dataset["train"]
eval_data  = split_dataset["test"]


print("[3/4] Loading DistilBERT for Classification to GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"      Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

print("[4/4] Starting the Fine-Tuning process...")
training_args = TrainingArguments(
    output_dir="/content/results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="/content/logs",
    logging_steps=100,
    fp16=True,  # Enables Mixed Precision Training (HUGE speedup on Colab GPUs)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()

print(f"\n✅ Training Complete! Saving model to Google Drive at: {DRIVE_SAVE_PATH}")
model.save_pretrained(DRIVE_SAVE_PATH)
tokenizer.save_pretrained(DRIVE_SAVE_PATH)
print("\n🎉 DONE! You can now download the folder from your Google Drive.")
