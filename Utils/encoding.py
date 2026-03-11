import os
import torch
import pandas as pd
from transformers import AutoTokenizer


MAX_LENGTH  = 128          
BATCH_SIZE  = 1000         
MODEL_NAME  = "distilbert-base-uncased"

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(BASE_DIR, "FYP1Data_cleaned.csv")
SAVE_DIR    = os.path.join(BASE_DIR, "Tensors")
os.makedirs(SAVE_DIR, exist_ok=True)  


print("[1/4] Loading dataset ...")
df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=["Questions", "Answers"])
df = df.reset_index(drop=True)

questions = df["Questions"].astype(str).tolist()
answers   = df["Answers"].astype(str).tolist()

print(f"     Total rows loaded: {len(df):,}")


print("[2/4] Loading DistilBERT tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"     Tokenizer ready: {MODEL_NAME}")


def tokenize_in_batches(text_list, label):
    """
    Tokenizes a list of strings in chunks of BATCH_SIZE.
    Returns two tensors:
        - all_input_ids      (shape: N x MAX_LENGTH)
        - all_attention_mask (shape: N x MAX_LENGTH)
    """
    all_input_ids      = []
    all_attention_mask = []

    total_batches = (len(text_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(text_list), BATCH_SIZE):
        batch_num  = (i // BATCH_SIZE) + 1
        batch_text = text_list[i : i + BATCH_SIZE]

        if batch_num % 10 == 0 or batch_num == 1:
            print(f"     [{label}] Processing batch {batch_num}/{total_batches} ...")

        encoded = tokenizer(
            batch_text,
            padding        = "max_length",   
            truncation     = True,          
            max_length     = MAX_LENGTH,
            return_tensors = None            
        )

        all_input_ids.append(torch.tensor(encoded["input_ids"],      dtype=torch.long))
        all_attention_mask.append(torch.tensor(encoded["attention_mask"], dtype=torch.long))

    all_input_ids      = torch.cat(all_input_ids,      dim=0)
    all_attention_mask = torch.cat(all_attention_mask, dim=0)

    return all_input_ids, all_attention_mask

print("[3/4] Tokenizing Questions ...")
q_input_ids, q_attention_mask = tokenize_in_batches(questions, "Questions")

print("[3/4] Tokenizing Answers ...")
a_input_ids, a_attention_mask = tokenize_in_batches(answers, "Answers")

print("[4/4] Saving tensors to disk ...")

torch.save(q_input_ids,       os.path.join(SAVE_DIR, "q_input_ids.pt"))
torch.save(q_attention_mask,  os.path.join(SAVE_DIR, "q_attention_mask.pt"))
torch.save(a_input_ids,       os.path.join(SAVE_DIR, "a_input_ids.pt"))
torch.save(a_attention_mask,  os.path.join(SAVE_DIR, "a_attention_mask.pt"))

print("\nEncoding complete! Tensors saved to:", SAVE_DIR)
print(f"   Questions tensor shape : {q_input_ids.shape}")   # (N, 128)
print(f"   Answers tensor shape   : {a_input_ids.shape}")   # (N, 128)
print("\nFiles saved:")
print("   Tensors/q_input_ids.pt")
print("   Tensors/q_attention_mask.pt")
print("   Tensors/a_input_ids.pt")
print("   Tensors/a_attention_mask.pt")
