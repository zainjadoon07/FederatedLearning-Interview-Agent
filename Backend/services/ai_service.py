import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Calculate the path to your locally trained model from Stage 1
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    "Utils", 
    "FYP_Baseline_Model_v2"
)

class AIModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Loads the DistilBERT model into RAM natively inside FastAPI."""
        print(f"Loading DistilBERT Model onto {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode for blazing fast inference
            print("DistilBERT Model successfully loaded and ready for scoring!")
        except Exception as e:
            print(f"Failed to load AI model. Error: {e}")

    def evaluate(self, question: str, answer: str) -> int:
        """
        Tokenizes the question and candidate's answer, runs inference, 
        and returns the classification label (0=Poor, 1=Average, 2=Excellent).
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Cannot execute evaluation.")
            
        inputs = self.tokenizer(
            question, 
            answer, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        
        # Strip token_type_ids because DistilBERT ignores them, preventing TypeErrors
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"}
        
        with torch.no_grad():  # Turn off gradients to double inference speed!
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Logits shape: [1, 3] -> grab the highest probability label
            predicted_class = torch.argmax(logits, dim=1).item()
            
        return predicted_class

# Create a singleton instance so we can easily import it anywhere
ai_evaluator = AIModelManager()
