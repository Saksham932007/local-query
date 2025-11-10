from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer, 
    DefaultDataCollator
)
from datasets import load_dataset
import os
import sys

# Global Constants
MODEL_CHECKPOINT = "distilbert-base-cased-distilled-squad"
MODEL_OUTPUT_DIR = "./my-custom-model"
DATASET_NAME = "squad"
DATASET_CONFIG = "plain_text"
MAX_LENGTH = 384
STRIDE = 128
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 2
BATCH_SIZE = 8
HACKATHON_MODE = True  # Set to True for faster, smaller training

def main():
    print("🚀 LocalQuery Fine-tuning Script")
    print("=" * 50)
    
    # Step 1: Load Dataset
    print("📥 Loading SQuAD dataset...")
    try:
        raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)
        print(f"✅ Dataset loaded successfully!")
        print(f"Train set size: {len(raw_datasets['train'])}")
        print(f"Validation set size: {len(raw_datasets['validation'])}")
        
        # HACKATHON MODE: Use smaller subset for faster training
        if HACKATHON_MODE:
            print("⚡ HACKATHON MODE: Using smaller dataset for faster training")
            raw_datasets["train"] = raw_datasets["train"].select(range(1000))
            raw_datasets["validation"] = raw_datasets["validation"].select(range(200))
            print(f"Reduced train set size: {len(raw_datasets['train'])}")
            print(f"Reduced validation set size: {len(raw_datasets['validation'])}")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Step 2: Load Tokenizer
    print("\n🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        print(f"✅ Tokenizer loaded: {MODEL_CHECKPOINT}")
        
        # Check if it's a Fast Tokenizer (required for QA)
        if not hasattr(tokenizer, "is_fast") or not tokenizer.is_fast:
            print("⚠️  Warning: This tokenizer is not a Fast Tokenizer. QA training might not work properly.")
        else:
            print("✅ Fast Tokenizer confirmed - ready for Q&A training")
            
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        sys.exit(1)