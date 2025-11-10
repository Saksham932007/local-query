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
    
    # Step 3: Define preprocessing function
    def preprocess_function(examples):
        """Preprocess the SQuAD dataset for Q&A fine-tuning"""
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    # Apply preprocessing to the dataset
    print("\n🔄 Preprocessing dataset...")
    try:
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        print("✅ Dataset preprocessing completed!")
        print(f"Tokenized train size: {len(tokenized_datasets['train'])}")
        print(f"Tokenized validation size: {len(tokenized_datasets['validation'])}")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        sys.exit(1)
    
    # Step 4: Load Model
    print("\n🤖 Loading model for fine-tuning...")
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
        print(f"✅ Model loaded: {MODEL_CHECKPOINT}")
        print(f"Model parameters: {model.num_parameters():,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)