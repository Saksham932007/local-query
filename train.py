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
    
    # Step 5: Set up Trainer
    print("\n⚙️  Setting up training configuration...")
    
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=f"{MODEL_OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to=None,  # Disable wandb/tensorboard for hackathon
        push_to_hub=False,
    )
    
    data_collator = DefaultDataCollator()
    print("✅ Training configuration completed!")
    
    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("✅ Trainer instantiated successfully!")
    
    # Step 6: Train!
    print("\n🔥 Starting training...")
    print("=" * 50)
    
    try:
        trainer.train()
        print("\n🎉 Training completed successfully!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ GPU Out of Memory Error: {e}")
            print("💡 Try reducing BATCH_SIZE or MAX_LENGTH in the constants")
            print("💡 Or switch to CPU training (though it will be slower)")
            sys.exit(1)
        else:
            print(f"❌ Training error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected training error: {e}")
        sys.exit(1)
    
    # Step 7: Save
    print("\n💾 Saving fine-tuned model...")
    try:
        trainer.save_model(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        print(f"✅ Model and tokenizer saved to: {MODEL_OUTPUT_DIR}")
        
        print("\n" + "=" * 50)
        print("🎯 Fine-tuning Complete!")
        print(f"📁 Model saved in: {MODEL_OUTPUT_DIR}")
        print("🚀 You can now use the fine-tuned model in app.py")
        print("   by enabling 'Use Fine-tuned Model' in the sidebar!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        sys.exit(1)