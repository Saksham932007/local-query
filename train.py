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