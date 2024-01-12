import gc
gc.collect()

import torch
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

from datasets import load_dataset
import pandas as pd
import numpy as np
import evaluate

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from argparse import ArgumentParser

from common import load_dataset_custom

parser = ArgumentParser(description='XAIFOOLER Training Model')
parser.add_argument('--model', type=str, default="distilbert-base-uncased")
parser.add_argument('--dataset', type=str, default="hate_speech18")
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--max-length', type=int, default=250)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--seed-dataset', type=int, default=12)

args, unknown = parser.parse_known_args()
#args = parser.parse_args()

MODEL_NAME = args.model
DATASET_NAME = args.dataset
batch_size = args.batch_size
saved_folder = "{}-{}-saved".format(MODEL_NAME, DATASET_NAME)
max_length = args.max_length
seed = args.seed_dataset


dataset_train_valid, dataset_test, categories = load_dataset_custom(DATASET_NAME, seed=args.seed_dataset)
print(dataset_test.features)
print(categories)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(categories))

def tokenize_function(examples):
    return tokenizer(examples["text"], 
	    			add_special_tokens=True,
	    			max_length=max_length if max_length else None,
	    			padding="max_length", 
	    			truncation=True)

tokenized_train = dataset_train_valid['train'].map(tokenize_function, batched=True)
tokenized_valid = dataset_train_valid['test'].map(tokenize_function, batched=True)
tokenized_test = dataset_test.map(tokenize_function, batched=True)

print("TRAIN/VAL/TEST")
print(len(tokenized_train), len(tokenized_valid), len(tokenized_test))

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': metric_acc.compute(predictions=predictions, references=labels),
    		'f1': metric_f1.compute(predictions=predictions, references=labels, average='weighted')}

training_args = TrainingArguments(output_dir="./{}-{}-checkpoints".format(MODEL_NAME, DATASET_NAME), 
								evaluation_strategy="epoch",
								save_strategy="epoch",
								num_train_epochs=args.epochs,
								per_device_train_batch_size=batch_size,
								per_device_eval_batch_size=batch_size,
								warmup_steps=100,
								weight_decay=0.01,
								logging_dir='./logs/',
								logging_steps=10,
								load_best_model_at_end=True
								)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,   
    compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()

result = trainer.evaluate(tokenized_test)
print("Evaluation on TEST SET")
print(result)

tokenizer.save_pretrained(saved_folder)
model.save_pretrained(saved_folder)
