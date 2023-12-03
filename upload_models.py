import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

models = [
         # 'distilbert-base-uncased-imdb-saved',
         # 'bert-base-uncased-imdb-saved',
         # 'roberta-base-imdb-saved',
         # 'distilbert-base-uncased-md_gender_bias-saved',
         # 'bert-base-uncased-md_gender_bias-saved',
         # 'roberta-base-md_gender_bias-saved',
         # 'bert-base-uncased-s2d-saved',
         # 'distilbert-base-uncased-s2d-saved',
         # 'roberta-base-s2d-saved'
         'distilbert-base-uncased-tweets_hate_speech_detection-saved',
         'bert-base-uncased-tweets_hate_speech_detection-saved',
         'roberta-base-tweets_hate_speech_detection-saved',
         ]

for MODEL_NAME in models:
   tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
   model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

   model.push_to_hub("thaile/{}".format(MODEL_NAME))
   tokenizer.push_to_hub("thaile/{}".format(MODEL_NAME))