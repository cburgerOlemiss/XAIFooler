import warnings
warnings.filterwarnings("ignore")

import torch
import math
import numpy
import scipy 
import time

def monkeypath_itemfreq(sampler_indices):
	return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import textattack
import transformers

from utils import *
import json
from timeit import default_timer as timer

import gc
gc.collect()
torch.cuda.empty_cache()

import os
from numpy import linalg as LA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

from optimum.onnxruntime import ORTModelForSequenceClassification
from common import *
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from eval import eval
from argparse import Namespace

if __name__ == "__main__":
	checkpoints = [
		("L2", 'L2_dataset-gb_model-{}-md_gender_bias-saved_s-512_k-10_n-2_-sr-1500_threshold-0.5_seed-1212_modifyrate-0.2_RBOrate-0.46'),
		("XAIFooler0.46", 'dataset-gb_model-{}-md_gender_bias-saved_s-512_k-10_n-2_-sr-1500_threshold-0.5_seed-1212_modifyrate-0.2_RBOrate-0.46'),
		("XAIFooler0.32", 'dataset-gb_model-{}-md_gender_bias-saved_s-512_k-10_n-2_-sr-1500_threshold-0.0_seed-1212_modifyrate-0.2_RBOrate-0.32'),
	]

	# checkpoints = [
	# 	("L2", 'L2_dataset-s2d_model-{}-s2d-saved_s-512_k-10_n-3_-sr-2500_threshold-0.5_seed-1212_modifyrate-0.2_RBOrate-0.62'),
	# 	("XAIFooler", 'dataset-s2d_model-{}-s2d-saved_s-512_k-10_n-3_-sr-2500_threshold-0.5_seed-1212_modifyrate-0.2_RBOrate-0.62'),
	# ]

	# checkpoints = [
	# 	("L2", 'L2_dataset-imdb_model-{}-imdb-saved_s-512_k-10_n-5_-sr-4500_threshold-0.5_seed-1212_modifyrate-0.1_RBOrate-0.75'),
	# 	("XAIFooler", 'dataset-imdb_model-{}-imdb-saved_s-512_k-10_n-5_-sr-4500_threshold-0.5_seed-1212_modifyrate-0.1_RBOrate-0.75'),
	# ]

	keep_cols = ['Model', 'Method', 'NewRBO Avg', 'L1(Top-n) Avg', 'SIM Avg', 'SM std', 'INST(Top-n)']
	col_names = ['Model', 'Method', 'NewRBO Avg(d)', 'L1 Avg(u)', 'Semantics(u)', 'Spearmon Corr(u)', 'Top-n Intersect(d)']

	dfs = []

	use = UniversalSentenceEncoder(
		threshold=0.840845057,
		metric="angular",
		compare_against_original=False,
		window_size=100,
		skip_text_shorter_than_window=True,
	)

	for name, checkpoint in checkpoints:
		for model in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']:
			try:
				args_file = f"./results/{checkpoint.format(model)}/config.json"

				with open(args_file, 'r') as f:
					args = json.loads(f.read())
					args = Namespace(**args)

					df = eval(args, filename=f"./results/{checkpoint.format(model)}/", use=use)
					df['Method'] = name
					df['Model'] = model
					df = df[keep_cols]
					df.columns = col_names
					dfs.append(df)
			except Exception as e:
				print("ERROR", e)
				pass

	df = pd.concat(dfs, axis=0).sort_values(['Model', 'Method'])

	print(df)
