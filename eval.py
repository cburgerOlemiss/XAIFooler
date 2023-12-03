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

def check_condition1(result, df1, df2, configs):
	attacked_text = result.perturbed_result.attacked_text
	modified_index = list(attacked_text.attack_attrs['modified_indices'])
	if modified_index:
		for j in modified_index:
			to_w_j = attacked_text.words[j]
			if to_w_j.lower() not in df1.get('feature') and \
				to_w_j.lower() in df2.get('feature').values[:configs['top_n']]:
				print(f"FAILED! `{to_w_j}`` appears in top_n but was not in the orgiginal text")
				return False
	return True

def check_condition2(exp1, exp2):
	pred_before = exp1[1]
	pred_after = exp2[1]
	return pred_before == pred_after


def topk_intersection(df1, df2, k=None):
	i1 = df1.get('feature').values[:k]
	i2 = df2.get('feature').values[:k]
	return len([x for x in i1 if x in i2])/k
	
def eval(args, filename=None, use=None):
	if not filename:
		filename = generate_filename(args)

	if not use:
		use = UniversalSentenceEncoder(
			threshold=0.840845057,
			metric="angular",
			compare_against_original=False,
			window_size=100,
			skip_text_shorter_than_window=True,
		)

	tmp = {}
	
	_, dataset_test, categories = load_dataset_custom(args.dataset, args.seed_dataset)
	dataset = textattack.datasets.HuggingFaceDataset(dataset_test)
	dataset = dataset._dataset
	stopwords = load_stopwords()
	data = []
	for i in range(len(dataset)):
		text = dataset[i].get(args.text_col)
		example = textattack.shared.attacked_text.AttackedText(text, stopwords=stopwords)
		if args.min_length and example.num_words_non_stopwords < args.min_length:
			continue
		if args.max_length and example.num_words > args.max_length:
			continue
		label = dataset[i].get(args.label_col)
		data.append(example)
	
	if args.num > 0:
		rng = np.random.default_rng(seed=args.seed_dataset)
		rng.shuffle(data)
		data = data[:args.num]

	data = set(data)

	with open(f'{filename}/config.json', 'r') as f:
		configs = json.loads(f.read())

	results2 = pickle.load(open(f"{filename}/results.pickle", 'rb'))
	results = []
	texts = set()
	for a in results2[::-1]:
		if a['example']:
			if a['example'] in data:
				if a['example'].text not in texts:
					results.append(a)
					texts.add(a['example'].text)

	tmp['Total'] = len(results)

	removed = [a for a in results if a['log']]
	print([a['log'] for a in removed])

	results = [a for a in results if not a['log']]
	tmp['Total Adj'] = len(results)

	# preds_before = np.array([a['exp_before'][1] for a in results])
	# preds_after = np.array([a['exp_after'][1] for a in results])
	# idx = np.where(preds_before == preds_after)[0]
	# results = [results[i] for i in idx]

	rbos = []
	sims = []
	l1s = []
	l11s = []
	new_rbos = []
	new_sms = []
	num_replacements = []
	num_errors = 0
	intersections = []

	for item in results:
		result = item['result']
		exp1 = item['exp_before']
		exp2 = item['exp_after']
		df1 = format_explanation_df(exp1[0], target=exp1[1])
		df2 = format_explanation_df(exp2[0], target=exp2[1])
		baseList = df1.get('feature').values
		targetList = df2.get('feature').values

		if not check_condition2(exp1, exp2) and not check_condition1(result, df1, df2, configs):
			num_errors += 1
			continue

		# RBO
		rbo = item['rbo']
		rbos.append(rbo)


		# INTERSECTION
		topkintersect = topk_intersection(df1, df2, k=configs['top_n'])
		intersections.append(topkintersect)

		# SIMILARITY
		if result: # result can be none if running inherent instability
			sent1 = result.original_result.attacked_text.text
			sent2 = result.perturbed_result.attacked_text.text

			if sent1 != sent2:
				emb1, emb2 = use.encode([sent1, sent2])
				sim = use.sim_metric(torch.tensor(emb1.reshape(1,-1)), torch.tensor(emb2.reshape(1,-1)))[0]
			else:
				sim = 1.0
			# print(sim.numpy())
			# print(sent1)
			# print(sent2)
			# print()

		else:
			sim = 1.0
		sims.append(sim)

		if result:
			modified_index = result.perturbed_result.attacked_text.attack_attrs['modified_indices']
			replacement_words = [result.perturbed_result.attacked_text.words[i] for i in modified_index]
			num_replacements.append(len(replacement_words))

		rboOutput = RBO(targetList[:configs['top_n']], baseList[:configs['top_n']], p=1.0)
		new_rbos.append(rboOutput)

		sm = SM(targetList[:configs['top_n']], baseList[:configs['top_n']])
		new_sms.append(sm)

		df2['rank'] = df2.index
		df1['rank'] = df1.index
		
		# print("===============")
		# print(sent1)
		# print("=>", sent2)
		# print(exp1[1])
		# print(exp2[1])
		# print(df1)
		# print(df2)

		rank1 = df1[:configs['top_n']]['rank'].values
		rank2 = df2.set_index('feature').reindex(df1['feature'])[:configs['top_n']]['rank'].values
		l1 = np.sum(np.abs(rank2 - rank1))

		rank1 = df1[:1]['rank'].values
		rank2 = df2.set_index('feature').reindex(df1['feature'])[:1]['rank'].values
		l11= np.sum(np.abs(rank2 - rank1))

		# print(rank1)
		# print(rank2)
		# print(l1)
		# print("===============")

		l1s.append(l1)
		l11s.append(l11)

		# print()
		# print(targetList[:configs['top_n']])
		# print(baseList[:configs['top_n']])
		# print("rboOutput", rboOutput)
		# print("SM", sm)


	# print(sims)
	# print("COMPARE", (np.array(new_sms) == np.array(new_rbos)).mean())
	for threshold in [0.5, 0.6, 0.7]:
		acc = (np.array(rbos) <= threshold).mean()
		tmp['ACC{}'.format(threshold)] = acc


	tmp['Num Errors'] = "{}/{}".format(num_errors, len(results))
	tmp['RBO Avg'] = np.mean(rbos)

	tmp['L1(Top-n) Avg'] = np.mean(l1s)
	tmp['L1(Top-1) Avg'] = np.mean(l11s)

	tmp['SIM Avg'] = np.mean(sims)
	tmp['SIM std'] = np.std(sims)

	tmp['NewRBO Avg'] = np.mean(new_rbos)
	tmp['NewRBO Std'] = np.std(new_rbos)

	tmp['SM Avg'] = np.mean(new_sms)
	tmp['SM std'] = np.mean(new_sms)

	tmp['Rep Avg'] = np.mean(num_replacements)

	tmp["INST(Top-n)"] = np.mean(intersections)


	df = pd.DataFrame.from_dict([tmp])

	return df

if __name__ == "__main__":
	args = load_args()
	
	print(eval(args))





