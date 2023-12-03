import warnings
warnings.filterwarnings("ignore")

# from sklearnex import patch_sklearn
# patch_sklearn(global_patch=True)

import torch
import math
import numpy
import scipy 

def monkeypath_itemfreq(sampler_indices):
	return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import textattack
import transformers

from utils import RANDOM_BASELINE, ADV_XAI_RBO
from timeit import default_timer as timer

import gc
gc.collect()
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

from optimum.onnxruntime import ORTModelForSequenceClassification
from common import *
import pickle
from tqdm import tqdm
import numpy as np
import os
import json
import time

from argparse import ArgumentParser

def save(results, filename):
	with open('{}/results.pickle'.format(filename), 'wb') as handle:
		pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print("UPDATED TO", filename)

def load(filename):
	results = None
	try:
		results = pickle.load(open(f"{filename}/results.pickle", 'rb'))
	except:
		pass
	return results

if __name__ == "__main__":
	args = load_args()
	filename = generate_filename(args)
	
	print("+++++++++++++++++++++++++++++++++++")
	print(filename)
	print(args)
	print("+++++++++++++++++++++++++++++++++++")

	try:
		os.makedirs(filename)
	except:
		pass

	with open(f'{filename}/config.json', 'w') as f:
		json.dump(args.__dict__, f, indent=2)

	models = ['distilbert-base-uncased-imdb-saved',
		 'bert-base-uncased-imdb-saved',
		 'roberta-base-imdb-saved',
		 'distilbert-base-uncased-md_gender_bias-saved',
		 'bert-base-uncased-md_gender_bias-saved',
		 'roberta-base-md_gender_bias-saved',
		 'bert-base-uncased-s2d-saved',
		 'distilbert-base-uncased-s2d-saved',
		 'roberta-base-s2d-saved']

	if args.model.replace('thaile/','') not in models:
		print("CAUTION! You are running a model not in the model cards.")

	model = ORTModelForSequenceClassification.from_pretrained(args.model, 
															export=True, 
															provider="CUDAExecutionProvider", 
															use_io_binding=True)
	tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=True)
	if args.max_length:
		tokenizer.model_max_length = args.max_length

	model.to(args.device)

	model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

	outputName = "output"
	startIndex = 0
	csvName = outputName + str(startIndex) + "_log.csv"
	folderName = "outputName" + str(startIndex)

	_, dataset_test, categories = load_dataset_custom(args.dataset, args.seed_dataset)
	dataset = textattack.datasets.HuggingFaceDataset(dataset_test)
	print(categories)

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
		data.append((example, label))

	# print("data", data[:5])
	categories = list(np.unique([tmp[1] for tmp in data]))
	print("CATEGORIES", categories)

	if args.num > 0:
		rng = np.random.default_rng(seed=args.seed_dataset)
		rng.shuffle(data)
		data = data[:args.num]
		# for i in data:
		#     print(i)

	pbar = tqdm(range(0, len(data)), bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar}')
	# pbar = range(len(dataset))

	def generate_attacker(ATTACK_CLASS, args, custom_seed=None, greedy_search=True):
		attack = ATTACK_CLASS.build(model_wrapper,
									categories = categories,
									featureSelector = args.top_n, 
									limeSamples = args.lime_sr,
									random_seed = args.seed if not custom_seed else custom_seed,
									success_threshold=args.success_threshold,
									model_batch_size=args.batch_size,
									max_candidates=args.max_candidate,
									logger=pbar if args.debug else None,
									modification_rate=args.modify_rate,
									rbo_p = args.rbo_p,
									similarity_measure=args.similarity_measure,
									greedy_search=greedy_search
									)

		attack_args = textattack.AttackArgs(num_examples=1,
											random_seed=args.seed if not custom_seed else custom_seed,
											log_to_csv=csvName, 
											checkpoint_interval=250, 
											checkpoint_dir="./checkpoints", 
											disable_stdout=False,
											)

		attacker = textattack.Attacker(attack, textattack.datasets.Dataset([]), attack_args)

		return attacker

	if args.method == "xaifooler":
		attacker = generate_attacker(ADV_XAI_RBO, args, custom_seed=None)

	elif args.method == "inherent":
		attacker1 = generate_attacker(ADV_XAI_RBO, args, custom_seed=np.random.choice(1000))
		attacker2 = generate_attacker(ADV_XAI_RBO, args, custom_seed=np.random.choice(1000))

	elif args.method == "random":
		attacker = generate_attacker(RANDOM_BASELINE, args, custom_seed=None, greedy_search=True)

	elif args.method == "truerandom":
		attacker = generate_attacker(RANDOM_BASELINE, args, custom_seed=None, greedy_search=False)

	results = []

	if not args.rerun:
		previous_results = load(filename)
		if previous_results:
			print("LOADED PREVIOUS RESULTS", len(previous_results))
			previous_texts = set([result['example'].text for result in previous_results if not result['log']])
			print(previous_texts)
			results = previous_results

	rbos = []
	for i in pbar:
		# try:
			example, label = data[i]
			print("****TEXT*****")
			print("Text", example.text)
			print("Label", label)
			print("# words (ignore stopwords)", example.num_words_non_stopwords)

			if not args.rerun and previous_results and example.text in previous_texts:
				print("ALREADY DONE, IGNORE...")
				continue
			# #soft split
			# if args.max_length:
			#     text = " ".join(text.split()[:args.max_length])

			if args.method in set(["xaifooler", "random", "truerandom"]):
				output = attacker.attack.goal_function.get_output(example)
				result = attacker.attack.attack(example, output)
				if result:
					print(result.__str__(color_method="ansi") + "\n")

					sent1 = result.original_result.attacked_text.text
					sent2 = result.perturbed_result.attacked_text.text

					exp1 = attacker.attack.goal_function.generateExplanation(sent1)
					exp2 = attacker.attack.goal_function.generateExplanation(sent2)

				else:
					print("PREDICTION MISMATCHED WITH EXPLANTION")
					results.append({'example': example, 'result': None, 'exp_before': None, 'exp_after': None, 'rbo': None, 'log': 'prediction mismatched'})
					if not args.debug:
						save(results, filename)
					continue

			elif args.method == "inherent":
				result = None

				sent1 = example.text
				sent2 = example.text

				exp1 = attacker1.attack.goal_function.generateExplanation(sent1)
				exp2 = attacker2.attack.goal_function.generateExplanation(sent2)
			
			print("Base prediction", exp1[1])
			print("Attacked prediction", exp2[1])
			print("sent1", sent1)
			print("sent2", sent2)

			df1 = format_explanation_df(exp1[0], target=exp1[1])
			df2 = format_explanation_df(exp2[0], target=exp2[1])
			print(df1)
			print(df2)

			targetList = df2.get('feature').values
			baseList = df1.get('feature').values

			rboOutput = RBO(targetList, baseList, p=args.rbo_p)
			print("rboOutput", rboOutput)
			rbos.append(rboOutput)
			# pbar.set_description(f"#{i} | Text: {text[:20]}... | RBO Score: {round(rboOutput,2)}")
			pbar.set_description('||Average RBO={}||'.format(np.mean(rbos)))


			pwp = 0
			adjusted_length = 0
			s1 = result.original_result.attacked_text.text.split() 
			s2 = result.perturbed_result.attacked_text.text.split()
        
			for i in range(len(s1)):
				#print("Comparing: ", s1[i] , s2[i])
				if s1[i][0].isalpha():  
					if s1[i] != s2[i]:
						pwp += 1
				else:
					#print(s1[i], " is non alphanumeric")
					adjusted_length += 1
			#print(pwp,len(s1),adjusted_length)
			pwp = pwp / (len(s1)-adjusted_length)
			print("Perturbed Word Proportion: ",pwp)
			
			results.append({'example': example, 'result': result, 'exp_before': exp1, 'exp_after': exp2, 'rbo': rboOutput, 'log': None,'perturbed_word_proportion': pwp})


			if not args.debug:
				save(results, filename)

		# except Exception as e:
		# 	print(e)
		# 	if not args.debug:
		# 		results.append({'example': example, 'result': None, 'exp_before': None, 'exp_after': None, 'rbo': None, 'log': e})
		# 		save(results, filename)


