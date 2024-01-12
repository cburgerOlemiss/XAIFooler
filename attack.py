import warnings
warnings.filterwarnings("ignore")

import torch
import math
import numpy
import scipy 

def monkeypath_itemfreq(sampler_indices):
   return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import textattack
import transformers

from utils import *
from timeit import default_timer as timer

import gc
gc.collect()
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

from optimum.onnxruntime import ORTModelForSequenceClassification
from common import *

# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
model = ORTModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-hate",
															export=True,
															provider="CUDAExecutionProvider",
															use_io_binding=True)
tokenizer = transformers.AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate", use_fast=True)

# model= torch.nn.DataParallel(model)
model.to('cuda:1')


model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

dataset = textattack.datasets.HuggingFaceDataset("hate_speech18", split="train")
categories = ['Non-Hate','Hate']

# subset = []
# for i in range(len(dataset)):
# 	subset.append(('this movie is really great, I am glad that we went and watch it', 1))
# datasetx = textattack.datasets.Dataset(subset)

# subset = []
# for i in range(22,30):
#     subset.append((dataset[i][0].get("text"), dataset[i][1]))
datasetx = textattack.datasets.Dataset([])


outputName = "output"
startIndex = 0
csvName = outputName + str(startIndex) + "_log.csv"
folderName = "outputName" + str(startIndex)
seed = 1

print("---main---")

# goal_function = ADV_XAI_GF(model_wrapper,
#                             categories=categories,
#                             featureSelector = 2,
#                             limeSamples = 250,
#                             random_seed = seed,
#                             use_cache = True,
#                             model_batch_size=512
#                             )
# start = timer()
# # explanation,prediction,probability = goal_function.generateExplanation('this movie is really great')
# explainer = goal_function.generateExplanation(dataset[1][0]['text'], return_explainer=True)
# print(timer() - start)

# explanation = explainer.explain_prediction(target_names=categories,
# 											feature_names=explainer.vec_.get_feature_names_out())

# def format_explanation_df(explanation):
# 	df = eli5.format_as_dataframes(explanation)['targets']
# 	idx = df.apply(lambda x: '<BIAS>' not in x['feature'], axis=1)
# 	df = df[idx]
# 	return df

# print(exDF)

# print(remove_bias(exDF))

# print(explainer.vec_)
# explanation = explainer.explain_prediction(target_names=categories,
# 											feature_names=explainer.vec_.get_feature_names_out())
# exDF = eli5.format_as_dataframes(explanation)['targets']
# print(exDF)

# start = timer()
# explanation,prediction,probability = goal_function.generateExplanation(dataset[1][0]['text'])
# print(timer() - start)
# print(prediction)


attack = ADV_XAI_Attack.build(model_wrapper,
                           categories = categories,
                           featureSelector = 3, 
                           limeSamples = 5000,
                           random_seed = seed,
                           success_threshold=0.5,
                           model_batch_size=512,
                           max_candidates=10
                            )

attack_args = textattack.AttackArgs(num_examples=1,
                                    random_seed=seed, 
                                    log_to_csv=csvName, 
                                    checkpoint_interval=250, 
                                    checkpoint_dir="./checkpoints", 
                                    disable_stdout=False,
                                    # query_budget=250 #Minimum number of queries, 15% of current document is chosen if it is larger
                                   )

attacker = textattack.Attacker(attack, datasetx, attack_args)

data = []
start = timer()

from tqdm import tqdm
for i in tqdm(range(0,1)):
	try:
		example = textattack.shared.attacked_text.AttackedText(dataset[i][0].get("text"))
		# print(example)
		# example = textattack.shared.attacked_text.AttackedText('this movie is really great, I am glad that we went and watch it')

		output = attacker.attack.goal_function.get_output(example)
		result = attacker.attack.attack(example, output)

		# print(attacker.attack.goal_function.)
		exp1 = attacker.attack.goal_function.generateExplanation(result.original_result.attacked_text.text)
		exp2 = attacker.attack.goal_function.generateExplanation(result.perturbed_result.attacked_text.text)

		df1 = format_explanation_df(exp1[0])
		df2 = format_explanation_df(exp2[0])

		# print(df1, df1.shape)
		# print(df2, df2.shape)

		# print(result.__str__(color_method="ansi") + "\n")

		# if len(df1) != len(df2):
		# 	break

		targetList = []
		for i in range(len(df2)):
			targetList.append(df2.get('feature')[i])
		baseList = []
		for i in range(len(df1)):
			baseList.append(df1.get('feature')[i])

		rboOutput = attacker.attack.goal_function.RBO(targetList, baseList,p=attacker.attack.goal_function.p_RBO)
		print("+++++++++++++++++++")
		print("RBO Score", rboOutput)
		print("+++++++++++++++++++")

		data.append({'example': example, 'result:': result, 'exp_before': exp1, 'exp_after': exp2, 'rbo': rboOutput, 'log': None})

	except Exception as e:
		print(e)
		data.append({'example': example, 'result:': None, 'exp_before': None, 'exp_after': None, 'rbo': None, 'log': e})

end = timer()

print(end - start)

import pickle
# with open('log_5000.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



# import pickle
# with open('log.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# results = attacker.attack_dataset()

# indexGoalFunction = initialIndexGF(model_wrapper,
# 									categories=categories,
# 									featureSelector = 2,
# 									limeSamples = 250,
# 									random_seed = seed
								   
# 									)

# search_method = GreedyWordSwapWIR_XAI(wir_method="delete", 
# 									initialIndexGF=indexGoalFunction,
# 									reverseIndices=True)

# search_method._get_index_order(datasetx[0])
