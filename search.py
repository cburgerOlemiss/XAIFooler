import textattack
from textattack.goal_functions.classification.classification_goal_function import ClassificationGoalFunction
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results.goal_function_result import (
	GoalFunctionResultStatus,
)

import torch
import math
import numpy as np
import scipy 
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
	transformation_consists_of_word_swaps_and_deletions,
)

from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
	InputColumnModification,
	RepeatModification,
	StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

from textattack.attack_recipes import AttackRecipe


import eli5 
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler, MaskingTextSamplers  

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class GreedyWordSwapWIR_XAI(SearchMethod):
	"""An attack that greedily chooses from a list of possible perturbations in
	order of index, after ranking indices by importance.

	Args:
		wir_method: method for ranking most important words
		model_wrapper: model wrapper used for gradient-based ranking
	"""

	def __init__(self, wir_method="unk", unk_token="[UNK]",initialIndexGF = None, reverseIndices = True):
		self.wir_method = wir_method
		self.unk_token = unk_token
		self.initialIndexGF = initialIndexGF
		self.reverseIndices = reverseIndices

	def _get_index_order(self, initial_text):
		"""Returns word indices of ``initial_text`` in descending order of
		importance."""

		# print('running', "get_indices_to_order")
		len_text, indices_to_order = self.get_indices_to_order(initial_text)

		if self.wir_method == "unk":
			leave_one_texts = [
				initial_text.replace_word_at_index(i, self.unk_token)
				for i in indices_to_order
			]
			leave_one_results, search_over = self.initialIndexGF.get_results(leave_one_texts)
			index_scores = np.array([result.score for result in leave_one_results])

		elif self.wir_method == "weighted-saliency":
			# first, compute word saliency
			leave_one_texts = [
				initial_text.replace_word_at_index(i, self.unk_token)
				for i in indices_to_order
			]
			leave_one_results, search_over = self.initialIndexGF.get_results(leave_one_texts)
			saliency_scores = np.array([result.score for result in leave_one_results])

			softmax_saliency_scores = softmax(
				torch.Tensor(saliency_scores), dim=0
			).numpy()

			# compute the largest change in score we can find by swapping each word
			delta_ps = []
			for idx in indices_to_order:

				# Exit Loop when search_over is True - but we need to make sure delta_ps
				# is the same size as softmax_saliency_scores
				if search_over:
					delta_ps = delta_ps + [0.0] * (
						len(softmax_saliency_scores) - len(delta_ps)
					)
					break

				transformed_text_candidates = self.get_transformations(
					initial_text,
					original_text=initial_text,
					indices_to_modify=[idx],
				)
				if not transformed_text_candidates:
					# no valid synonym substitutions for this word
					delta_ps.append(0.0)
					continue
				swap_results, search_over = self.initialIndexGF.get_results(
					transformed_text_candidates
				)
				score_change = [result.score for result in swap_results]
				if not score_change:
					delta_ps.append(0.0)
					continue
				max_score_change = np.max(score_change)
				delta_ps.append(max_score_change)

			index_scores = softmax_saliency_scores * np.array(delta_ps)

		elif self.wir_method == "delete":
			leave_one_texts = [
				initial_text.delete_word_at_index(i) for i in indices_to_order
			]
			leave_one_results, search_over = self.initialIndexGF.get_results(leave_one_texts)
			index_scores = np.array([result.score for result in leave_one_results])
			# print("index_scores", index_scores)
			# print(leave_one_texts, index_scores)

		elif self.wir_method == "gradient":
			victim_model = self.get_victim_model()
			index_scores = np.zeros(len_text)
			grad_output = victim_model.get_grad(initial_text.tokenizer_input)
			gradient = grad_output["gradient"]
			word2token_mapping = initial_text.align_with_model_tokens(victim_model)
			for i, index in enumerate(indices_to_order):
				matched_tokens = word2token_mapping[index]
				if not matched_tokens:
					index_scores[i] = 0.0
				else:
					agg_grad = np.mean(gradient[matched_tokens], axis=0)
					index_scores[i] = np.linalg.norm(agg_grad, ord=1)

			search_over = False

		elif self.wir_method == "random":
			index_order = indices_to_order
			np.random.shuffle(index_order)
			search_over = False
		else:
			raise ValueError(f"Unsupported WIR method {self.wir_method}")

		if self.wir_method != "random":
			if self.reverseIndices:
				index_order = np.array(indices_to_order)[(index_scores).argsort()]
			else:
				index_order = np.array(indices_to_order)[(-index_scores).argsort()]

		return index_order, search_over


	def perform_search(self, initial_result):

		attacked_text = initial_result.attacked_text
		
		## Need to generate a base explanation here, search function does not have access to the goal function's methods by default
		## Using a custom goal function to order the indices by probability difference as in the textfooler paper
		
		self.initialIndexGF.init_attack_example(attacked_text, initial_result.ground_truth_output)        
		
		# Sort words by order of importance
		index_order, search_over = self._get_index_order(attacked_text)

		i = 0
		cur_result = initial_result
		results = None

		# print('index_order', index_order)
		# print("curr attack", cur_result.attacked_text.words)
		while i < len(index_order) and not search_over:
			self.goal_function.explainer = None
			self.goal_function.explainer_index = i

			to_modify_word = cur_result.attacked_text.words[index_order[i]]

			print("\n==========================================")
			print("MODIFYING", to_modify_word)
			# print("features", self.goal_function.features[0])
			if to_modify_word.lower() in self.goal_function.features[0]:
				print("preventing from modifying top-n features", to_modify_word.lower(), self.goal_function.features[0])
				i += 1
				continue

			# if 'modified_indices' in cur_result.attacked_text.attack_attrs:
			# 	modified_indices = cur_result.attacked_text.attack_attrs['modified_indices']
			# 	words = set([cur_result.attacked_text.words[i] for i in modified_indices])
			# 	if to_modify_word in words or to_modify_word.lower() in words:
			# 		print("preventing from modifying THE SAME WORD", to_modify_word, words)
			# 		i += 1
			# 		continue

			# print("iteration at token index", i)
			transformed_text_candidates = self.get_transformations(
				cur_result.attacked_text,
				original_text=initial_result.attacked_text,
				indices_to_modify=[index_order[i]],
			)
			print("Found", len(transformed_text_candidates), "transformed_text_candidates")
			i += 1
			if len(transformed_text_candidates) == 0:
				continue

			# print('self.get_goal_results', self.get_goal_results)
			# print("checking get_goal_results...")
			results, search_over = self.get_goal_results(transformed_text_candidates) 
									# target_original=cur_result.attacked_text.words[index_order[i]])
			print("search_over", search_over)
			# print(results)


			results = sorted(results, key=lambda x: -x.score)

			# Skip swaps which don't improve the score (v.s. the best score found right now)
			if results[0].score > cur_result.score:
				print("SCORE IMPROVED", results[0].score, cur_result.score)
				# print("-->", results[0])
				cur_result = results[0]
			else:
				continue

			# If we succeeded, return the index with best similarity.
			if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
				best_result = cur_result
				# @TODO: Use vectorwise operations
				max_similarity = -float("inf")
				for result in results:
					if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
						continue
					candidate = result.attacked_text

					# important, not the best RBO is selected, also need to check similarity score
					try:
						similarity_score = candidate.attack_attrs["similarity_score"] 
					except KeyError:
						# If the attack was run without any similarity metrics,
						# candidates won't have a similarity score. In this
						# case, break and return the candidate that changed
						# the original score the most.
						break
					if similarity_score > max_similarity:
						max_similarity = similarity_score
						best_result = result
				return best_result

			# if i > len(index_order):
			# 	break
			
		return cur_result

	def check_transformation_compatibility(self, transformation):
		"""Since it ranks words by their importance, GreedyWordSwapWIR is
		limited to word swap and deletion transformations."""
		return transformation_consists_of_word_swaps_and_deletions(transformation)

	@property
	def is_black_box(self):
		if self.wir_method == "gradient":
			return False
		else:
			return True

	def extra_repr_keys(self):
		return ["wir_method"]