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
from sklearn.feature_extraction.text import CountVectorizer

from common import *

class initialIndexGF(ClassificationGoalFunction):
    """
    Goal Function for ordering indices for certain attack initializations.
    
    Also used for generating individual explanations after attack completion.

    """

    def __init__(self, *args, 
                  categories,
                  lossFunction = 0.5,
                  minWeightDistance = 0, 
                  featureSelector = 1,
                  limeSamples = 5000,
                  random_seed = 1,
                  model_batch_size=128,
                  **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.baseDoc = None
        
        self.categories = categories
        self.limeSamples = limeSamples
        self.random_seed = random_seed
        
        self.baseExplanation = None
        self.baseProbability = None
        self.basePrediction = None
        self.baseRanks = None
        
        self.lossFunction = lossFunction
            
        self.minWeightDistance = minWeightDistance
        self._min_weight_distance()
        
        self.featureSelector = featureSelector
            
        self.features = None
        self.featureCount = None
                    
        self.perturbedExplanation = None
        self.perturbedProbability = None
        self.perturbedPrediction = None
        
        self.targets = None
        
        self.complete = False
        
        self.tempScore = None
        
    
        
    def pred_proba(self, attacked_text):
        """Returns output for display based on the result of calling the
        model."""
        if type(attacked_text) == str:
            attacked_text = textattack.shared.attacked_text.AttackedText(attacked_text)
        
        return np.expand_dims(self._call_model([attacked_text])[0].numpy(),axis=1)
    
    def pred_proba_LIME_Sampler(self, attacked_text):
        """Returns output for the LIME sampler based on the result of calling the model.
            Expects default LIME sampler output as a tuple of str.
            Use pred_proba for individual str or attacked_text
        """
        # if type(attacked_text) == str:
        #   attacked_text = textattack.shared.attacked_text.AttackedText(attacked_text)
        
        output = torch.stack(self._call_model_LIME_Sampler(attacked_text),0)
        return output.numpy()
    
 
    def generateBaseExplanation(self,document):
        explainer, explanation, prediction, probability = generate_explanation_single(self, document, 
                                                                custom_n_samples=None, 
                                                                debug=False, 
                                                                return_explainer=True)
        self.baseExplanationDataframe = format_explanation_df(explanation, target=prediction)

        return explanation, prediction, probability
    
    def generateExplanation(self,document):
        explainer, explanation, prediction, probability = generate_explanation_single(self, document, 
                                                                custom_n_samples=None, 
                                                                debug=False, 
                                                                return_explainer=True)
        return explanation, prediction, probability
    
    def selectTopFeatures(self,top_n=1):
        """
        Returns features of base explanation to be held constant.
        top_n returns top n features, default is the top feature
        """       
        targets = self.baseExplanationDataframe
        if top_n > 0:
            features = []
            weights = []
            ranks = []
            result = []
            for i in range(min(top_n, len(targets))):
                features.append(targets.iloc[i][1])
                weights.append(targets.iloc[i][2])
                ranks.append(i)
                
            result.append(features)
            result.append(weights)
            result.append(ranks)
            
            
            return result
        else:
            raise ValueError()
        
    def _countFeatures(self,document):
                
        featureCount = []
        
        for i in range(len(self.features[0])):
            featureCount.append(document.text.count(self.features[0][i]))
         
        return featureCount
    
    def _compareFeatureCounts(self,attacked_text):
        featureCountsTransformed = []
        for i in range(len(self.features[0])):
            featureCountsTransformed.append(attacked_text.text.count(self.features[0][i]))

        if len(self.featureCount) != len(featureCountsTransformed):
            return False
        
        for i,j in zip(self.featureCount,featureCountsTransformed):
            if i != j:
                return False
        return True
            
    def _min_weight_distance(self):
        if self.minWeightDistance == None:
            self.minWeightDistance == 0
        elif self.minWeightDistance < 0:
            raise ValueError("minWeight distance must be > 0")
            
    def init_attack_example(self, attacked_text, ground_truth_output):
        
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        
        self.initial_attacked_text = attacked_text

        self.ground_truth_output = ground_truth_output

        self.num_queries = 0
        
        self.complete = False
        
        self.baseExplanation = self.generateBaseExplanation(attacked_text)
        self.baseProbability = self.baseExplanation[2]
        self.basePrediction = self.baseExplanation[1]
        
        if self.featureSelector is not list:
            
            self.features = self.selectTopFeatures(top_n=self.featureSelector)
        else:
            raise NotImplementedError()
            
        self.featureCount = self._countFeatures(attacked_text)
        
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _
    
    def _is_goal_complete(self, model_output, _):
        
        if (model_output.numel() == 1) and isinstance(
            self.ground_truth_output, float
        ):
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            return model_output.argmax() != self.ground_truth_output
    


    def get_result(self, attacked_text, **kwargs):
        """A helper method that queries ``self.get_results`` with a single
        ``AttackedText`` object."""
        results, search_over = self.get_results([attacked_text], **kwargs)
        result = results[0] if len(results) else None
        return result, search_over


    def get_results(self, attacked_text_list, check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]

        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)

        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )        
        return results, self.num_queries == self.query_budget


    def _get_goal_status(self, model_output, attacked_text, check_skip=False):
        should_skip = check_skip and self._should_skip(model_output, attacked_text)
        if should_skip:
            return GoalFunctionResultStatus.SKIPPED
        if self.maximizable:
            return GoalFunctionResultStatus.MAXIMIZING
        if self._is_goal_complete(model_output, attacked_text):
            return GoalFunctionResultStatus.SUCCEEDED
        return GoalFunctionResultStatus.SEARCHING

    def _should_skip(self, model_output, attacked_text):
        return self._is_goal_complete(model_output, attacked_text)

    def _get_score(self, model_output, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            return 1 - model_output[self.ground_truth_output]
       
    def get_explanations(self, originalText, attackedText):
        """Returns selected features, base and perturbed explanations after attack completion
            Requires the original text and the perturbed text.
            
            Used only for generating individuals explanation to save as csv files post attack.
        """
        baseExplanation =  self.generateBaseExplanation(originalText)
        
        if originalText == attackedText:
            perturbedExplanation = baseExplanation
        else:
            perturbedExplanation = self.generateExplanation(attackedText)
        perturbedExplanation = self.generateExplanation(attackedText)
        
        self.baseExplanation = baseExplanation
        
        features = self.selectTopFeatures(top_n=self.featureSelector)
        
        bTargets = format_explanation_df(baseExplanation[0])
        # bTargets = bDF.get('targets')
        
        pTargets = format_explanation_df(perturbedExplanation[0])
        # pTargets = pDF.get('targets')
        
        return(features,bTargets,pTargets)
    
    def _call_model_LIME_Sampler(self, attacked_text_list):
        """Gets predictions for a list of ``AttackedText`` objects.

        Gets prediction from cache if possible. If prediction is not in
        the cache, queries model and stores prediction in cache.
        """
        #print(attacked_text_list,type(attacked_text_list))
        if type(attacked_text_list) is tuple:
            attacked_text_list = [textattack.shared.attacked_text.AttackedText(string) for string in attacked_text_list]
        else:
            attacked_text_list = [textattack.shared.attacked_text.AttackedText(string) for string in attacked_text_list[0]]

        local_cache = set()

        if not self.use_cache:
            return self._call_model_uncached(attacked_text_list)

        else:
            # print("using cache...")
            # print("with size", self._call_model_cache.get_size())
            uncached_list = []
            for text in attacked_text_list:
                if text in self._call_model_cache:
                    # Re-write value in cache. This moves the key to the top of the
                    # LRU cache and prevents the unlikely event that the text
                    # is overwritten when we store the inputs from `uncached_list`.
                    self._call_model_cache[text] = self._call_model_cache[text]
                else:
                    if text.text not in local_cache:
                        uncached_list.append(text)
                        local_cache.add(text.text)

                # else:
                #   # print("not available yet [{}]".format(text.text))
                #   # print(text in self._call_model_cache)
                #   # print(self._call_model_cache)
                #   uncached_list.append(text)


            # uncached_list = [
            #   text
            #   for text in attacked_text_list
            #   if text not in self._call_model_cache
            # ]

            # print("[B] calling models on ___ texts", len(uncached_list))
            outputs = self._call_model_uncached(uncached_list)
            for text, output in zip(uncached_list, outputs):
                self._call_model_cache[text] = output
            all_outputs = [self._call_model_cache[text] for text in attacked_text_list]
            return all_outputs
    
    
    def extra_repr_keys(self):
        return []