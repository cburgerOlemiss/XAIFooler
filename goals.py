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
from timeit import default_timer as timer
from sklearn.feature_extraction.text import CountVectorizer

from common import *


class ADV_XAI_GF(ClassificationGoalFunction):
    """
    Goal Function for XAI attack.
    
    Steps: 
        Set Maximum Prob Difference (Optional and Currently Unused)

        Choose similarity measure:
            RBO: Top weighted with the mass controlled by a convergent series with parameter p

            Center of Mass (COM): Determines the center of mass for the feature weights of an explanation with respect to the order of the words in the document being attacked.
                Standard COM: Difference between the index of the center of the base and perturbed explanations.
                COM_proportional: Difference between the total mass accumulated up to the center index (>=50% of the total mass) for the base and perturbed explanations.
                COM_rank_weighted: Difference between the index of the center of the base and perturbed explnations where the feature mass is weighted by the order of the mass in the explanation.

            L2: L2 norm of the base and perturbed explanations w.r.t the original document.

        Set parameters:

            For RBO, p in (0,1], the top weightedness of the explanation. Where p -> increases the top mass, p -> 1 distributes mass more equally among lower ranked features.

        Set Sampling Rate
        Set initial similarity threshold: similarity difference needed to accept the first perturbation
        Set successful similarity threshold: similarity difference needed to declare the attack successful, that is, stop the attack early.
        Set adjusted query budget based on 15% of the document length
        Generate base explanation for a document
        Select feature(s) to hold constant
        Generate perturbations in batches of size defined in the attack recipe
        Repeat until success or query exhaustion
                            

    """

    def __init__(self, *args, 
                 categories,
                 lossFunction = 0.5,
                 minWeightDistance = 0, 
                 featureSelector = 1,
                 limeSamples = 5000,
                 random_seed = 1,
                 initial_acceptance_threshold = 0.95,
                 success_threshold = 0.50,
                 p_RBO = 0.80,
                 use_cache = True,
                 model_batch_size=128,
                 logger=None,
                 similarity_measure = 'rbo',
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        #Original document before perturbations
        self.baseDoc = None

        #Initialization parameters
        self.categories = categories
        self.limeSamples = limeSamples
        self.random_seed = random_seed
        self.initial_acceptance_threshold = initial_acceptance_threshold
        self.success_threshold = success_threshold
        self.p_RBO = p_RBO



        self.similarity_measure = similarity_measure
        self.RBO_flag = False
        self.COM_flag = False
        self.COM_proportional_flag = False
        self.COM_rank_weighted_flag = False
        self.l2_flag = False
        self.jaccard_flag = False
        self.jaccard_weighted_flag = False
        self.kendall_flag = False
        self.kendall_weighted_flag = False
        self.spearman_flag = False
        self.spearman_weighted_flag = False

        if self.similarity_measure == 'rbo':
            self.RBO_flag = True
        elif self.similarity_measure == 'com':          
            self.COM_flag = True
        elif self.similarity_measure == 'com_proportional':          
            self.COM_proportional_flag = True
        elif self.similarity_measure == 'com_rank_weighted':          
            self.COM_rank_weighted_flag = True
        elif self.similarity_measure == 'l2':
            self.l2_flag = True
        elif self.similarity_measure == 'jaccard':
            self.jaccard_flag = True
        elif self.similarity_measure == 'jaccard_weighted':
            self.jaccard_weighted_flag = True
        elif self.similarity_measure == 'kendall':
            self.kendall_flag = True
        elif self.similarity_measure == 'kendall_weighted':
            self.kendall_weighted_flag = True
        elif self.similarity_measure == 'spearman':
            self.spearman_flag = True
        elif self.similarity_measure == 'spearman_weighted':
            self.spearman_weighted_flag = True
        
        #Values for the base explanation
        self.baseExplanation = None
        self.baseExplanationDataframe = None
        self.baseProbability = None
        self.basePrediction = None
        self.baseRanks = None
        self.baseCOM = None
        self.baseTotalMass = 0

        self.baseOrderedExplanation = None
        self.baseAbsWeightSum = None

        self.baseOrderedWeights = None

        self.initial_attacked_text_length = None
        
        #Unused, only one loss function implemented here
        self.lossFunction = lossFunction
        
        #Probability difference variables, currently unused
        self.minWeightDistance = minWeightDistance
        self._min_weight_distance()
        
        #Select top n features and get their number of occurences
        #Store them in the variables below
        self.featureSelector = featureSelector
         
        self.features = None
        self.featureCount = None

        self.explainer = None
        #    
        
        #Values for the perturbed explanation
        self.perturbedExplanation = None
        self.perturbedProbability = None
        self.perturbedPrediction = None
        
        self.targets = None
        
        #Flag for sucessful attack completion
        self.complete = False
        
        #Placeholder for current score
        self.tempScore = None
        #Current best score
        self.bestScore = None
        
        #Current document's number of perturbations
        self.numPerturbations = None
        #Max number of allowed perturbations
        #Set in initialize function
        self.maxPerturbations = None
        
        self.GF_query_budget = None


        self.use_cache = use_cache
        self.batch_size = model_batch_size
        self.logger = logger

        #Flag for multiclass
        self.multiclass = False
        if len(self.categories) > 2:
            self.multiclass = True
        # self.baseExplanationIndexOffset = None
     
    def pred_proba(self, attacked_text):
        """Returns output for display based on the result of calling the
        model."""
        if type(attacked_text) == str:
            attacked_text = textattack.shared.attacked_text.AttackedText(attacked_text)
        
        return np.expand_dims(self._call_model([attacked_text])[0].numpy(),axis=1)
    
    def pred_proba_LIME_Sampler(self, attacked_texts):
        """Returns output for the LIME sampler based on the result of calling the model.
           Expects default LIME sampler output as a tuple of str.
           Use pred_proba for individual str or attacked_text
        """
        # print('pred_proba_LIME_Sampler', attacked_texts)
        # if type(attacked_text) == str:
        #   attacked_text = textattack.shared.attacked_text.AttackedText(attacked_text)
        
        output = torch.stack(self._call_model_LIME_Sampler(attacked_texts),0)
        return output.numpy()
    
    
    def generateBaseExplanation(self, document, custom_n_samples=None):
        explainer, explanation, prediction, probability = generate_explanation_single(self, document, 
                                                                custom_n_samples=custom_n_samples, 
                                                                debug=True, 
                                                                return_explainer=True)

        self.baseExplanationDataframe = format_explanation_df(explanation, target=prediction)
        print(self.baseExplanationDataframe)
        self.base_feature_set = set(self.baseExplanationDataframe.get('feature'))
        self.base_explainer = explainer

        # if self.multiclass:
        #     self.baseExplanationDataFrame = self.baseExplanationDataframe[self.baseExplanationDataframe['target'] == prediction]

        # self.baseExplanationIndexOffset = self.baseExplanationDataframe[self.baseExplanationDataframe['target'] == prediction].index[0]

        return explanation,prediction,probability
    
    def generateExplanation(self, document, return_explainer=False, custom_n_samples=None):
        explainer, explanation, prediction, probability = generate_explanation_single(self, document, 
                                                                custom_n_samples=custom_n_samples, 
                                                                debug=False, 
                                                                return_explainer=True)
        if return_explainer:
            return explainer

        return explanation, prediction, probability
    

    def selectTopFeatures(self, top_n=1):
        """
        Returns features of base explanation to be held constant.
        top_n returns top n features, default is the top feature
        """
        
        targets = self.baseExplanationDataframe
        #print(targets)
        
        targets = format_explanation_df(self.baseExplanation[0], self.basePrediction)
        targets['weight'] = targets['weight'].abs()
        #print(exdf)
        targets = targets.sort_values(by='weight',ascending=False,ignore_index=True)
        
        if top_n > 0:
            features = []
            weights = []
            ranks = []
            result = []
            # targets = exDF.get('targets')
            # print(targets)
            # print(top_n)
            for i in range(min(top_n, len(targets))):
                features.append(targets.iloc[i][1])
                weights.append(targets.iloc[i][2])
                ranks.append(i)
                
            result.append(features)
            result.append(weights)
            result.append(ranks)
            
            print("top features: ",features)
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

        # if len(self.featureCount) != len(featureCountsTransformed):
        #     return False
        
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
        
        # print("INITIAL ATTACK EXAMPLE TO RESET")

        self.initial_attacked_text = attacked_text
        print(self.initial_attacked_text)
        self.initial_attacked_text_length = len(attacked_text.words)
        print(self.initial_attacked_text_length)
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        self.numPerturbations = 0
        # self.baseExplanationIndexOffset = None
        
        self.complete = False
        self.tempScore = None
        self.bestScore = None
        
        self.baseExplanation = self.generateBaseExplanation(attacked_text)
        self.baseProbability = self.baseExplanation[2]
        self.basePrediction = self.baseExplanation[1]

        '''
        print("Base Explanation:")
        print(self.baseExplanation)
        print("Explanation df")
        exdf = format_explanation_df(self.baseExplanation[0], self.basePrediction)
        print(exdf)
        print("features")
        f =  exdf.get('feature').values
        print(f)
        print("weights")
        w = exdf.get('weight').values
        print(w)
        '''
        if len(self.baseExplanationDataframe) == 0:
            return None, None

        # self.explainer_set = defaultdict()
        self.explainer_index = -1
        
        #Calculate document length adjusted query budget
        #self.query_budget = int((len(attacked_text.text)/6)/2 * 15/4)
        #qb = int((len(attacked_text.text)/6)/2 * 15/4)

        #Update query budget if document length adjusted value is larger
        #Otherwise accept the default
        # if qb > self.query_budget:
        #     self.query_budget = qb
        
        #Calculate maximum allowed perturbations
        self.maxPerturbations = int(len(attacked_text.text)/6*0.15)
        
        #At least 1 perturbation is always allowed
        if self.maxPerturbations == 0:
            self.maxPerturbations = 1
        
        #Select top n features
        if self.featureSelector is not list:
            self.features = self.selectTopFeatures(top_n=self.featureSelector)
        else:
            raise NotImplementedError()
        
        #Count the number of instance of the top n features
        self.featureCount = self._countFeatures(attacked_text)
        
        #Set base Center of mass
        if self.COM_flag:
            self.baseCOM = self.COM(attacked_text,format_explanation_df(self.baseExplanation[0], self.basePrediction))

        if self.COM_proportional_flag: 
            self.baseCOM = self.COM_proportional(attacked_text,format_explanation_df(self.baseExplanation[0], self.basePrediction))

        if self.COM_rank_weighted_flag:
            self.baseCOM = self.COM_rank_weight(attacked_text,format_explanation_df(self.baseExplanation[0], self.basePrediction))

        #Calculate ordered weights for the unperturbed document here and store to prevent recalculation in is_goal_compelte
        if self.l2_flag:
            attacked_text_list = attacked_text.words
            ordered_weights = [0] * len(attacked_text_list)
            features = format_explanation_df(self.baseExplanation[0], self.basePrediction).get('feature').values.tolist()
            weights = format_explanation_df(self.baseExplanation[0], self.basePrediction).get('weight').values.tolist()

            feat2weight = dict(zip(features, weights))
            for j in range(len(attacked_text_list)):
                ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)

            '''
            for i,(f,w) in enumerate(zip(features,weights)):
                try:
                    #print("Placing feature ",f, " with weight ", w, " at index ", attacked_text_list.index(f) , " (attacked_text[i]) = ", attacked_text_list[i])
                    ordered_weights[attacked_text_list.index(f)] = w
                except:
                    #This should not occur as features are identically cased and taken from the document.
                    print("Feature ", f, " not in document.")

            self.baseOrderedWeights = ordered_weights
            for i in ordered_weights:
                self.baseTotalMass += abs(i) 
            '''

            self.baseOrderedWeights = np.array(ordered_weights)
            self.baseTotalMass = np.sum(np.abs(ordered_weights))

        exdf = format_explanation_df(self.baseExplanation[0], self.basePrediction)
        exdf['weight'] = exdf['weight'].abs()
        #print(exdf)
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)
        
        self.baseAbsWeightSum = sum(exdf['weight'])
        self.baseOrderedExplanation = exdf
                
        #print(self.baseOrderedExplanation)
        #print(self.baseAbsWeightSum)

        result, _ = self.get_result(attacked_text, check_skip=True)
        # print(result)
        # print("++++++++++++++++++=")
        return result, _
    
    
    # compute RBO
    def RBO(self,list1,list2,p):
        
         comparisonLength = min(len(list1),len(list2))
        
         set1 = set()
         set2 = set()
        
         summation = 0
         for i in range(comparisonLength):
             set1.add(list1[i])
             set2.add(list2[i])            
            
             summation += math.pow(p,i+1) * (len(set1&set2) / (i+1))
        
         return ((len(set(list1)&set(list2))/comparisonLength) * math.pow(p,comparisonLength)) + (((1-p)/p) * summation)
    
    # compute Center of Mass
    def COM(self,attacked_text,explanation_dataframe):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        features = explanation_dataframe.get('feature').values.tolist()
        weights = explanation_dataframe.get('weight').values.tolist()
        #occurences = explanation_dataframe.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)
        '''
        for i,(f,w) in enumerate(zip(features,weights)):
            try:
                #print("Placing feature ",f, " with weight ", w, " at index ", attacked_text_list.index(f) , " (attacked_text[i]) = ", attacked_text_list[i])
                if occurences[i] == 1:
                    ordered_weights[attacked_text_list.index(f)] = w
                else:
                    for j in range(len(attacked_text_list)):
                        if attacked_text_list[j] == f:
                            ordered_weights[j] = w
            except:
                #This should not occur as features are identically cased and taken from the document.
                print("Feature ", f, " not in document.")
        '''  
        #print(attacked_text_list)        
        #print(ordered_weights)
        
        total_mass = np.sum(np.abs(ordered_weights))

        '''
        total_mass = 0
        for i in ordered_weights:
            total_mass += abs(i)
        #print("Total mass = ", total_mass)
        '''   
        center_mass = total_mass / 2
        #print("Center mass = ", center_mass)
              
        current_mass = 0
        for i in range(len(ordered_weights)):
            current_mass += abs(ordered_weights[i])
            #print("Current mass = ", current_mass)
            try:
                if current_mass >= center_mass and ordered_weights[i+1] != 0:
                    return i
            except:
                return i


        # compute Center of Mass weighted by feature index
    def COM_rank_weight(self,attacked_text,explanation_dataframe):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        features = explanation_dataframe.get('feature').values.tolist()
        weights = explanation_dataframe.get('weight').values.tolist()
        #occurences = explanation_dataframe.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)*(j+1)
        '''
        for i,(f,w) in enumerate(zip(features,weights)):
            try:
                #print("Placing feature ",f, " with weight ", w, " at index ", attacked_text_list.index(f) , " (attacked_text[i]) = ", attacked_text_list[i])
                if occurences[i] == 1:
                    ordered_weights[attacked_text_list.index(f)] = w*(i+1)
                else:
                    for j in range(len(attacked_text_list)):
                        if attacked_text_list[j] == f:
                            ordered_weights[j] = w*(i+1)
            except:
                #This should not occur as features are identically cased and taken from the document.
                print("Feature ", f, " not in document.")
        '''    
        #print(attacked_text_list)        
        #print(ordered_weights)
        
        total_mass = np.sum(np.abs(ordered_weights))

        '''
        total_mass = 0
        for i in ordered_weights:
            total_mass += abs(i)
        #print("Total mass = ", total_mass)
        '''   
            
        center_mass = total_mass / 2
        #print("Center mass = ", center_mass)
              
        current_mass = 0
        for i in range(len(ordered_weights)):
            current_mass += abs(ordered_weights[i])
            #print("Current mass = ", current_mass)
            try:
                if current_mass >= center_mass and ordered_weights[i+1] != 0:
                    return i
            except:
                return i       
    
       # compute proportion of mass associated with the center index
    def COM_proportional(self,attacked_text,explanation_dataframe):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        features = explanation_dataframe.get('feature').values.tolist()
        weights = explanation_dataframe.get('weight').values.tolist()
        #occurences = explanation_dataframe.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)

        '''
        for i,(f,w) in enumerate(zip(features,weights)):
            try:
                #print("Placing feature ",f, " with weight ", w, " at index ", attacked_text_list.index(f) , " (attacked_text[i]) = ", attacked_text_list[i])
                if occurences[i] == 1:
                    ordered_weights[attacked_text_list.index(f)] = w
                else:
                    for j in range(len(attacked_text_list)):
                        if attacked_text_list[j] == f:
                            ordered_weights[j] = w
            except:
                #This should not occur as features are identically cased and taken from the document.
                print("Feature ", f, " not in document.")
        '''     
        #print(attacked_text_list)        
        #print(ordered_weights)
        
        total_mass = np.sum(np.abs(ordered_weights))

        '''
        total_mass = 0
        for i in ordered_weights:
            total_mass += abs(i)
        #print("Total mass = ", total_mass)
        '''   
        center_mass = total_mass / 2
        #print("Center mass = ", center_mass)
              
        current_mass = 0
        for i in range(len(ordered_weights)):
            current_mass += abs(ordered_weights[i])
            if current_mass >= center_mass and ordered_weights[i+1] != 0:
                    break
    
        return current_mass / total_mass
    
    def l2(self,attacked_text,explanation_dataframe):
        t0 = timer()
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        features = explanation_dataframe.get('feature').values.tolist()
        weights = explanation_dataframe.get('weight').values.tolist()
        # occurences = explanation_dataframe.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)

        l2 = np.linalg.norm(self.baseOrderedWeights - np.array(ordered_weights))
        t1 = timer()
        print("l2 took...", t1 - t0)

        return l2
    
    def jaccard(self,list1,list2):
        set1 = set(list1)
        set2 = set(list2)
        
        return (len(set.intersection(set1,set2)) / len(set.union(set1,set2)) )
    
    def jaccard_weighted(self,attacked_text,explanation_dataframe):
        #Need ordered weights and features from the base explanation
        
        #for each feature in base explanation, if it does not exist in the perturbed explanation reduce 
        # sim by its (normalized) weight in the base explanation
        
        sim = 1
        
        features = explanation_dataframe.get('feature').values.tolist()
        
        for i in range(len(self.baseOrderedExplanation)):
            if self.baseOrderedExplanation['feature'][i] not in features:
                       sim -= self.baseOrderedExplanation['weight'][i]/self.baseAbsWeightSum
        
        return sim
            
    def kendall(self,attacked_text,explanation_dataframe):
           
        explanation_dataframe['weight'] = explanation_dataframe['weight'].abs()
        #print(explanation_dataframe)
        explanation_dataframe = explanation_dataframe.sort_values(by='weight',ascending=False)
        
        features = explanation_dataframe.get('feature').values.tolist()
        #max_dissonance = max(len(self.baseOrderedExplanation),len(features)):
        current_dissonance = 0
       
        #print(features)
        #print(self.baseOrderedExplanation['feature'])
        
        l1 = len(self.baseOrderedExplanation['feature'])
        l2 = len(features)
        
        #print(l1,l2)
                       
        if l1 > l2:
            max_dissonance = l1
            diff = l1-l2
            shorter_explanation = l2
        else:
            max_dissonance = l2
            diff = l2-l1
            shorter_explanation = l1
                       
        
        for i in range(shorter_explanation):
            #print("Comparing ",self.baseOrderedExplanation['feature'][i],features[i])
            if self.baseOrderedExplanation['feature'][i] != features[i]:
                       current_dissonance += 1
                       #print("Features Different, Current Dissonance = ",current_dissonance)
                        
        current_dissonance += diff
        #print('diff = ', diff)
        
        return 1 - (current_dissonance / max_dissonance)
    
    def kendall_weighted(self,attacked_text,explanation_dataframe):
        
        explanation_dataframe['weight'] = explanation_dataframe['weight'].abs()
        #print(explanation_dataframe)
        explanation_dataframe = explanation_dataframe.sort_values(by='weight',ascending=False)
        
        features = explanation_dataframe.get('feature').values.tolist()
        #max_dissonance = max(len(self.baseOrderedExplanation),len(features)):
        current_dissonance = 0
        dissonant_weights = 0
        
        #print(features,self.baseOrderedExplanation['feature'])
        
        l1 = len(self.baseOrderedExplanation)
        l2 = len(features)
        
        #print(l1,l2)
                       
        if l1 > l2:
            max_dissonance = l1
            diff = l1-l2
            #shorter_explanation = l2
        else:
            max_dissonance = l2
            diff = l2-l1
            #shorter_explanation = l1
                       
        for i in range(l1):
            #print("Comparing ",self.baseOrderedExplanation['feature'][i],features[i])
            if self.baseOrderedExplanation['feature'][i] != features[i]:
                       current_dissonance += 1
                       dissonant_weights += self.baseOrderedExplanation['weight'][i]/self.baseAbsWeightSum
                       #print("Features Different, Current Dissonance = ",current_dissonance)
                       #print("Missing Weight = ",self.baseOrderedExplanation['weight'][i]/self.baseAbsWeightSum)
        current_dissonance += diff
        #print('diff = ', diff)
        return (1 - (current_dissonance / max_dissonance)) * (1-dissonant_weights)
        
        
    
    def spearman(self,attacked_text,explanation_dataframe):
        
        #l1 distance between features
        
        #for each feature in base explanation, calculate the distance if it remains the perturbed explanation,
        #otherwise penalize with some value (half of the length of the explanation (uniform mean of possible distance))
        
        #maximum total distance is 1/2 floor(explanation_Size squared)
        
        #sum distances divide by max distance
        #print(self.baseOrderedExplanation['feature'])
        #print(explanation_dataframe)
        
        explanation_dataframe['weight'] = explanation_dataframe['weight'].abs()
        #print(explanation_dataframe)
        explanation_dataframe = explanation_dataframe.sort_values(by='weight',ascending=False)
        
        features = explanation_dataframe.get('feature').values.tolist()
        #print(self.baseOrderedExplanation['feature'])
        #print(features)
        
        current_distance = 0
        
        l1 = len(self.baseOrderedExplanation)
        #l2 = len(self.features)
        
        max_distance = int((l1*l1) / 2)
        penalty = int(l1 / 2)
                       
        #if l1 > l2:
            #max_distance = int((l1*l1) / 2)
            #diff = l1-l2
            #shorter_explanation = l2
        #else:
            #max_distance = int((l2*l2) / 2)
            #diff = l2-l1
            #shorter_explanation = l1
                       
        for i in range(l1):
            if self.baseOrderedExplanation['feature'][i] in features:
                       #print("Base Feature: ",self.baseOrderedExplanation['feature'][i]," loc = ", i, " Perturbed Feature: ", features[features.index(self.baseOrderedExplanation['feature'][i])], " loc = ",features.index(self.baseOrderedExplanation['feature'][i]))
                       current_distance += abs(i - features.index(self.baseOrderedExplanation['feature'][i]))
            else:
                       current_distance += penalty
                       #print(self.baseOrderedExplanation['feature'][i], " not in ", features)
                    
        #print(current_distance)
        
        return 1 - (current_distance / max_distance)               
            
    def spearman_weighted(self,attacked_text,explanation_dataframe):
        
        #for each feature in base explanation, calculate the distance if it remains the perturbed explanation,
        #otherwise penalize with normalized weight * maximum distance
                
        explanation_dataframe['weight'] = explanation_dataframe['weight'].abs()
        #print(explanation_dataframe)
        explanation_dataframe = explanation_dataframe.sort_values(by='weight',ascending=False)
        
        features = explanation_dataframe.get('feature').values.tolist()
        #print(self.baseOrderedExplanation['feature'])
        #print(features)
        
        
        
        l1 = len(self.baseOrderedExplanation)
        #l2 = len(self.features)
        
        max_distance = int((l1*l1) / 2)
        current_distance = max_distance
        penalty = int(l1 / 2)
                       
        #if l1 > l2:
            #max_distance = int((l1*l1) / 2)
            #diff = l1-l2
            #shorter_explanation = l2
        #else:
            #max_distance = int((l2*l2) / 2)
            #diff = l2-l1
            #shorter_explanation = l1
        
        missing_features = 0
        missing_feature_weight = 0
                       
        for i in range(l1):
            if self.baseOrderedExplanation['feature'][i] in features:
                       #print("Base Feature: ",self.baseOrderedExplanation['feature'][i]," loc = ", i, " Perturbed Feature: ", features[features.index(self.baseOrderedExplanation['feature'][i])], " loc = ",features.index(self.baseOrderedExplanation['feature'][i]))
                       current_distance -= abs(i - features.index(self.baseOrderedExplanation['feature'][i]))
            else:
                       missing_features += 1
                       missing_feature_weight += self.baseOrderedExplanation['weight'][i]/self.baseAbsWeightSum
                       #print(self.baseOrderedExplanation['feature'][i], " not in ", features)
                       #print("Weight of feature is: ",self.baseOrderedExplanation['weight'][i]/self.baseAbsWeightSum)
                        
                       #current_distance -= max(max_distance * self.baseOrderedExplanation['weight'][i]/self.baseAbsWeightSum,penalty)
                       current_distance -= max_distance * self.baseOrderedExplanation['weight'][i]/self.baseAbsWeightSum
        
        
        #print("Current distance: ",current_distance, " Total missing feature weight: ", missing_feature_weight)
        #return (1-(current_distance / max_distance)) * (1-missing_feature_weight)
        return (current_distance / max_distance)

    def _is_goal_complete(self, model_output, attacked_text):
          
        #Empty tempScore
        self.tempScore = None
        self.tempExplanation = None
        
        # Max number of perturbations reached
        # if self.numPerturbations >= self.maxPerturbations:
        #   print("Max number of perturbations reached")
        #   return False
        
        # Check that perturbed text retains all of the top n features
        # for i in self.features[0] #top features from base explanation:
        #   if i not in attacked_text.text:
        #       print("FAILED! Check that perturbed text retains all of the top n features", i)
        #       return False


        #Certain samples can occasionally return instances of only a single class throwing a value error.
        try:
            perturbedExplanation = self.generateExplanation(attacked_text)
            # print("generating explanation for ", attacked_text)
            # print("explanation", perturbedExplanation)
        except Exception as e:
            print("!!!!!!!!Exception!!!!!!!", e)
            return False

        #Base prediction class must be the same as the attacked prediction class
        if self.basePrediction != perturbedExplanation[1]:
            if self.logger:
                print("FAILED! Base prediction class must be the same as the attacked prediction class",)
            return False

        targets = format_explanation_df(perturbedExplanation[0], self.basePrediction)
        # uniques = np.unique(targets_full.get('target').values)
        # print(uniques)
        # if not self.multiclass and len(uniques) == 1 and uniques[0] != self.basePrediction:
        if len(targets) == 0:
            if self.logger:
                print("FAILED! Explanation prediction does not cover base prediction")
            return False

        # targets2 = format_explanation_df(perturbedExplanation[0])

        # print("generating target with prediction", self.basePrediction, perturbedExplanation[1])
        # print("targets2 without filtering:")
        # print(targets2)
        # self.tempExplanation = exDF
        # targets = exDF.get('targets')
      
        #Select only features within base prediction class for comparison
        # if self.multiclass:
        #     targets = targets[targets['target'] == self.basePrediction]

        # check that non of the replacement is within the top-n, 
        # except that the replacement is already in the text
        # targets.get('feature')[:self.top_n]
        if 'newly_modified_indices' in attacked_text.attack_attrs:
            # print(attacked_text.attack_attrs)
            new_modified_index = list(attacked_text.attack_attrs['newly_modified_indices'])[0]
            from_w = attacked_text.attack_attrs['previous_attacked_text'].words[new_modified_index]
            to_w = attacked_text.words[new_modified_index]
            if self.logger:
                print("modified {} -> {}".format(from_w, to_w))
            # print("targets")
            # print(perturbedExplanation[0].targets)
            # print("vectorizer features", self.explainer.vec_.get_feature_names_out())
            # print("self.base_feature_set", self.base_feature_set)
            # print("targets.get('feature')[:self.featureSelector]", targets.get('feature')[:self.featureSelector])
            modified_index = list(attacked_text.attack_attrs['modified_indices'])
            for j in modified_index:
                to_w_j = attacked_text.words[j]
                if to_w_j.lower() not in self.base_feature_set and to_w_j.lower() in targets.get('feature')[:self.featureSelector].values:
                    if self.logger:
                        print(f"FAILED! the replacement ``{from_w}``->``{to_w}`` force ``{to_w_j}`` appears in top_n {self.featureSelector} but was not in the orgiginal text")
                    # print(targets)
                    return False

        # print("newly_modified_indices not found in ", attacked_text.attack_attrs)
            
        # print("attacked_text", attacked_text)


        #The direct ranking is not important for the other similarity measures as they based on the feature mass. Only RBO needs to calculate the below.
        if self.RBO_flag:
            decreaseFlag = False
            for i in range(len(self.features[0])): # check in the top features
                #Check if local explanation is missing feature
                # if targets[targets['feature'] == self.features[0][i]].empty: 
                #   print("FAILED! Check if local explanation is missing feature", self.features)
                #   return False
        
                #Check feature rank to ensure decrease
                #if self.features[2][i] >= targets.index[targets.feature == self.features[0][i]][0]:
                #    print("Feature rank not smaller than original")
                #    return False
                
                #At least one of the selected features was reduced in rank
                #error here needs to debug #TODO
                # print("top_features", self.features)
                # print("targets", targets)
                # print("i", i)
                # print("=={}==".format(self.features[0][i]))
                if not (self.features[2][i] >= targets.index[targets.feature == self.features[0][i]][0]):
                    decreaseFlag = True
                    # break
                    
                #if (targets.index[targets.feature == self.features[0][i]][0] > self.features[2][i] ):
                    #print("Increased the rank of a selected feature")
                    #return False
                    
            # No feature is has a lower rank
            if decreaseFlag == False: #
                if self.logger:
                    print("FAILED! No feature is has a lower rank")
                    # print(self.baseExplanationDataframe)
                    # print(targets)
                return False
        

        #Base prediction probability is within some neighborhood of the attacked prediction probability
        # if abs(self.baseProbability - perturbedExplanation[2]) > self.lossFunction:
        #   return False
        
        #RBO calculation 
        # print(self.baseExplanationDataframe)
        # print(targets)
        # print(self.basePrediction)

        # targetList = []
        # # offset = targets[targets['target'] == self.basePrediction].index[0]
        # for i in range(len(targets)):
            # idx = targets.apply(lambda x: x['feature'] == self.basePrediction, axis=1)
            # targetList.append(targets[idx].get('feature'))
        targetList = targets.get('feature').values # assume that they are already sorted
        # baseList = []
        # for i in range(len(self.baseExplanationDataframe)):
            # idx = self.baseExplanationDataframe.apply(lambda x: x['feature'] == self.basePrediction, axis=1)
            # baseList.append(self.baseExplanationDataframe[idx].get('feature'))
        baseList = self.baseExplanationDataframe.get('feature').values # assume that they are already sorted

        # print(targetList)
        # print(baseList)

        if self.RBO_flag:
            #print("calculating RBO using p=", self.p_RBO)
            rboOutput = self.RBO(targetList,baseList, p=self.p_RBO)
            # print("targetList", targetList)
            # print("baseList", baseList)
            if self.logger:
                print("Internal rboOutput", rboOutput)
            self.tempScore = rboOutput
            
            # #No current best candidate found, set as current best if similarity is different enough
            # #Used to prevent small differences in RBO value being flagged as a viable candidate to keep the perturbation
            # if self.bestScore == None:
            #   if  rboOutput < self.initial_acceptance_threshold:
            #       self.tempScore = rboOutput
            #   else:
            #       return False

            # #Previous best candidate already exists, accept peturbation if below current best
            # #This is very greedy and accepts any outcome that is less similar, not ideal for minimzing number of perturbations
            # #but current perturbed document percentage is low even being this greedy, so not of much concern.
            # else:
            #   if rboOutput < self.bestScore:
            #       self.tempScore = rboOutput
            #   else:
            #       return False
                    
            # self.bestScore = rboOutput
            
            # self.numPerturbations += 1
            
            #Explanation still too similar, accept perturbation and continue search
            if self.tempScore > self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {}".format(self.tempScore, self.success_threshold))      
                return False
            
            return True
        
        elif self.COM_flag:
            #print("calculating com")
            comOutput = self.COM(attacked_text,targets)
             
            if self.logger:
                print("Internal comOutput", comOutput)

            self.tempScore = comOutput
            # if self.bestScore is None:
            #     self.bestScore = self.tempScore
            # elif abs(self.tempScore-self.baseCOM) > abs(self.bestScore-self.baseCOM):      
            #     self.bestScore = comOutput
            
            # self.numPerturbations += 1
            
            #Explanation still too similar, accept perturbation and continue search
            if self._get_score(model_output,_) < self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.baseCOM, self.success_threshold))      
                return False
            
            return True
        
        elif self.COM_proportional_flag: 
            #print("calculating com")
            comOutput = self.COM_proportional(attacked_text,targets)
            
            if self.logger:
                print("Internal comOutput", comOutput)

            self.tempScore = comOutput
            # if self.bestScore is None:
            #     self.bestScore = self.tempScore
            # elif abs(self.tempScore-self.baseCOM) > abs(self.bestScore-self.baseCOM):      
            #     self.bestScore = comOutput
            
            # self.numPerturbations += 1
            
            #Explanation still too similar, accept perturbation and continue search
            if abs(self.tempScore-self.baseCOM) < self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.baseCOM, self.success_threshold))      
                return False
            
            return True

        elif self.COM_rank_weighted_flag:
             #print("calculating com")
            comOutput = self.COM_rank_weight(attacked_text,targets)
            
            if self.logger:
                print("Internal comOutput", comOutput)

            self.tempScore = comOutput
            # if self.bestScore is None:
            #     self.bestScore = self.tempScore
            # elif abs(self.tempScore-self.baseCOM) > abs(self.bestScore-self.baseCOM):      
            #     self.bestScore = comOutput
            
            # self.numPerturbations += 1
            
            #Explanation still too similar, accept perturbation and continue search
            if self._get_score(model_output, None) < self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.baseCOM, self.success_threshold))      
                return False
            
            return True

        elif self.l2_flag:
            #print("calculating l2")
            l2 = self.l2(attacked_text,targets)
            
            if self.logger:
                print("Internal l2", l2)

            self.tempScore = l2
            # if self.bestScore is None:
            #     self.bestScore = self.tempScore
            # elif abs(self.tempScore-self.bestScore) > abs(self.bestScore-self.baseCOM):      
            #     self.bestScore = l2
            
            # self.numPerturbations += 1
            
            #Explanation still too similar, accept perturbation and continue search
            if l2 / self.baseTotalMass < self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.success_threshold))      
                return False
            
            return True
        

        ## New Similarity Measures
        
        elif self.jaccard_flag:
            print("calculating jaccard")
            
            sim = self.jaccard(targetList,baseList)
            
            if self.logger:
                print("Internal Jaccard", sim)

            self.tempScore = sim
            if self.bestScore is None:
                self.bestScore = self.tempScore
            elif abs(self.tempScore-self.bestScore) > self.bestScore:      
                self.bestScore = sim
            
            self.numPerturbations += 1
            
            if sim > self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.bestScore, self.success_threshold))      
                return False
            
            return True
            
            
        
        elif self.jaccard_weighted_flag:
            print("calculating jaccard-weighted")
            
            sim = self.jaccard_weighted(attacked_text,targets)
            
            if self.logger:
                print("Internal Jaccard Weighted", sim)

            self.tempScore = sim
            if self.bestScore is None:
                self.bestScore = self.tempScore
            elif abs(self.tempScore-self.bestScore) > self.bestScore:      
                self.bestScore = sim
            
            self.numPerturbations += 1
            
            if sim > self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.bestScore, self.success_threshold))      
                return False
            
            return True
            
            
        
        elif self.spearman_flag:
            print("calculating spearman")
            
            sim = self.spearman(attacked_text,targets)
            
            if self.logger:
                print("Internal Spearman", sim)

            self.tempScore = sim
            if self.bestScore is None:
                self.bestScore = self.tempScore
            elif abs(self.tempScore-self.bestScore) > self.bestScore:      
                self.bestScore = sim
            
            self.numPerturbations += 1
            
            if sim > self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.bestScore, self.success_threshold))      
                return False
            
            return True
        
        elif self.spearman_weighted_flag:
            print("calculating spearman-weighted")
            
            sim = self.spearman_weighted(attacked_text,targets)
            
            if self.logger:
                print("Internal Spearman Weighted", sim)

            self.tempScore = sim
            if self.bestScore is None:
                self.bestScore = self.tempScore
            elif abs(self.tempScore-self.bestScore) > self.bestScore:      
                self.bestScore = sim
            
            self.numPerturbations += 1
            
            if sim > self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.bestScore, self.success_threshold))      
                return False
            
            return True
        
        elif self.kendall_flag:
            print("calculating kendall")
            
            sim = self.kendall(attacked_text,targets)
            
            if self.logger:
                print("Internal Kendall", sim)

            self.tempScore = sim
            if self.bestScore is None:
                self.bestScore = self.tempScore
            elif abs(self.tempScore-self.bestScore) > self.bestScore:      
                self.bestScore = sim
            
            self.numPerturbations += 1
            
            if sim > self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.bestScore, self.success_threshold))      
                return False
            
            return True
        
        elif self.kendall_weighted_flag:
            print("calculating kendall-weighted")
            
            sim = self.kendall_weighted(attacked_text,targets)
            
            if self.logger:
                print("Internal Kendall Weighted", sim)

            self.tempScore = sim
            if self.bestScore is None:
                self.bestScore = self.tempScore
            elif abs(self.tempScore-self.bestScore) > self.bestScore:      
                self.bestScore = sim
            
            self.numPerturbations += 1
            
            if sim > self.success_threshold:  
                if self.logger:
                    print("FAILED! Explanation still too similar, {} {} {}".format(self.tempScore, self.bestScore, self.success_threshold))      
                return False
            
            return True

    def get_result(self, attacked_text, **kwargs):
        """A helper method that queries ``self.get_results`` with a single
        ``AttackedText`` object."""
        results, search_over = self.get_results([attacked_text], **kwargs)
        result = results[0] if len(results) else None
        return result, search_over


    def get_results(self, attacked_text_list, replacement=None, check_skip=False):
        # print("main attack function triggered")
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        
        # if self.query_budget < float("inf"):
            
        #   #End search early if no initial perturbation found with x attempts or max perturbations reached
        #   if (self.numPerturbations == 0 and self.num_queries > 75) or (self.maxPerturbations <= self.numPerturbations):
        #       queries_left = 11
        #       attacked_text_list = attacked_text_list[:queries_left]
        #   else:
        #       queries_left = self.query_budget - self.num_queries
        #       attacked_text_list = attacked_text_list[:queries_left]

        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)

        # print("target_original", target_original)
        for i, (attacked_text, raw_output) in enumerate(zip(attacked_text_list, model_outputs)):
            # print("candidate", i)
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
                    #base_explanation = self.baseExplanationDataframe, #This was breaking the standard textattack flow
                    # self.attacked_explanation=
                )
            )
        # print("RESULTS GET_RESULTS", len(results))
        return results, False


    def _get_goal_status(self, model_output, attacked_text, check_skip=True):
        # print("Calling _get_goal_status")
        should_skip = check_skip and self._should_skip(model_output, attacked_text)
        if should_skip:
            if self.logger:
                print("STATUS: Skipped")
            return GoalFunctionResultStatus.SKIPPED
        if self.maximizable:
            # print("STATUS: Maximizing")
            return GoalFunctionResultStatus.MAXIMIZING
        if self._is_goal_complete(model_output, attacked_text):
            if self.logger:
                print("STATUS: Succeeded")
            return GoalFunctionResultStatus.SUCCEEDED
        if self.logger:
            print("STATUS: Searching")
        return GoalFunctionResultStatus.SEARCHING

    def _should_skip(self, model_output, attacked_text):
        if (model_output.argmax() != self.ground_truth_output):
            if self.logger:
                print("skipping because the model's prediction was wrong", model_output.argmax(), "vs", self.ground_truth_output)
            return True
        else:
            return False


    def cal_score(self, baseList, targetList, metric="RBO"):
        return None


    def _get_score(self, model_output, _):
       
        if self.RBO_flag:
        #Checks dissimilarity rather than similarity
            if self.tempScore is not None:
                return 1 - self.tempScore
            else:
                return 0
            
        elif self.COM_flag:
            #For center of mass index baseCOM we can move either left or right away from baseCOM. We need to choose the correct side.
            #We the bound the com score in [0,1] by computing the ratio of the distance between the perturbed explanation's com and the baseCOM, 
            # and the maximum possible distance between the baseCOM and the left or right end of the list.
            if self.tempScore is not None:
                if self.tempScore > self.baseCOM: #perturbed com on the right side of the base com
                    return (self.tempScore - self.baseCOM) / ((self.initial_attacked_text_length -1) - self.baseCOM)
                elif self.tempScore < self.baseCOM: #perturbed com on the left side of the base com
                    return (self.baseCOM - self.tempScore) / (self.baseCOM)
                else:
                    return 0
            else:
                return 0
            
        elif self.COM_proportional_flag: 
            if self.tempScore is not None:
                return abs(self.tempScore-self.baseCOM)
            return 0
        
        elif self.COM_rank_weighted_flag: #Currently same as standard COM
            if self.tempScore is not None:
                if self.tempScore > self.baseCOM: #perturbed com on the right side of the base com
                    return (self.tempScore - self.baseCOM) / ((self.initial_attacked_text_length -1) - self.baseCOM)
                elif self.tempScore < self.baseCOM: #perturbed com on the left side of the base com
                    return (self.baseCOM - self.tempScore) / (self.baseCOM)
                else:
                    return 0
            else:
                return 0
        
        elif self.l2_flag:
            if self.tempScore is not None:
                return self.tempScore
            return 0
        elif self.jaccard_flag:
            if self.tempScore is not None:
                return 1 - self.tempScore
            else:
                return 0
        
        elif self.jaccard_weighted_flag:
            if self.tempScore is not None:
                return 1 - self.tempScore
            else:
                return 0
        
        elif self.spearman_flag:
            if self.tempScore is not None:
                return 1 - self.tempScore
            else:
                return 0
        
        elif self.spearman_weighted_flag:
            if self.tempScore is not None:
                return 1 - self.tempScore
            else:
                return 0
        
        elif self.kendall_flag:
            if self.tempScore is not None:
                return 1 - self.tempScore
            else:
                return 0
        
        elif self.kendall_weighted_flag:
            if self.tempScore is not None:
                return 1 - self.tempScore
            else:
                return 0
       
    # def get_explanations(self, originalText, attackedText):
    #     """Returns selected features, base and perturbed explanations after attack completion
    #        Requires the original text and the perturbed text.
    #     """
    #     baseExplanation =  self.generateBaseExplanation(originalText)
        
    #     if originalText == attackedText:
    #         perturbedExplanation = baseExplanation
    #     else:
    #         perturbedExplanation = self.generateExplanation(attackedText)
    #     perturbedExplanation = self.generateExplanation(attackedText)
        
    #     self.baseExplanation = baseExplanation
        
    #     features = self.selectTopFeatures(top_n=self.featureSelector)
        
    #     bTargets = format_explanation_df(baseExplanation[0])
    #     # bTargets = bDF.get('targets')
        
    #     pTargets = format_explanation_df(perturbedExplanation[0])
    #     # pTargets = pDF.get('targets')
        
    #     return(features,bTargets,pTargets)
    
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

            # print("-->calling models on ___ texts", len(uncached_list))
            outputs = self._call_model_uncached(uncached_list)
            for text, output in zip(uncached_list, outputs):
                self._call_model_cache[text] = output
            all_outputs = [self._call_model_cache[text] for text in attacked_text_list]
            return all_outputs
    
    
    def extra_repr_keys(self):
        return []