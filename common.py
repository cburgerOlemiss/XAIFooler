import scipy 
import numpy
def monkeypath_itemfreq(sampler_indices):
   return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import eli5
import textattack
from textattack.shared.utils import words_from_text
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler, MaskingTextSamplers  
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
from datasets import load_dataset, ClassLabel

#from sklearnex import patch_sklearn
#patch_sklearn(global_patch=True)

from sklearn.linear_model import LogisticRegression

# from cuml.common.device_selection import set_global_device_type, get_global_device_type
# from cuml.linear_model import LogisticRegression as CULogisticRegression
# print('default execution device:', get_global_device_type())
# set_global_device_type('gpu')
# print('new device type:', get_global_device_type())

import sk2torch

from argparse import ArgumentParser
import math
import numpy as np

from scipy import stats

def SM(list1, list2):
    coef, p = stats.spearmanr(list1, list2)
    return 1- max(0, coef)

def p_generator(p,d):
    def sum_series(p, d):
       # tail recursive helper function
       def helper(ret, p, d, i):
           term = math.pow(p, i)/i
           if d == i:
               return ret + term
           return helper(ret + term, p, d, i+1)
       return helper(0, p, d, 1)
    
    return  1 - math.pow(p, d-1) + (((1-p)/p) * d *(np.log(1/(1-p)) - sum_series(p, d-1)))

def find_p(top_n_mass = 0.9):
    # top_n_mass = 0.90 #What percentage of mass we wish to have on the top n features.
    n = 3 #Number of top features
    for i in range(1,100,1):
        p = i/100
        output = p_generator(p,n)
        if abs(output-top_n_mass) < 0.01:
            print("Set rbo_p = ",p, " for ", output*100, "% mass to be assigned to the top ", n, " features." )
            break

    # compute RBO
def RBO(list1,list2,p):
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
def COM(attacked_text,perturbedExplanation):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=perturbedExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        #occurences = p_df.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)
        
        total_mass = np.sum(np.abs(ordered_weights))

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
def COM_rank_weight(attacked_text,perturbedExplanation):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=perturbedExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        #occurences = p_df.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)*(j+1)

        #print(attacked_text_list)        
        #print(ordered_weights)
        
        total_mass = np.sum(np.abs(ordered_weights))
            
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
def COM_proportional(attacked_text,perturbedExplanation):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=perturbedExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        #occurences = p_df.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)

        #print(attacked_text_list)        
        #print(ordered_weights)
        
        total_mass = np.sum(np.abs(ordered_weights))

        center_mass = total_mass / 2
        #print("Center mass = ", center_mass)
              
        current_mass = 0
        for i in range(len(ordered_weights)):
            current_mass += abs(ordered_weights[i])
            if current_mass >= center_mass and ordered_weights[i+1] != 0:
                    break
    
        return current_mass / total_mass
    
def l2(attacked_text,baseExplanation,perturbedExplanation):
        #t0 = timer()
        attacked_text_list = words_from_text(attacked_text)
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        # occurences = perturbedExplanation.get('value').values.tolist()
        base_ordered_weights = [0] * len(attacked_text_list)
        baseFeatures = format_explanation_df(baseExplanation[0], baseExplanation[1]).get('feature').values.tolist()
        baseWeights = format_explanation_df(baseExplanation[0], baseExplanation[1]).get('weight').values.tolist()

        feat2weightBase = dict(zip(baseFeatures, baseWeights))
        for j in range(len(attacked_text_list)):
            base_ordered_weights[j] = feat2weightBase.get(attacked_text_list[j], 0)


        baseOrderedWeights = np.array(base_ordered_weights)
        #baseTotalMass = np.sum(np.abs(base_ordered_weights))

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)

        l2 = np.linalg.norm(baseOrderedWeights - np.array(ordered_weights))
        #t1 = timer()
        #print("l2 took...", t1 - t0)

        return l2
    
def jaccard(list1,list2):
        set1 = set(list1)
        set2 = set(list2)
        
        return (len(set.intersection(set1,set2)) / len(set.union(set1,set2)) )
    
def jaccard_weighted(baseExplanation,perturbedExplanation):
        #Need ordered weights and features from the base explanation
        
        #for each feature in base explanation, if it does not exist in the perturbed explanation reduce 
        # sim by its (normalized) weight in the base explanation
        
        sim = 1
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])
        features = p_df.get('feature').values.tolist()

        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        #print(exdf)
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseAbsWeightSum = sum(exdf['weight'])
        baseOrderedExplanation = exdf
        
        for i in range(len(baseOrderedExplanation)):
            if baseOrderedExplanation['feature'][i] not in features:
                       sim -= baseOrderedExplanation['weight'][i]/baseAbsWeightSum
        
        return sim
            
def kendall(baseExplanation,perturbedExplanation):
        
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()
        current_dissonance = 0


        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseOrderedExplanation = exdf
        
        l1 = len(baseOrderedExplanation['feature'])
        l2 = len(features)
        
                       
        if l1 > l2:
            max_dissonance = l1
            diff = l1-l2
            shorter_explanation = l2
        else:
            max_dissonance = l2
            diff = l2-l1
            shorter_explanation = l1
                       
        
        for i in range(shorter_explanation):
            if baseOrderedExplanation['feature'][i] != features[i]:
                       current_dissonance += 1
                        
        current_dissonance += diff
        
        return 1 - (current_dissonance / max_dissonance)
    
def kendall_weighted(baseExplanation,perturbedExplanation):
        
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])
        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()
        current_dissonance = 0
        dissonant_weights = 0

        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseAbsWeightSum = sum(exdf['weight'])
        baseOrderedExplanation = exdf
        
        
        l1 = len(baseOrderedExplanation)
        l2 = len(features)
        
                       
        if l1 > l2:
            max_dissonance = l1
            diff = l1-l2
        else:
            max_dissonance = l2
            diff = l2-l1
                       
        for i in range(l1):
            if i > l2-1:
                       current_dissonance += 1
                       dissonant_weights += baseOrderedExplanation['weight'][i]/baseAbsWeightSum
            elif baseOrderedExplanation['feature'][i] != features[i]:
                       current_dissonance += 1
                       dissonant_weights += baseOrderedExplanation['weight'][i]/baseAbsWeightSum

        current_dissonance += diff
        return (1 - (current_dissonance / max_dissonance)) * (1-dissonant_weights)
        
        
    
def spearman(baseExplanation,perturbedExplanation):
        
        #l1 distance between features
        
        #for each feature in base explanation, calculate the distance if it remains in the perturbed explanation,
        #otherwise penalize with some value (half of the length of the explanation (uniform mean of possible distance))
        
        #maximum total distance is 1/2 floor(explanation_Size squared)
        
        #sum distances divide by max distance

        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()

        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseOrderedExplanation = exdf
        
        current_distance = 0
        
        l1 = len(baseOrderedExplanation)
        
        max_distance = int((l1*l1) / 2)
        penalty = int(l1 / 2)
                                        
        for i in range(l1):
            if baseOrderedExplanation['feature'][i] in features:
                       current_distance += abs(i - features.index(baseOrderedExplanation['feature'][i]))
            else:
                       current_distance += penalty
                            
        return 1 - (current_distance / max_distance)               
            
def spearman_weighted(baseExplanation,perturbedExplanation):
        
        #for each feature in base explanation, calculate the distance if it remains in the perturbed explanation,
        #otherwise penalize with normalized weight * maximum distance
        
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

                
        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()
        
        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseAbsWeightSum = sum(exdf['weight'])
        baseOrderedExplanation = exdf
        
        
        
        l1 = len(baseOrderedExplanation)
        
        max_distance = int((l1*l1) / 2)
        current_distance = max_distance
      
        
        missing_features = 0
        missing_feature_weight = 0
                       
        for i in range(l1):
            if baseOrderedExplanation['feature'][i] in features:
                       current_distance -= abs(i - features.index(baseOrderedExplanation['feature'][i]))
            else:
                       missing_features += 1
                       missing_feature_weight += baseOrderedExplanation['weight'][i]/baseAbsWeightSum
                       
                       current_distance -= max_distance * baseOrderedExplanation['weight'][i]/baseAbsWeightSum
        
        
        return (current_distance / max_distance)




def generate_comparative_similarities(attacked_text,baseExplanation,perturbedExplanation,RBO_weights = None):
    sims = []

    if RBO_weights is None:
         RBO_weights = [0.6,0.7,0.8,0.9]
            
    df1 = format_explanation_df(baseExplanation[0], target=baseExplanation[1])
    df2 = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

    baseList = df1.get('feature').values
    perturbedList = df2.get('feature').values
    
    
    #Set base Center of mass
    #COM
    #baseCOM = COM(attacked_text,format_explanation_df(baseExplanation[0], baseExplanation[1]))

    #COM_proportional 
    #basePropCOM = COM_proportional(attacked_text,format_explanation_df(self.baseExplanation[0], baseExplanation[1]))

    #COM_rank_weighted
    #baseRwCOM = COM_rank_weight(attacked_text,format_explanation_df(self.baseExplanation[0], baseExplanation[1]))

    for w in RBO_weights:
        sims.append(RBO(perturbedList,baseList,w))
    sims.append(jaccard(perturbedList,baseList))
    sims.append(jaccard_weighted(baseExplanation,perturbedExplanation))
    #sims.append(COM(attacked_text,perturbedExplanation))
    #sims.append(COM_proportional(attacked_text,perturbedExplanation))
    #sims.append(COM_rank_weight(attacked_text,perturbedExplanation))
    sims.append(kendall(baseExplanation,perturbedExplanation))
    sims.append(kendall_weighted(baseExplanation,perturbedExplanation))
    sims.append(spearman(baseExplanation,perturbedExplanation))
    sims.append(spearman_weighted(baseExplanation,perturbedExplanation))
    sims.append(l2(attacked_text,baseExplanation,perturbedExplanation))

    
    return sims


def generate_filename(args):
    if args.method == "xaifooler":
        if args.similarity_measure == "rbo":
            filename = f"./results/dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
        else:
            filename = f"./results/{args.similarity_measure.upper()}_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
    elif args.method == "ga":
        if args.similarity_measure == "rbo":
            filename = f"./results/GA/{args.crossover}--{args.parent_selection}--dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
        else:
            filename = f"./results/GA/{args.crossover}--{args.parent_selection}--{args.similarity_measure.upper()}_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"

    else:
        filename = f"./results/{args.method.upper()}BASELINE_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
    return filename


def load_args():
    parser = ArgumentParser(description='XAIFOOLER')
    parser.add_argument('--lime-sr', type=int, default=None)
    parser.add_argument('--top-n', type=int, default=3)
    parser.add_argument('--model', type=str, default="thaile/distilbert-base-uncased-s2d-saved")
    parser.add_argument('--dataset', type=str, default="s2d")
    parser.add_argument('--label-col', type=str, default="label")
    parser.add_argument('--text-col', type=str, default="text")
    # parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-candidate', type=int, default=10)
    parser.add_argument('--success-threshold', type=float, default=0.5)
    parser.add_argument('--rbo-p', type=float, default=0.8)
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--modify-rate', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--min-length', type=int, default=10)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--seed-dataset', type=int, default=12)
    parser.add_argument('--method', type=str, default="xaifooler")
    #'xaifooler', ga, random, truerandom
    #parser.add_argument('--search-method',type=str,default = 'default')
    parser.add_argument('--crossover', type=str, default = '1point')
    #uniform, 1point
    parser.add_argument('--parent-selection', type=str, default = 'truncation')
    #roulette, truncation
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--rerun', action='store_true', default=False)
    parser.add_argument('--similarity-measure', type=str, default='rbo')
    #options are rbo, l2, com (general definiton), com_rank_weighted (closest to paper), com_proportional (incomplete)
    #New Similarity Measures are: jaccard, kendall, spearman, append _weighted to each for the weighted version

    args, unknown = parser.parse_known_args()
    return args

def load_dataset_custom(DATASET_NAME, seed):
    print("LOADING", DATASET_NAME)

    if DATASET_NAME == 'imdb':
        dataset = load_dataset(DATASET_NAME)
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)
        dataset_test = dataset['test']
        categories = dataset_train_valid['train'].unique('label')

    if DATASET_NAME == 'gb':
        DATASET_NAME = "md_gender_bias"
        dataset = load_dataset(DATASET_NAME, 'convai2_inferred', split='train').filter(lambda example: example["binary_label"] in set([0, 1]))
        dataset = dataset.shuffle(seed=seed).select(range(30000))
        dataset = dataset.rename_column("binary_label", "label")
        categories = dataset.unique('label')
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed)
        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    elif DATASET_NAME == 's2d':
        DATASET_NAME = 'gretelai/symptom_to_diagnosis'
        dataset = load_dataset("gretelai/symptom_to_diagnosis")
        dataset = dataset.rename_column("output_text", "label")
        dataset = dataset.rename_column("input_text", "text")
        dataset = dataset.class_encode_column("label")
        categories = dataset['train'].unique('label')
        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    elif DATASET_NAME == 'hate_speech_offensive':
        dataset = load_dataset(DATASET_NAME, split='train').filter(lambda example: example["class"] in set([0, 1]))
        dataset = dataset.rename_column("class", "label")
        categories = dataset.unique('label')
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed)
        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    elif DATASET_NAME == 'tweets_hate_speech_detection':
        ### tweets_hate_speech_detection
        dataset = load_dataset(DATASET_NAME, split='train')
        dataset = dataset.rename_column("tweet", "text")
        categories = dataset.unique('label')
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed)

        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    return dataset_train_valid, dataset_test, categories


def format_explanation_df(explanation, target=-1):	
    df = eli5.format_as_dataframes(explanation)['targets']	
    df['abs_weight'] = np.abs(df['weight'])	
    df = df.sort_values(by=['abs_weight'], ascending=False)	
    if target > -1:	
        idx = df.apply(lambda x: x['target'] == target, axis=1)	
        df = df[idx]	
        df = df.reset_index(drop=True)	
    return df


def check_bias(x, y):
    return '<BIAS>' not in x

def load_stopwords():
    import nltk
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

def generate_explanation_single(self, document, custom_n_samples=None, debug=False, return_explainer=False):
    if type(document) == str:
        document = textattack.shared.attacked_text.AttackedText(document)

    explainer = TextExplainer(
                    clf=LogisticRegression(class_weight='balanced', random_state=self.random_seed, max_iter=300, n_jobs=-1),
                    # clf = CULogisticRegression(class_weight='balanced', max_iter=100, verbose=True),
                    # clf=DecisionTreeClassifier(class_weight='balanced', random_state=self.random_seed, max_depth=10),
                    vec=CountVectorizer(stop_words='english', lowercase=True),
                    n_samples=self.limeSamples if not custom_n_samples else custom_n_samples, 
                    random_state=self.random_seed, 
                    sampler=MaskingTextSampler(random_state=self.random_seed))
    
    prediction = self.categories[self.get_output(document)]
    probability = self.pred_proba(document)
    probability = float(probability.max())
   
    start = timer()
    explainer.fit(document.text, self.pred_proba_LIME_Sampler)
    end = timer()

    if debug:
        print("Lime took...", end - start)

    explanation = explainer.explain_prediction(target_names=self.categories,
                                                feature_names=explainer.vec_.get_feature_names_out(),
                                                feature_filter=check_bias,)
   
    if return_explainer:
        return explainer,explanation,prediction,probability

    return (explanation,prediction,probability)