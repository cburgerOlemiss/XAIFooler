import scipy 

def monkeypath_itemfreq(sampler_indices):
   return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import eli5
import textattack
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler, MaskingTextSamplers  
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
from datasets import load_dataset, ClassLabel

from sklearnex import patch_sklearn
patch_sklearn(global_patch=True)

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

def generate_filename(args):
    if args.method == "xaifooler":
        if args.similarity_measure == "rbo":
            filename = f"./results/dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.rbo_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
        else:
            filename = f"./results/{args.similarity_measure.upper()}_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.rbo_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
    else:
        filename = f"./results/{args.method.upper()}BASELINE_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.rbo_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
    return filename


def load_args():
    parser = ArgumentParser(description='XAIFOOLER')
    parser.add_argument('--lime-sr', type=int, default=250)
    parser.add_argument('--top-n', type=int, default=3)
    parser.add_argument('--model', type=str, default="cburger/BERT_HS")
    parser.add_argument('--dataset', type=str, default="hate_speech18")
    parser.add_argument('--label-col', type=str, default="label")
    parser.add_argument('--text-col', type=str, default="text")
    # parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-candidate', type=int, default=10)
    parser.add_argument('--success-threshold', type=float, default=0.5)
    parser.add_argument('--rbo-p', type=float, default=0.8)
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--modify-rate', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--min-length', type=int, default=10)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--seed-dataset', type=int, default=12)
    parser.add_argument('--method', type=str, default="xaifooler")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--rerun', action='store_true', default=False)
    parser.add_argument('--similarity-measure', type=str, default='rbo') 
    #options are rbo, l2, com (general definiton), com_rank_weighted (closest to paper), com_proportional (incomplete)
    #New Similarity Measures are: jaccard, kendall, spearman, append _weighted to each for the weighted version

    #Only values that need to be altered for different runs are, similarity-measure, success-threshold, and if using rbo, rbo-p
    
    args = parser.parse_args()
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
        dataset = load_dataset(DATASET_NAME)
        dataset = dataset.rename_column("output_text", "label")
        dataset = dataset.rename_column("input_text", "text")
        # labels = ClassLabel(num_classes=len(categories),names=categories)
        # dataset = dataset.cast_column("label", labels)
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
    # idx = df.apply(lambda x: '<BIAS>' not in x['feature'], axis=1)	
    # df = df[idx]	
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
                    clf=LogisticRegression(class_weight='balanced', random_state=self.random_seed, max_iter=100, n_jobs=-1),
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