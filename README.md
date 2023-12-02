

Work in progress (12/2/23): Currently cleaning up the code for public release.


# XAIFooler

This is the introduction and repository for **XAIFooler** the algorithm behind "Are Your Explanations Reliable?" Investigating the Stability of LIME in Explaining Text Classifiers by Marrying XAI and Adversarial Attack" (EMNLP 2023).


# Requirements

XAIFooler is built on the **TextAttack** Framework and uses the **ELI5** implementation of **LIME**. It is recommended to install both packages in a dedicated environment through Anaconda or similar. Installations on Windows systems may encounter issues with automatic installation of **ELI5** due to deprecated packages. Linux installations have not been problematic.

# Running the Code
Running the code requires only **main.py** be executed. 

## Adversarial Search Options
The possible arguments for adjusting the adversarial search process can be seen directly in **common.py** and are as follows:

 - Method - Which search method to implement, options are XAIFooler, Inherent (None), Random (Greedy), Random (Non-Greedy)
 - LIME's sampling rate (ELI5 has the default implementation at 5000, see the paper for more efficient values for particular datasets)
 - The top-n feature threshold - Which features are required to be held constant between each iteration of the search
 - Max Candidates - The number of nearest neighbors calculated for word replacement
 - Modify Rate - The maximum percentage of possible perturbed words
 - Success Threshold - Not used in the paper's experiments, this sets a similarity threshold to allow early termination of the search once a desired dissimilarity between the original explanation and the perturbed explanation is reached.
 - Similarity Measure - Which measure to use for explanation comparison during the search process (Options are RBO, L2, Center of Mass (COM), Center of Mass (Rank Weighted), Jaccard, Kendall, Spearman, Jaccard (Weighted), Spearman (Weighted), Kendall (Weighted)
 - RBO-p - for the RBO similarity measure, the weight parameter that controls the top-weightedness 
 - Random Seed
 - Model - BERT, RoBERTa, DistilBERT versions fine-tuned on the respective dataset
 - Dataset - IMDB, Symptoms to Diagnosis, Twitter Hate Speech 
 
## Paper

[arXiv](https://arxiv.org/abs/2305.12351)

>@inproceedings{ xaifooler, title={``Are Your Explanations Reliable?" Investigating the Stability of LIME in Explaining Text Classifiers by Marrying XAI and Adversarial Attack}, author={Christopher Burger, Lingwei Chen, Thai Le}, booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing}, year={2023} }
