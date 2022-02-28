# NLP_assignment_2

### Authors
YC Roderick Li, Lahorka Nikolovski, Panagiota Tselenti and Jingyue Zhang

### Repository structure
```
.
├── code
│   ├── arg_identification.py
│   ├── classification.py
│   ├── evaluation.py
│   ├── feature_extraction.py
│   ├── main.py
│   ├── predicate_identification.py
│   └── requirements.txt
├── data
│   └── README.md
└── README.md
```

### Usage
All the experiments carried out during the assignment can be replicated by running `code/main.py` script. The script needs to be run from the command line, and two arguments need to be passed to it: the path to the .connlu file that will be used for training the classifier, and the path to the test dataset. 


Example call:
```
python code/main.py data/en_ewt-up-train.conllu data/en_ewt-up-test.conllu
```

Important: The files have to be in .conllu format. The files used in the experiment can be downloaded from https://github.com/System-T/UniversalPropositions/tree/master/UP_English-EWT.

### Experimental setup
Here, we describe the pipeline used in main.py.

Firstly, a rule based system is deployed to identify predicates in the training and test datasets. The performance of the system is evaluated on the test dataset. 

Then, a rule based system is used to identify arguments for the predicates that were extracted in the first step. This is again done for the training and test datasets, and evaluated on the test dataset. The resulting errors stem both from errors produced by predicate identification, and from argument identification.

The third and final step of the pipeline is training a machine learning system to classify the arguments that have been identified in step 2. We use the LinearSVC implementation of SVM from Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC. We chose this algorithm because SVM has shown good performance on the SRL task ([Introduction to the CoNLL-2005 Shared Task: Semantic Role Labeling](https://aclanthology.org/W05-0620) (Carreras & Màrquez, 2005)). 
The algorithm is trained on only those instances from the training dataset that have been identified as arguments after the first two rule-based steps, and used to predict labels of only those test instances that have been identified as arguments in step 2. The results are evaluated by looking at labels predicted by our system for all the gold labelled arguments in the test dataset.

Lastly, we also evaluate our rule-based argument identification system by using it on gold label predicates, to see how well it performs on its own. We do the same for argument classification: we train the system on all the gold labelled arguments, and use it to predict the label for all the gold labelled arguments in the test 
dataset. We do this to evaluate the performance of our classifier, not taking into account the error propagation from the previous steps of the pipeline.
