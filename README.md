# NLPtech Assignment 2 part 1: Traditional Semantic Role Labeling

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

### Brief data description
The datasets that are used were provided by the Vrije Universiteit Amsterdam and Universal Proposition Banks. All datasets are in CoNLL-U format. The training and development datasets have the same attribute values:
* Column0  Index
* Column1  Token
* Column2  Lemma
* Column3  POS-simple version
* Column4  POS-complex version
* Column5  Morphological information
* Column6  Index of head word
* Column7  Dependency relation
* Column8  Combination of column6 and column7
* Column9  Notes
* Column10 Gold predicates
* Column11-N Gold arguments

From column 0 to column 9, word-levelled information and dependency information are provided as features can be used. From column 10 to the last column, gold data is provided. In column 10, gold predicate information is labelled with predicate sense. If a token is not a predicate, it is labelled ‘_’ . From column 11 to the last column, argument information is labelled.

### Identification
#### Predificates identification
A rule-based system was created to extract predicates. By implementing this system into our experiment, we hope to add one column next to the gold predicates column, then check if they match.
##### Rule-based approach
Before applying the rule-based predicates identification approach, there are two preprocessing steps: First, converting predicates information of column 10 into binary, ‘PRED’ or ‘_’.  Second, setting two methods for generating column 11. One is “gold”, another is “rule”. Implementing ‘gold’ will generate gold predicates in column11. Implementing ‘rule’ will detect predicates based on the rule-based approach.
The rule-based system operates on the following simple conditions, it takes column 3,4,5,7 and 10 into account, the output is a list of assigned class in the same order as the tokens:
1) If ‘VERB’ occurs in column3, and if dependency relation in column7 is not ‘amod’, ‘case’ or ‘mark’, assign the label ‘PRED’ into column 11. If not, assign ‘_’.
2) If ‘AUX’ occurs in column3, and ‘VerbForm==Fin’ doesn't occur in column5, assign the label ‘PRED’ into column 11. If not, assign ‘_’.
3) If ‘JJ’ or ‘JJR’ occurs in column4, and ‘cl’ occurs in column7 or something in column7 endswith ‘comp’ occurs, assign the label ‘PRED’ into column 11. If not, assign ‘_’.
##### Motivation of rules
For building predicates identification rules, we get insight from previous related work and diving deeply into the Universal Proposition Banks dataset.  
For the first rule, we choose all verbs since predicates basically are verbs. Besides, excluding ‘amod’, ‘case’ and ‘mark’ , there are examples from dataset:
1) verb+‘amod’ will act as modifier not a predicate.

    *President Bush on Tuesday nominated two individuals to replace `retiring` jurists on ……*
2) verb+’case’ will not act as predicate.

    *_Following_ on the heels of Ben’s announcement yesterday.*
3) verb+ ‘mark’ will not act as predicate.

    *……every party should be the exception to the suspension rather than have a general rule _concerning_ how direct access should work for all parties.*

For the second rule, we observed that most of auxiliaries are labelled as predicates in the dataset. But we noticed there are some exceptions when the form of verb is finite verb, which means verbs or auxiliaries that have a non-empty mood. Basically, those auxiliaries are ‘could’, ‘would’, ‘may’, ‘will’, ‘should’ etc. 

For the third rule, we tried to find regular patterns for adjectives and comparatives of adjectives that were labelled as predicates. We observed that adjectives have specified dependency relations such as ‘acl’, ‘acl:relcl’, ‘advcl’, ‘ccomp’ and ‘xcomp’, are more likely to be labelled as predicate


#### Arguments identification

