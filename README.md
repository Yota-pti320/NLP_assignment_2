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
All the experiments carried out during the assignment can be replicated by running the `code/main.py` script. 
The script needs to be run from the command line, and two arguments need to be passed to it: 
the path to the .connlu file that will be used for training the classifier, and the path to the test dataset. 


Example call:
```
python code/main.py data/en_ewt-up-train.conllu data/en_ewt-up-test.conllu
```

Important: The files need to be in .conllu format. The files used in the experiment can be downloaded from 
https://github.com/System-T/UniversalPropositions/tree/master/UP_English-EWT.

### Experimental setup
Here, we describe the pipeline used in `main.py`.

Firstly, a rule based system is deployed to identify predicates in the training and test datasets. The performance of 
the system is evaluated on the test dataset. 

Then, a rule based system is used to identify arguments for the predicates that were extracted in the first step. 
This is again done for the training and test datasets, and evaluated on the test dataset. 
The resulting errors stem both from errors produced by predicate identification, and from argument identification.

The third and final step of the pipeline is training a machine learning system to classify the arguments that have been 
identified in step 2. We use the LinearSVC implementation of SVM from Scikit-learn: 
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC. 
We chose this algorithm because SVM has shown good performance on the SRL task 
([Introduction to the CoNLL-2005 Shared Task: Semantic Role Labeling](https://aclanthology.org/W05-0620) 
(Carreras & Màrquez, 2005)). 
The algorithm is trained on only those instances from the training dataset that have been identified as arguments after 
the first two rule-based steps, and used to predict labels of only those test instances that have been identified as 
arguments in step 2. We take this approach because we think it will help the classifier perform better on the test data.
For example, the classifier will learn to predict that some arguments are actually not arguments, and assign 
them the label `_` when tested. This could not happen if we instead trained the system on gold arguments.

Lastly, we also evaluate our rule-based argument identification system by using it on gold label predicates, to see how 
well it performs on its own. We do the same for argument classification: we train the system on all the gold labelled 
arguments, and use it to predict the labels for all the gold labelled arguments in the test 
dataset. 
We do this to evaluate the performance of our classifier, not taking into account the error propagation from the 
previous steps of the pipeline.

### Brief data description
The datasets that are used were provided by Universal Proposition Banks. All datasets are in CoNLL-U format. The training and development datasets have the same attribute values:
* Column0  ID: Word index, integer starting at 1 for each new sentence; may be a range for tokens with multiple words.
* Column1  FORM: Word form or punctuation symbol.
* Column2  LEMMA: Lemma or stem of word form.
* Column3  UPOSTAG: Universal part-of-speech tag drawn from our revised version of the Google universal POS tags.
* Column4  XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
* Column5  FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
* Column6  HEAD: Head of the current token, which is either a value of ID or zero (0).
* Column7  DEPREL: Universal Stanford dependency relation to the HEAD (root if HEAD = 0) or a defined language-specific subtype of one.
* Column8  DEPS: List of secondary dependencies (head-deprel pairs).
* Column9  MISC: Any other annotation.
* Column10 Gold predicates
* Column11-N Gold arguments

From column 0 to column 9, word-level linguistic information is provided. From column 10 to the last column, gold labels are provided. In column 10, gold predicate information is labelled with predicate sense. From column 11 to the last column, argument information is labelled.


### Identification
#### Predificates identification
A rule-based system was created to extract predicates. By implementing this system into our experiment, we hope to add one column next to the gold predicates column, then check if they match.
##### Rule-based approach
A rule-based system was created to detect predicates. Before applying the rules, a preprocessing step is performed: predicate senses are converted to ‘PRED’ to indicate the token is a gold predicate.  
The rule-based system operates on the following simple conditions: 
1) If ‘VERB’ occurs in column 3, and if dependency relation in column 7 is not ‘amod’, ‘case’ or ‘mark’, assign the label ‘PRED’ to column 11. 
2) If ‘AUX’ occurs in column 3, and ‘VerbForm==Fin’ doesn't occur in column 5, assign the label ‘PRED’ to column 11.
3) If ‘JJ’ or ‘JJR’ occurs in column 4, and ‘cl’ occurs in column 7 or it ends with ‘comp’, assign the label ‘PRED’ to column 11.

These three rules are motivated by observation about the dataset.  
For the first rule, we choose all verbs since predicates are mostly verbs. However, verbs with a ‘amod’, ‘case’ and ‘mark’ dependency relation are excluded because of the following reasons:

1) verb + ‘amod’ will act as a modifier, not a predicate

    *President Bush on Tuesday nominated two individuals to replace `retiring` jurists on ……*
2) verb + ’case’ will not act as predicate

    *`Following` on the heels of Ben’s announcement yesterday.*
3)  verb + ‘mark’ will not act as predicate

    *……rather than have a general rule `concerning` how direct access should work for all parties.*

For the second rule, we observed that most of auxiliaries are labelled as predicates in the dataset. But we noticed there are some exceptions when they are finite, which means they have a non-empty mood. Basically, those auxiliaries are ‘could’, ‘would’, ‘may’, ‘will’, ‘should’ etc. 

For the third rule, we tried to find regular patterns for adjectives and comparatives of adjectives that were labelled as predicates. We observed that adjectives that have specified dependency relations such as ‘acl’, ‘acl:relcl’, ‘advcl’, ‘ccomp’ and ‘xcomp’, are more likely to be labelled as predicates.


#### Arguments identification
For the second step, a rule-based approach is used to identify arguments for the predicates that were extracted in the previous step.
The rule-based approach operates on the following simple conditions:
1) Iterate the sentences. If a sentence has predicates, extract the index of predicates.
2) Iterate sentence again, if token in the sentence has the index of head word just same as the index of predicate, and its dependency relation is not in ["det", "punct", "mark", "parataxis"], it will be labelled ‘ARG’. This rule is motivated by our observation of the data.


### Arguments classification
In the third step, an SVM classifier is trained to assign specific argument labels based on predicates and arguments detected before. Extracted features (explained in detail in the next section) will be fed into our system.  The classification instances are the instances that have been identified as arguments in the previous step. In other words, both training and prediction will only be performed on instances that have an “ARG” label. Ideally, a well-performing classifier can further classify arguments accurately as ARG0, ARG1, ARG2, ARG-TMP, etc. On the other hand, another classifier is developed to be trained and predict on gold predicates and gold arguments. Evaluation will be done on both systems to showcase the performance of a standalone classification task and the effect of error propagation.







