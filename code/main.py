import csv
import sys
from predicate_identification import identify_predicates_and_return_output_path
from arg_identification import identify_arguments_and_return_output_path
from feature_extraction import extract_features_and_return_output_path
from classification import classify_arguments_and_return_predictions
from evaluation import get_gold_and_pred, generate_confusion_matrix, calculate_precision_recall_f1_score
from sklearn.metrics import classification_report
import numpy as np


def write_predictions_to_file(in_path: str, out_path: str, predictions: np.ndarray):
    """
    Read in a file containing results of argument identification and write the contents to a new file, changing
    'ARG' label to a label obtained after argument classification.
    """
    with open(in_path, encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar='\\')
        with open(out_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
            i = 0
            for row in reader:
                if not row:  # empty line
                    writer.writerow([])
                elif row[-1] != 'ARG':
                    writer.writerow(row)
                else:
                    writer.writerow(row[:-1] + [predictions[i]])
                    i += 1


def main():
    """
    Accept paths to training and testing datasets from the command line and run and evaluate all the steps of
    the experiment.

    1. Identify predicates: rule-based approach. Evaluate performance on the test set.
    2. Identify arguments for the predicates identified in step 1.: rule-based approach. Evaluate performance on the
    test set.
    3. Train ML classifier on train dataset arguments obtained after rule-based predicate and argument identification.
    Use the classifier to predict labels for test set arguments identified in step 2. Evaluate the performance of the
    classifier in relation to all gold arguments from the test set.
    4. Evaluate the performance of rule-based argument identification on gold predicates.
    5. Evaluate the performance of argument classification after training the classifier on all gold arguments from the
    training set and using it to predict labels for all gold arguments from the test set.

    """
    train_path, test_path = sys.argv[1:]

    # identify predicates on the test set - rule-based approach
    test_preds_path = identify_predicates_and_return_output_path(test_path, 'rule')
    # evaluate the performance
    y_true, y_pred = get_gold_and_pred(test_preds_path, 'predicate_identification')
    # report = classification_report(y_true, y_pred, digits=3, output_dict=True)["PRED"]
    print("-----Evaluation on rule-based predicate identification------")
    print(calculate_precision_recall_f1_score(y_true, y_pred, metric='PRED'))
    # print(report)  # of all gold predicates, how many did we identify as predicates
    print(generate_confusion_matrix(y_true, y_pred))

    # identify arguments for the predicates identified in the previous step - rule-based approach
    test_args_path = identify_arguments_and_return_output_path(test_preds_path, 'rule')
    # evaluate the performance
    y_true, y_pred = get_gold_and_pred(test_args_path, 'argument_identification')
    # report = classification_report(y_true, y_pred, digits=3, output_dict=True)["ARG"]
    print("-----Evaluation on rule-based argument identification (predicates: rules)------")
    print(calculate_precision_recall_f1_score(y_true, y_pred, metric='ARG'))
    # print(report)  # of all gold arguments, how many did we identify as arguments
    print(generate_confusion_matrix(y_true, y_pred))

    # identify predicates and arguments in the training dataset - rule-based approach
    train_preds_path = identify_predicates_and_return_output_path(train_path, 'rule')
    train_args_path = identify_arguments_and_return_output_path(train_preds_path, 'rule')

    # extract features for only those arguments the rule-based approach identified
    train_features_path = extract_features_and_return_output_path(train_args_path)
    test_features_path = extract_features_and_return_output_path(test_args_path)

    # and use them to train the classifier and obtain predictions on the test set
    predictions = classify_arguments_and_return_predictions(train_features_path, test_features_path)
    write_predictions_to_file(test_args_path, test_args_path.replace('.tsv', '-predictions.tsv'), predictions)

    # evaluate classifier: of all gold arguments, how many did we classify correctly?
    y_true, y_pred = get_gold_and_pred(test_args_path.replace('.tsv', '-predictions.tsv'), 'argument_classification')
    print("-----Evaluation on argument classification (predicates: rules; arguments: rules)------")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # evaluate rule-based argument identification after gold predicate identification
    test_preds_path = identify_predicates_and_return_output_path(test_path, 'gold')
    test_args_path = identify_arguments_and_return_output_path(test_preds_path, 'rule')
    y_true, y_pred = get_gold_and_pred(test_args_path, 'argument_identification')
    # report = classification_report(y_true, y_pred, digits=3, output_dict=True)["ARG"]
    print("-----Evaluation on rule-based argument identification (predicates: gold)------")
    print(calculate_precision_recall_f1_score(y_true, y_pred, metric='ARG'))
    # print(report)  # of all gold arguments, how many did we identify as arguments
    print(generate_confusion_matrix(y_true, y_pred))

    # evaluate classification after gold predicate and argument identification
    # use all arguments to train the classifier, and test on all arguments from the training set
    train_preds_path = identify_predicates_and_return_output_path(train_path, 'gold')
    train_args_path = identify_arguments_and_return_output_path(train_preds_path, 'gold')
    train_features_path = extract_features_and_return_output_path(train_args_path)

    test_args_path = identify_arguments_and_return_output_path(test_preds_path, 'gold')
    test_features_path = extract_features_and_return_output_path(test_args_path)

    predictions = classify_arguments_and_return_predictions(train_features_path, test_features_path)
    write_predictions_to_file(test_args_path, test_args_path.replace('.tsv', '-predictions.tsv'), predictions)

    y_true, y_pred = get_gold_and_pred(test_args_path.replace('.tsv', '-predictions.tsv'), 'argument_classification')
    print("-----Evaluation on argument classification (predicatse: gold; arguments: gold)------")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))


if __name__ == '__main__':
    main()
