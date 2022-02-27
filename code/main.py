# main.py file carries out the entire experiment (feature extraction, training, testing)
# using command line arguments for potential parameters (e.g. filepaths)

import csv
import sys
from predicate_identification import identify_predicates_and_return_output_path
from arg_identification import identify_arguments_and_return_output_path
from feature_extraction import extract_features_and_return_output_path
from classification import classify_arguments_and_return_predictions
from evaluation import get_gold_and_pred, generate_confusion_matrix
from sklearn.metrics import classification_report


def write_predictions_to_file(path, predictions):
    with open(path, encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar='\\')
        with open(path.replace('.tsv', '-predictions.tsv'), 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
            i = 0
            for row in reader:
                if not row:
                    writer.writerow([])
                elif row[-1] != 'ARG':
                    writer.writerow(row)
                else:
                    writer.writerow(row[:-1] + [predictions[i]])
                    i += 1


def main(args=None):
    if not args:
        args = sys.argv[1:]

    train_path, test_path = args

    train_path_preds = identify_predicates_and_return_output_path(train_path, 'rule')
    train_path_args = identify_arguments_and_return_output_path(train_path_preds, 'rule')
    train_path_features = extract_features_and_return_output_path(train_path_args)

    test_path_preds = identify_predicates_and_return_output_path(test_path, 'rule')

    y_true, y_pred = get_gold_and_pred(test_path_preds, 'predicate_identification')
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)["PRED"]
    print(report)  # of all gold predicates, how many did we identify as predicates
    print(generate_confusion_matrix(y_true, y_pred))

    test_path_args = identify_arguments_and_return_output_path(test_path_preds, 'rule')

    y_true, y_pred = get_gold_and_pred(test_path_args, 'argument_identification')
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)["ARG"]
    print(report)  # of all gold arguments, how many did we identify as arguments
    print(generate_confusion_matrix(y_true, y_pred))

    test_path_features = extract_features_and_return_output_path(test_path_args)

    predictions = classify_arguments_and_return_predictions(train_path_features, test_path_features)
    write_predictions_to_file(test_path_args, predictions)

    # of all gold arguments, how many did we classify correctly
    y_true, y_pred = get_gold_and_pred(test_path_args.replace('.tsv', '-predictions.tsv'), 'argument_classification')
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # evaluate argument identification after gold predicate identification
    test_path_preds = identify_predicates_and_return_output_path(test_path, 'gold')
    test_path_args = identify_arguments_and_return_output_path(test_path_preds, 'rule')
    y_true, y_pred = get_gold_and_pred(test_path_args, 'argument_identification')
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)["ARG"]
    print(report)  # of all gold arguments, how many did we identify as arguments
    print(generate_confusion_matrix(y_true, y_pred))

    # evaluate classification after gold predicate and argument identification
    train_path_preds = identify_predicates_and_return_output_path(train_path, 'gold')
    train_path_args = identify_arguments_and_return_output_path(train_path_preds, 'gold')
    train_path_features = extract_features_and_return_output_path(train_path_args)

    test_path_args = identify_arguments_and_return_output_path(test_path_preds, 'gold')
    test_path_features = extract_features_and_return_output_path(test_path_args)

    predictions = classify_arguments_and_return_predictions(train_path_features, test_path_features)
    write_predictions_to_file(test_path_args, predictions)

    y_true, y_pred = get_gold_and_pred(test_path_args.replace('.tsv', '-predictions.tsv'), 'argument_classification')
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))


if __name__ == '__main__':
    main()
