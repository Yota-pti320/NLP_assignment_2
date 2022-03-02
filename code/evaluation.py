from sklearn.metrics import classification_report, confusion_matrix
import csv
import pandas as pd


def get_gold_and_pred(path: str, task: str):
    """Extract gold and predicted labels from a file and return them."""
    gold, pred = [], []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='\\')
        if task == "predicate_identification":
            for row in reader:
                if row:
                    if row[10] == 'PRED' or row[11] == 'PRED':   # if gold or predicted label is PRED
                        gold.append(row[10])
                        pred.append(row[11])
        if task == "argument_identification":
            for row in reader:
                if row:
                    if row[-2] not in ['_', 'V'] or row[-1] == 'ARG':   # if gold or predicted label is an ARG label
                        if row[-2] == "_":
                            gold.append("_")
                        elif row[-2] == "V":
                            gold.append("V")
                        else:
                            gold.append("ARG")
                        pred.append(row[-1])
        if task == "argument_classification":
            for row in reader:
                if row:
                    if row[-2] not in ['V', '_']:   # if gold label is ARG label
                        gold.append(row[-2])
                        pred.append(row[-1])
    return gold, pred


# reusing my code from https://github.com/LahiLuk/TMgp4-negation-cue-detection/blob/main/code/utils.py

def calculate_precision_recall_f1_score(gold_labels, predictions, metric=None, digits=3):
    """Calculate evaluation metrics."""
    # get the report in dictionary form
    report = classification_report(gold_labels, predictions, zero_division=0, output_dict=True)
    # transform dictionary into a dataframe and round the results
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(digits)
    df_report['support'] = df_report['support'].astype(int)
    if metric:
        return df_report.loc[metric]
    return df_report


def generate_confusion_matrix(gold_labels, predictions):
    """Generate a confusion matrix."""
    labels = sorted(set(gold_labels))
    cf_matrix = confusion_matrix(gold_labels, predictions, labels=labels)
    # transform confusion matrix into a dataframe
    df_cf_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)
    return df_cf_matrix
