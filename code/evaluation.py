# Evaluate predicate identification and argument classification

from sklearn.metrics import classification_report, confusion_matrix
import csv


def get_gold_and_pred(path: str, task: str):
    gold = []
    pred = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='\\')
        if task == "predicate_identification":
            for row in reader:
                if row:
                    gold.append(row[10])
                    pred.append(row[11])

        if task == "argument_identification":
            for row in reader:
                if row:
                    if row[12] != "_":
                        gold.append("A")
                    else:
                        gold.append("_")
                    pred.append(row[13])

        if task == "argument_classification":
            for row in reader:
                if row:
                    gold.append(row[12])
                    pred.append(row[13])
    return gold, pred


def main():
    paths = ["../data/en_ewt-up-dev-pred_iden-rule.tsv"]
    tasks = ["predicate_identification", "argument_identification", "argument_classification"]
    for path, task in zip(paths, tasks):
        y_true, y_pred = get_gold_and_pred(path, task)
        print(f"----------Evaluation for {task}----------")
        if task == "predicate_identification":
            print(classification_report(y_true, y_pred, outputdict=True)["PRED"])
        else:
            print(classification_report(y_true, y_pred))
            print(f"----------Confusion matrix for {task}----------")
            print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
