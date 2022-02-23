# identify predicates and arguments - store output
from typing import List
import csv
import sys


def read_row_as_list(file_path: str) -> List:
    """
    Read file rows -> List
    """
    rows = []
    if file_path.endswith(".tsv"):
        with open(file_path, newline='') as csvfile:
            content = csv.reader(csvfile, delimiter='\t', quotechar='\\')
            for row in content:
                rows.append(row)
    else:
        with open(file_path, encoding="utf-8") as file:
            content = file.readlines()
        for line in content:
            line = line.rstrip("\n")
            if not line.startswith("#"):
                row = []
                for item in line.split("\t"):
                    row.append(item)
                rows.append(row)
    return rows


def group_rows_by_sents(rows: List) -> List[List]:
    """
    List of rows -> List of sentences of the rows
    """
    sents = []
    sent = []
    for row in rows:
        if len(row) > 1:
            sent.append(row)
        else:
            sents.append(sent)
            sent = []
    return sents


def identify_predicates(sent: List, method:str) -> List[int]:
    """
    For each sent -> find id of predicates in a sentence
    :method: "predgold":using gold predicate senses or "rule":using self-defined rules
    """
    predicate_ids = []
    for row in sent:
        if method == "predgold":
            if (row[10] != "") and (row[10] != "_"):
                predicate_ids.append(int(row[0]))
        elif method == "rule":
            if "." not in row[0]:
                if (row[3] == "VERB") and (row[7] not in ["amod", "case", "mark"]):
                    predicate_ids.append(int(row[0]))
                elif (row[3] == "AUX") and (row[5] != "VerbForm=Fin"):
                    predicate_ids.append(int(row[0]))
                elif (row[4] in ["JJ", "JJR"]) and (("cl" in row[7]) or row[7].endswith("comp")):
                    predicate_ids.append(int(row[0]))
    if len(predicate_ids) == 0:
        predicate_ids.append("_")  # To still give arg label to sents without predicates

    # Make predicate identification column
    predicate_label = "_ " * len(sent)  # Assign "_" to all tokens first
    predicate_label = predicate_label.strip().split()
    for p_id in predicate_ids:
        if type(p_id) == int:
            predicate_label[(p_id - 1)] = "PRED"
    return predicate_label, predicate_ids


def identify_arguments(sent: List[List], p_id: int) -> List:
    """
    A list of input sentences -> List of rows,
                                with predicate and argument labels for each token
    """
    # ! Looping each sentence, each predicate!

    arg_label = "_ " * len(sent)  # Assign "_" to all tokens first
    arg_label = arg_label.strip().split()

    if type(p_id) == int:
        arg_label[(p_id - 1)] = "V"  # The label for that predicate = "V"
        # (p_id-1) = index in the list

        # Find its argument(s)
        ## Rule: ARG if head==V and not det or punct or mark or parataxis
        arg_filter = ["det", "punct", "mark", "parataxis"]
        arg_id = []
        for row in sent:
            if (row[6] == str(p_id)) and (row[7] not in arg_filter):
                arg_id.append(int(row[0]))
        ## if cop -> nsubj=ARG1, head=ARG2?
        #
        for i in arg_id:
            arg_label[(i - 1)] = "ARG"

    return arg_label  # As one column for one predicate in a sentence


def add_each_iter_to_sent(sent, sent_label):
    sent_output = []
    for i, row in enumerate(sent):
        row.append(sent_label[i])
        sent_output.append(row)
    return sent_output


def output_identification(datafile: str, all_sent_output: List, method):
    ext = datafile.split("/")[-1].split(".")[-1]
    outfile = datafile.rstrip("."+ext) + f"-{method}-identification.tsv"
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        for sent in all_sent_output:
            for row in sent:
                writer.writerow(row)
            writer.writerow("")


def main(argv):
    argv = sys.argv[1:]
    
    if not argv:
        argv = [0, "../data/en_ewt-up-dev.conllu", "rule"]

    file_path = argv[1]
    rows = read_row_as_list(file_path)
    sents = group_rows_by_sents(rows)

    # Choice for predicate identification: predgold or rule
    method = argv[2]

    all_sent_output = []
    for sent in sents:
        # Choices: predgold or rule
        predicate_label, predicate_ids = identify_predicates(sent, method)
        sent_no_gold = [row[:10] for row in sent]
        sent_with_pred = add_each_iter_to_sent(sent_no_gold, predicate_label)
        for p_id in predicate_ids:
            arg_label = identify_arguments(sent_with_pred, p_id)
            each_iteration = add_each_iter_to_sent(sent_with_pred, arg_label)
            sent_with_pred = each_iteration
        all_sent_output.append(sent_with_pred)

    output_identification(file_path, all_sent_output, method)

    
if __name__ == "__main__":
    main(argv)
