from typing import List
import csv
import os


def read_sentences_from_connlu(path):
    """Read in all sentences from a connlu file"""
    sentences = []
    with open(path, encoding='utf-8') as infile:
        sentence = []
        for line in infile:
            if line.startswith('#'):
                continue
            else:
                row = line.split()
                if not sentence:
                    length = len(row)  # memorize the length of the first token
                if not row:  # if the row is empty, indicating sentence boundary
                    sentences.append(sentence)
                    sentence = []
                elif len(row) > 11:  # sentence has 1 or more predicates
                    sentence.append(row)
                else:  # sentence doesn't have predicates
                    while len(row) < 12:
                        row.append('_')
                    while len(row) < length:  # there are some weird sentences starting with indexes containing .
                        # this accounts for them
                        row.append('_')
                    sentence.append(row)
    return sentences


def identify_predicates(sent: List[List], method: str) -> List[List]:
    """
    For each sent -> find id of predicates in a sentence
    :method: "gold":using gold predicate senses or "rule":using self-defined rules
    """
    sent_with_pred = []
    for row in sent:
        row.insert(11, "_")
        if (row[10] != "") and (row[10] != "_"):
            row[10] = "PRED"
        if method == "gold":
            if row[10] == "PRED":
                row[11] = "PRED"
        elif method == "rule":
            if (row[3] == "VERB") and (row[7] not in ["amod", "case", "mark"]):
                row[11] = "PRED"
            elif (row[3] == "AUX") and (row[5] != "VerbForm=Fin"):
                row[11] = "PRED"
            elif (row[4] in ["JJ", "JJR"]) and (("cl" in row[7]) or row[7].endswith("comp")):
                row[11] = "PRED"
        sent_with_pred.append(row)
    return sent_with_pred


def write_results_pred_ident_to_tsv(output_path: str, all_sent_output: List) -> None:
    """Write results of predicate identification to a file"""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        for sent in all_sent_output:
            for row in sent:
                writer.writerow(row)
            writer.writerow([])  # keep an empty line between every sentence


def identify_predicates_and_return_output_path(path: str, method: str) -> str:
    """Read in all sentences from the file and for each sentence, identify predicates using a rule-based approach or
    gold labels. Write the predictions to a file and return the file path."""
    sents = read_sentences_from_connlu(path)
    all_sent_output = [identify_predicates(sent, method) for sent in sents]
    output_path = path.replace(os.path.splitext(path)[1], f'-pred_iden-{method}.tsv')
    write_results_pred_ident_to_tsv(output_path, all_sent_output)
    return output_path
