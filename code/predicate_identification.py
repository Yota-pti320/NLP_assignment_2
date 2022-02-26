# identify predicates and arguments - store output
from typing import List
import csv


def read_sentences_from_connlu(path):
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


def identify_predicates(sent: List[List], method:str) -> List[List]:
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


def output_identification(output_path: str, all_sent_output: List, method):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        for sent in all_sent_output:
            for row in sent:
                writer.writerow(row)
            writer.writerow("")


def identify_predicates_and_return_output_path(path, method):
    sents = read_sentences_from_connlu(path)
    all_sent_output = []
    for sent in sents:
        sent_with_pred = identify_predicates(sent, method)
        all_sent_output.append(sent_with_pred)
    output_path = path.replace('.conllu', f'-pred_iden-{method}.tsv')
    output_identification(output_path, all_sent_output, method)
    return output_path
