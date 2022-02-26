import csv
from typing import List


def read_sentences_from_tsv(path):
    sentences = []
    with open(path, encoding='utf-8') as infile:
        csvreader = csv.reader(infile, delimiter='\t', quotechar='\\')
        sentence = []  # prepare a container for the first sentence
        for row in csvreader:
            if row:  # if the line is not empty
                sentence.append(row)  # append info for this token
            else:  # empty lines indicate sentence boundaries
                sentences.append(sentence)
                sentence = []  # prepare a container for the next sentence
    return sentences


def count_predicates(labels):
    return sum([1 for label in labels if label == 'PRED'])


def extract_predicate_labels(sentence, predicate_column):
    predicate_labels = []
    for token in sentence:
        predicate_labels.append(token[predicate_column])
    return predicate_labels


def construct_a_sentence(sentence, pred_column, i=-1):
    result = []
    for token in sentence:
        if i >= 0:
            result.append(token[:pred_column + 1] + [token[pred_column + 1 + i]])
        else:
            result.append(token[:pred_column + 1] + ['_'])
    return result


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


def predict_arguments_for_sentences(sentences, gold_pred_column, pred_pred_column):
    result = []
    for sentence in sentences:
        gold_pred_labels = extract_predicate_labels(sentence, gold_pred_column)
        pred_pred_labels = extract_predicate_labels(sentence, pred_pred_column)
        num_pred_gold = count_predicates(gold_pred_labels)
        num_pred_pred = count_predicates(pred_pred_labels)
        if num_pred_gold == 0 and num_pred_pred == 0:
            result.append([[token + ['_'] for token in sentence]])
            # list for consistency, so we can iterate through all sents in the same way
        else:
            new_sentences = []
            pred_id = 1
            i = 0
            for gold_pred_label, pred_pred_label in zip(gold_pred_labels, pred_pred_labels):

                if gold_pred_label == '_' and pred_pred_label == '_':
                    pred_id += 1
                    continue
                elif gold_pred_label == 'PRED' and pred_pred_label == '_':
                    this_sentence = construct_a_sentence(sentence, pred_pred_column, i)
                    new_sentences.append([token + ['_'] for token in this_sentence])
                    pred_id += 1
                    i += 1
                elif gold_pred_label == '_' and pred_pred_label == 'PRED':
                    this_sentence = construct_a_sentence(sentence, pred_pred_column)
                    pred_arg_labels = identify_arguments(sentence, pred_id)
                    new_sentences.append([token + [label] for token, label in zip(this_sentence, pred_arg_labels)])
                    pred_id += 1
                else:
                    this_sentence = construct_a_sentence(sentence, pred_pred_column, i)
                    pred_arg_labels = identify_arguments(sentence, pred_id)
                    new_sentences.append([token + [label] for token, label in zip(this_sentence, pred_arg_labels)])
                    pred_id += 1
                    i += 1
            result.append(new_sentences)
    return result


def write_sentences_to_tsv(sents, path):
    with open(path, 'w', newline='') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        for sent in sents:
            for s in sent:
                for token in s:
                    csvwriter.writerow(token)
                csvwriter.writerow([])


def extract_gold_arguments(sentence):
    gold_arguments = []
    for token in sentence:
        if token[-1] == "_":
            gold_arguments.append("_")
        elif token[-1] == "V":
            gold_arguments.append("V")
        else:
            gold_arguments.append("ARG")
    return gold_arguments


def extract_gold_arguments_for_sentences(sentences, gold_pred_column, pred_pred_column):
    result = []
    for sentence in sentences:
        gold_pred_labels = extract_predicate_labels(sentence, gold_pred_column)
        pred_pred_labels = extract_predicate_labels(sentence, pred_pred_column)
        num_pred_gold = count_predicates(gold_pred_labels)
        num_pred_pred = count_predicates(pred_pred_labels)
        if num_pred_gold == 0 and num_pred_pred == 0:
            result.append([[token + ['_'] for token in sentence]])
            # list for consistency, so we can iterate through all sents in the same way
        else:
            new_sentences = []
            pred_id = 1
            i = 0
            for gold_pred_label, pred_pred_label in zip(gold_pred_labels, pred_pred_labels):
                if gold_pred_label == '_' and pred_pred_label == '_':
                    pred_id += 1
                    continue
                else:
                    this_sentence = construct_a_sentence(sentence, pred_pred_column, i)
                    pred_arg_labels = extract_gold_arguments(this_sentence)
                    new_sentences.append([token + [label] for token, label in zip(this_sentence, pred_arg_labels)])
                    pred_id += 1
                    i += 1
            result.append(new_sentences)
    return result


def identify_arguments_and_return_output_path(path, method):
    sentences = read_sentences_from_tsv(path)
    gold_predicate = 10
    pred_predicate = 11
    if method == 'rule':
        new_sentences = predict_arguments_for_sentences(sentences, gold_predicate, pred_predicate)
        output_path = path.replace('.tsv', '-arg_iden-rule.tsv')
        write_sentences_to_tsv(new_sentences, output_path)
    else:
        new_sentences = extract_gold_arguments_for_sentences(sentences, gold_predicate, pred_predicate)
        output_path = path.replace('.tsv', '-arg_iden-gold.tsv')
        write_sentences_to_tsv(new_sentences, output_path)
    return output_path
