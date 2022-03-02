import csv
from typing import List


def read_sentences_from_tsv(path: str) -> List[List[List[str]]]:
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


def count_predicates(labels: List[str]) -> int:
    return sum([1 for label in labels if label == 'PRED'])


def extract_predicate_labels(sentence: List[List[str]], predicate_column: int) -> List[str]:
    return [token[predicate_column] for token in sentence]


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


def predict_arguments_for_sentences(sentence, gold_pred_column, pred_pred_column, method):
    """For each sentence, predict arguments for each of its predicates. Store all sentences with
    predicted labels to a list and return it."""
    # result = []
    # for sentence in sentences:
    gold_pred_labels = extract_predicate_labels(sentence, gold_pred_column)
    pred_pred_labels = extract_predicate_labels(sentence, pred_pred_column)
    num_pred_gold, num_pred_pred = count_predicates(gold_pred_labels), count_predicates(pred_pred_labels)
    if num_pred_gold == 0 and num_pred_pred == 0:  # if there are no predicates, "predict" label _ for each token
        return [[token + ['_'] for token in sentence]]
        # list for consistency, so we can iterate through all sents in the same way
    else:
        new_sentences = []  # we will create a separate sentence for every predicate and append it to this list
        pred_id = 1  # keep track of token index (column[0] of each row; starts with 1)
        i = 1  # count number of gold predicates to know which column to extract for the labels
        for gold_pred_label, pred_pred_label in zip(gold_pred_labels, pred_pred_labels):
            if gold_pred_label == '_' and pred_pred_label == '_':
                pred_id += 1
                continue
            elif gold_pred_label == 'PRED' and pred_pred_label == '_':
                this_sentence = [token[:pred_pred_column + 1]  # + 1 because we want to include the predicate column
                                 + [token[pred_pred_column + i]] for token in sentence]
                new_sentences.append([token + ['_'] for token in this_sentence])  # no predicate identified, so no args
                # can be predicted for this sent
                pred_id += 1
                i += 1
            elif gold_pred_label == '_' and pred_pred_label == 'PRED':
                this_sentence = [token[:pred_pred_column + 1] + ['_'] for token in sentence]  # there were no args in gold
                pred_arg_labels = identify_arguments(sentence, pred_id)
                new_sentences.append([token + [label] for token, label in zip(this_sentence, pred_arg_labels)])
                pred_id += 1
            else:
                this_sentence = [token[:pred_pred_column + 1] + [token[pred_pred_column + i]] for token in sentence]
                if method == 'rule':
                    pred_arg_labels = identify_arguments(sentence, pred_id)
                else:
                    pred_arg_labels = ['ARG' if token[-1] not in ['_', 'V'] else token[-1] for token in
                                       this_sentence]
                new_sentences.append([token + [label] for token, label in zip(this_sentence, pred_arg_labels)])
                pred_id += 1
                i += 1
        return new_sentences


def write_results_arg_ident_to_tsv(sents, path):
    """Write results of argument identification to a file"""
    with open(path, 'w', newline='') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        for sent in sents:
            for s in sent:
                for token in s:
                    csvwriter.writerow(token)
                csvwriter.writerow([])  # keep an empty line between every sentence


def identify_arguments_and_return_output_path(path, method):
    """Read in all sentences from the file and for each predicate, identify arguments using a rule-based approach or
    gold labels. Write the predictions to a file and return the file path."""
    sentences = read_sentences_from_tsv(path)
    gold_pred_column = 10
    pred_pred_column = 11
    all_sent_output = [predict_arguments_for_sentences(sent, gold_pred_column, pred_pred_column, method) for sent in
                       sentences]
    output_path = path.replace('.tsv', f'-arg_iden-{method}.tsv')
    write_results_arg_ident_to_tsv(all_sent_output, output_path)
    return output_path
