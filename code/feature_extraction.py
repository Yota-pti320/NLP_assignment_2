# extract features - store output
import csv


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
        # sentences.append(sentence)  # append the last sentence if the last line of the file is not empty > check
        # # after Rod finishes!
    return sentences


def construct_a_sentence(sentence, pred_column, i):
    result = []
    for token in sentence:
        result.append(token[:pred_column] + [token[pred_column + i]])
    return result


def extract_predicate_lemma(sentence):
    for token in sentence:
        if token[-1] == 'V':
            return token[2]


def extract_lemma(token):
    return token[2]


def extract_features_and_labels(sentence, pred_column):
    new_sentence = []
    # extract features that depend on the whole sentence and are the same for all tokens
    predicate_lemma = extract_predicate_lemma(sentence)  # extract lemma of the predicate as feature
    for token in sentence:
        # extract features that only depend on this token
        lemma = extract_lemma(token)   # extract lemma as feature
        # collect all features
        features = [lemma, predicate_lemma]
        # extract the label
        label = [token[pred_column]]
        new_sentence.append(features + label)
    return new_sentence


def extract_features_and_labels_from_sentences(sentences, pred_column):
    result = []
    for sentence in sentences:
        num_pred = len(sentence[0]) - pred_column
        if num_pred > 1:  # sentence has more than 1 predicates
            new_sentences = []
            for i in range(num_pred):
                # align the labels depending on the predicate in question
                this_sentence = construct_a_sentence(sentence, pred_column, i)
                new_sentence = extract_features_and_labels(this_sentence, pred_column)
                new_sentences.append(new_sentence)
            result.append(new_sentences)
        else:  # sentence has 1 or 0 predicates
            new_sentence = extract_features_and_labels(sentence, pred_column)
            result.append([new_sentence])  # list for consistency, so we can iterate through all sents in the same way
    return result


def write_sentences_to_tsv(sents, path):
    with open(path, 'w', newline='') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        for sent in sents:
            for s in sent:
                for token in s:
                    csvwriter.writerow(token)
                if len(sent) > 1 and sent.index(s) != len(sent) - 1:
                    csvwriter.writerow(['X'])  # indicate repeats of the same sentence
            csvwriter.writerow(['Y'])  # indicate sentence breaks


def main(path):
    sentences = read_sentences_from_tsv(path)
    pred_column = 11
    new_sentences = extract_features_and_labels_from_sentences(sentences, pred_column)
    write_sentences_to_tsv(new_sentences, path.replace('.tsv', '-features.tsv'))


# test
main('../data/dev2.tsv')
