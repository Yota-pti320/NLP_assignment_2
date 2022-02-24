# extract features - store output
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


def construct_a_sentence(sentence, pred_column, i):
    result = []
    for token in sentence:
        result.append(token[:pred_column] + [token[pred_column + i]])
    return result


def extract_predicate_lemma(sentence):
    # iterate through all the tokens in the sentence until you find the predicate, then return its lemma
    for token in sentence:
        if token[-1] == 'V':
            return token[2]
    # if there are no predicates, returns None


def extract_lemma(token):
    return token[2]


def extract_features_and_labels(sentence, pred_column):
    new_sentence = []
    # extract features that depend on the whole sentence and are the same for all tokens
    predicate_lemma = extract_predicate_lemma(sentence)  # extract lemma of the predicate as feature
    for token in sentence:
        # check if token is an argument > we only extract features for arguments
        if token[-1] not in ['V', '_']:
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
        csvwriter.writerow(['lemma', 'pred_lemma', 'label'])  # write header row with feature names
        for sent in sents:
            for s in sent:
                for token in s:
                    csvwriter.writerow(token)
            #     if len(sent) > 1 and sent.index(s) != len(sent) - 1:
            #         csvwriter.writerow(['X'])  # indicate repeats of the same sentence
            # csvwriter.writerow(['Y'])  # indicate sentence breaks


def main(train_path, test_path):
    train_sentences = read_sentences_from_connlu(train_path)
    test_sentences = read_sentences_from_tsv(test_path)
    pred_column = 11
    # extract features and gold labels from the training file
    new_sentences = extract_features_and_labels_from_sentences(train_sentences, pred_column)
    write_sentences_to_tsv(new_sentences, '../data/train-features.tsv')
    # extract features from the test file
    new_sentences = extract_features_and_labels_from_sentences(test_sentences, pred_column)
    write_sentences_to_tsv(new_sentences, '../data/dev-features.tsv')


# test
main('../data/en_ewt-up-train.conllu', '../data/en_ewt-up-dev-rule-identification.tsv')
