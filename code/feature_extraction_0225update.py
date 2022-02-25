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
    return sentences


def extract_predicate_lemma(sentence):
    # iterate through all the tokens in the sentence until you find the predicate, then return its lemma
    for token in sentence:
        if token[-1] == 'V':
            return token[2]
    # if there are no predicates, returns None


def extract_lemma(token):
    return token[2]


def extract_POS(token):
    return token[3]


def extract_predicate_POS(sentence):
    for token in sentence:
        if token[-1] == 'V':  # ?
            return token[3]


def extract_voice(sentence):
    for token in sentence:
        if token[-1] == 'V':
            if 'Voice=Pass' in token[5]:
                voice = 'Passive'
            else:
                voice = 'Active'
            return voice


def extract_features_and_labels(sentence):
    new_sentence = []
    # extract features that depend on the whole sentence and are the same for all tokens
    predicate_lemma = extract_predicate_lemma(sentence)  # extract lemma of the predicate as feature
    predicate_POS = extract_predicate_POS(sentence)  # extract POS of the predicate as feature
    voice = extract_voice(sentence)  # extract voice of the predicate as feature
    for token in sentence:
        # extract the label
        label = token[-2]
        # check if token is an argument > we only extract features for arguments
        if label not in ['V', '_']:
            # extract features that only depend on this token
            lemma = extract_lemma(token)   # extract lemma as feature
            arg_POS = extract_POS(token)   # extract POS of arguments as feature
            # collect all features
            new_sentence.append([lemma, arg_POS, predicate_lemma, predicate_POS, voice, label])
    return new_sentence


def extract_features_and_labels_from_sentences(sentences):
    return [extract_features_and_labels(sentence) for sentence in sentences]


def write_sentences_to_tsv(sents, path):
    with open(path, 'w', newline='') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['lemma', 'arg_pos', 'pred_lemma', 'pred_pos', 'voice', 'label'])  # write header row with feature names
        for sent in sents:
            for token in sent:
                csvwriter.writerow(token)
            #     if len(sent) > 1 and sent.index(s) != len(sent) - 1:
            #         csvwriter.writerow(['X'])  # indicate repeats of the same sentence
            # csvwriter.writerow(['Y'])  # indicate sentence breaks


def main(path):
    sentences = read_sentences_from_tsv(path)
    new_sentences = extract_features_and_labels_from_sentences(sentences)
    write_sentences_to_tsv(new_sentences, '../data/dev-features.tsv')


# test
main('../data/arg_iden_output.tsv')
