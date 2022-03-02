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
            return token[2].lower()
    # if there are no predicates, returns None


def extract_lemma(token):
    return token[2].lower()


def extract_POS(token):
    return token[3]


def extract_head_word(token, sentence):
    head_index = token[6]
    for token in sentence:
        if token[0] == head_index:
            head_word = token[2].lower()
            return head_word


def extract_dependency_relation(token):
    return token[7]


def extract_position_arg(token, sentence):
    for row in sentence:
        if row[-1] == "V":
            predicate_id = row[0]
    if predicate_id:
        if float(token[0]) < float(predicate_id):
            position = "before"
        elif float(token[0]) > float(predicate_id):
            position = "after"
        return position


def extract_predicate_POS(sentence):
    for token in sentence:
        if token[-1] == 'V':
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
    """Extract a list of features for every predicate in a sentence as well as the gold label."""
    output = []
    # extract features that depend on the whole sentence and are the same for all tokens
    predicate_lemma = extract_predicate_lemma(sentence)  # extract lemma of the predicate as feature
    predicate_POS = extract_predicate_POS(sentence)  # extract POS of the predicate as feature
    voice = extract_voice(sentence)  # extract voice of the predicate as feature
    for token in sentence:
        # check if token is an argument > we only extract features for arguments
        if token[-1] not in ['V', '_']:
            # extract features that only depend on this token
            lemma = extract_lemma(token)   # extract lemma as feature
            head_word = extract_head_word(token, sentence)
            dep_rel = extract_dependency_relation(token)
            arg_POS = extract_POS(token)   # extract POS of arguments as feature
            position = extract_position_arg(token, sentence)
            label = token[-2]
            output.append([lemma, arg_POS, head_word, dep_rel, predicate_lemma, predicate_POS, position, voice,
                           label])
    return output


def write_results_feature_extraction_to_tsv(sents, path):
    """Write result of feature extraction to a file"""
    with open(path, 'w', newline='') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['lemma', 'arg_pos', 'head_word', 'dep_rel', 'pred_lemma', 'pred_pos', 'position', 'voice',
                            'label'])  # write header row with feature names
        for sent in sents:
            for token in sent:
                csvwriter.writerow(token)


def extract_features_and_return_output_path(path: str) -> str:
    """From a file with predicates and arguments identified, extract selected features for every argument.
    Write results to a file and return a path to it"""
    sentences = read_sentences_from_tsv(path)
    all_sent_output = [extract_features_and_labels(sentence) for sentence in sentences]
    output_path = path.replace('.tsv', '-features.tsv')
    write_results_feature_extraction_to_tsv(all_sent_output, output_path)
    return output_path
