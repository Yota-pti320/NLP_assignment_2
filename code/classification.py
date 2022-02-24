# load features as input for the classifier

# system output (i. e. the predictions) on the test set need to be produced in the same format
# as the training and development data

from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import csv


# reusing parts of our code from https://github.com/LahiLuk/TMgp4-negation-cue-detection/blob/main/code/SVM.py

def extract_features_and_labels(file_path):
    """Extract a set of features and labels from file."""

    features = []
    labels = []

    with open(file_path, 'r', encoding='utf8') as infile:
        # restval specifies value to be used for missing values
        reader = csv.DictReader(infile, restval='', delimiter='\t', quotechar='\\')
        features_list = reader.fieldnames[:-1]
        label_column = reader.fieldnames[-1]
        for row in reader:
            feature_dict = {}
            for feature_name in features_list:
                if row[feature_name]:  # if there is a value for this feature
                    feature_dict[feature_name] = row[feature_name]
            features.append(feature_dict)
            labels.append(row[label_column])

    return features, labels


def create_classifier(train_features, train_labels):
    """Vectorize features and create classifier from training data."""

    classifier = LinearSVC(random_state=42)
    vec = DictVectorizer()
    train_features_vectorized = vec.fit_transform(train_features)
    classifier.fit(train_features_vectorized, train_labels)

    return classifier, vec


def get_predictions(test_path, vectorizer, classifier):
    """Vectorize test features and get predictions."""

    test_features = extract_features_and_labels(test_path)[0]
    test_features_vectorized = vectorizer.transform(test_features)
    print(f'Using vectors of dimensionality {test_features_vectorized.shape[1]}')
    predictions = classifier.predict(test_features_vectorized)

    return predictions


def write_predictions_to_file(predictions, path):
    with open(path, encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar='\\')
        with open('../data/dev-predictions.tsv', 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(next(reader)[:-1] + ['prediction'])  # write header row
            for row, prediction in zip(reader, predictions):
                writer.writerow(row[:-1] + [prediction])


def main(train_path, test_path) -> None:
    train_features, gold_labels = extract_features_and_labels(train_path)
    classifier, vectorizer = create_classifier(train_features, gold_labels)
    predictions = get_predictions(test_path, vectorizer, classifier)
    # print(type(predictions))
    write_predictions_to_file(predictions, test_path)


# test
main('../data/train-features.tsv', '../data/dev-features.tsv')
