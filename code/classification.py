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
    predictions = classifier.predict(test_features_vectorized)
    return predictions


def write_predictions_to_features_file(predictions, in_path, out_path):
    """Write predictions to a file containing features."""
    with open(in_path, encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar='\\')
        with open(out_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(next(reader) + ['prediction'])  # write header row
            for row, prediction in zip(reader, predictions):
                writer.writerow(row + [prediction])


def classify_arguments_and_return_predictions(train_features_path, test_features_path):
    """Train an SVM classifier on training set arguments. Return predictions of the classifier on test set arguments."""
    train_features, gold_labels = extract_features_and_labels(train_features_path)
    classifier, vectorizer = create_classifier(train_features, gold_labels)
    predictions = get_predictions(test_features_path, vectorizer, classifier)
    write_predictions_to_features_file(predictions, test_features_path,
                                       test_features_path.replace('.tsv', '-predictions.tsv'))
    return predictions
