from functools import partial

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

from classifier.classifier import Classifier
from classifier.rich_analyzer import rich_analyzer

__author__ = 'Andrea Esuli'

YES_LABEL = 'yes'
NO_LABEL = 'no'
BINARY_LABELS = [YES_LABEL, NO_LABEL]


class MultiLabelOnlineClassifier(Classifier):
    def __init__(self, name, classes, n_features=(2 ** 21), average=False):
        self._name = name
        analyzer = partial(rich_analyzer, word_ngrams=[2, 3], char_ngrams=[5])
        # int(n_features/len(classes)) makes memory usage constant for the classifier as a whole
        self._vec = HashingVectorizer(n_features=int(n_features / len(classes)), analyzer=analyzer)
        self._clf = {label: PassiveAggressiveClassifier(average=average, n_jobs=-1, max_iter=1000, tol=1e-3) for label
                     in classes}
        for label in self._clf:
            self._clf[label].partial_fit(self._vec.transform(['']), [NO_LABEL], BINARY_LABELS)

    def partial_fit(self, X, y):
        X = self._vec.transform(X)
        label_set = set([label.split(':')[0] for label in y])
        for label in label_set:
            label_X_idx = list()
            label_y = list()
            for idx, y_x in enumerate(y):
                y_label, y_value = y_x.split(':')
                if y_label == label:
                    label_X_idx.append(idx)
                    label_y.append(y_value)
            self._clf[label].partial_fit(X[label_X_idx, :], label_y, BINARY_LABELS)

    def predict(self, X):
        X = self._vec.transform(X)
        predictions = list()
        for label in sorted(self._clf):
            predictions.append(
                [label + ':yes' if label_label == YES_LABEL else label + ':no' for label_label in
                 self._clf[label].predict(X)])
        return list(zip(*predictions))

    def decision_function(self, X):
        X = self._vec.transform(X)
        scores = list()
        for label in sorted(self._clf):
            scores.append([(-score, score) for score in self._clf[label].decision_function(X)])
        return list(zip(*scores))

    def name(self):
        return self._name

    def rename(self, new_name):
        self._name = new_name

    def rename_label(self, label_name, new_name):
        clf = self._clf.pop(label_name, None)
        if clf:
            self._clf[new_name] = clf

    def labels(self):
        return sorted(self._clf)
