from collections import defaultdict
from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier

from ics.classifier.classifier import Classifier, NO_LABEL, YES_LABEL, BINARY_LABELS
from ics.classifier.lri import LightweightRandomIndexingVectorizer
from ics.classifier.rich_analyzer import rich_analyzer

__author__ = 'Andrea Esuli'


class LRIOnlineClassifier(Classifier):

    def __init__(self, name, classes, n_features=(2 ** 18), average=False, C=1.0, fit_intercept=True, max_iter=1000,
                 tol=1e-3, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0,
                 loss="hinge"):
        self._name = name
        self.average = average
        analyzer = partial(rich_analyzer, word_ngrams=[2, 3], char_ngrams=[5])
        self._vec = LightweightRandomIndexingVectorizer(n_features=n_features,
                                                        analyzer=analyzer)
        self._clf = {
            label: PassiveAggressiveClassifier(average=average, n_jobs=-1,
                                               max_iter=max_iter, tol=tol, C=C, fit_intercept=fit_intercept,
                                               early_stopping=early_stopping, validation_fraction=validation_fraction,
                                               n_iter_no_change=n_iter_no_change, shuffle=shuffle, verbose=verbose,
                                               loss=loss) for
            label in classes}
        for label in self._clf:
            self._clf[label].partial_fit(self._vec.transform(['']), [NO_LABEL],
                                         BINARY_LABELS)

    def learn(self, X, y):
        if len(y) == 0:
            return
        X = self._vec.transform(X)
        if type(y[0]) == list or type(y[0]) == tuple:
            label_X_idx = defaultdict(list)
            label_y = defaultdict(list)
            for idx, y_x in enumerate(y):
                y_label, y_value = y_x
                label_X_idx[y_label].append(idx)
                if type(y_value) is str:
                    y_value = y_value == 'True' or y_value == 'true' or y_value == '1' or y_value == '+1' or y_value == 'yes'
                elif type(y_value) is int:
                    y_value = y_value > 0
                if y_value:
                    label_y[y_label].append(YES_LABEL)
                else:
                    label_y[y_label].append(NO_LABEL)
            for label in set(label_y):
                self._clf[label].partial_fit(X[label_X_idx[label], :],
                                             label_y[label],
                                             BINARY_LABELS)
        else:
            for label in self._clf:
                y_label = [YES_LABEL if value == label else NO_LABEL for value
                           in y]
                self._clf[label].partial_fit(X, y_label, BINARY_LABELS)

    def classify(self, X, multilabel=True):
        if multilabel:
            X = self._vec.transform(X)
            predictions = list()
            for label in self.labels():
                predictions.append(
                    [
                        (label, True) if label_label == YES_LABEL else (
                            label, False)
                        for label_label in self._clf[label].predict(X)])
            return list(zip(*predictions))
        else:
            scores = self.decision_function(X)
            predictions = list()
            labels = self.labels()
            for x_score in scores:
                predictions.append(labels[np.argmax(x_score)])
            return predictions

    def decision_function(self, X):
        X = self._vec.transform(X)
        scores = list()
        for label in self.labels():
            scores.append([score for score in
                           self._clf[label].decision_function(X)])
        return list(zip(*scores))

    def name(self):
        return self._name

    def rename(self, new_name):
        self._name = new_name

    def labels(self):
        return sorted(self._clf)

    def rename_label(self, label_name, new_name):
        clf = self._clf.pop(label_name, None)
        if clf:
            self._clf[new_name] = clf

    def delete_label(self, label_name):
        if label_name in self._clf:
            del self._clf[label_name]

    def add_label(self, label_name):
        self._clf[label_name] = PassiveAggressiveClassifier(
            average=self.average,
            n_jobs=-1,
            max_iter=1000,
            tol=1e-3)
        self._clf[label_name].partial_fit(self._vec.transform(['']), [NO_LABEL],
                                          BINARY_LABELS)

    def get_label_classifier(self, label_name):
        classifier = LRIOnlineClassifier(label_name, [label_name])
        classifier._clf[label_name] = deepcopy(self._clf[label_name])
        return classifier
