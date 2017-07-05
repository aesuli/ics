from functools import partial

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.multiclass import unique_labels

from classifier.rich_analyzer import rich_analyzer

__author__ = 'Andrea Esuli'


class OnlineClassifier(object):
    def __init__(self, name, classes, n_features=(2 ** 21), average=False):
        self.name = name
        # TODO add configurable parameters for vec and clf
        # int(n_features/len(classes)) makes memory usage constant for
        # the entire classifier when the one-vs-all multi-class method
        analyzer = partial(rich_analyzer, word_ngrams=[2, 3], char_ngrams=[4])
        self._vec = HashingVectorizer(n_features=int(n_features / len(classes)), analyzer=analyzer)
        self._clf = SGDClassifier(average=average, n_jobs=-1)
        self._clf.partial_fit(self._vec.transform(['']), [classes[0]], classes)

    def partial_fit(self, X, y):
        X = self._vec.transform(X)
        self._clf.partial_fit(X, y, self._clf.classes_)

    def predict(self, X):
        X = self._vec.transform(X)
        return self._clf.predict(X)

    def decision_function(self, X):
        X = self._vec.transform(X)
        return self._clf.decision_function(X)

    def rename_class(self, label_name, new_name):
        self._clf.classes_ = unique_labels(np.asarray([new_name if name == label_name else name for name in self._clf.classes_]))

    def classes(self):
        return self._clf.classes_


if __name__ == '__main__':
    text = 'one two three'
    for clf in [
        OnlineClassifier('test', ['yes', 'no']),
        OnlineClassifier('test', ['yes', 'no'], average=100),
        OnlineClassifier('test', ['yes', 'no'], average=2),
        OnlineClassifier('test', ['yes', 'no', 'maybe']),
        OnlineClassifier('test', ['yes', 'no'], average=100),
        OnlineClassifier('test', ['yes', 'no', 'maybe'], average=2)
    ]:
        print(clf.predict([text]))
        print(clf.decision_function([text]))
        clf.partial_fit(['one two three'], ['yes'])
        print(clf.predict([text]))
        print(clf.decision_function([text]))
        clf.partial_fit(['two three'], ['yes'])
        print(clf.predict([text]))
        print(clf.decision_function([text]))
        clf.partial_fit(['two two'], ['yes'])
        print(clf.predict([text]))
        print(clf.decision_function([text]))
        clf.partial_fit(['one two', 'four three'], ['yes', 'no'])
        print(clf.predict([text, text]))
        print(clf.decision_function([text, text]))
