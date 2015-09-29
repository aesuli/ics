from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


__author__ = 'Andrea Esuli'


class OnlineClassifier(object):
    def __init__(self, name, classes, n_features=(2 ** 21), average=False):
        self.name = name
        # TODO add configurable parameters for vec and clf
        # int(n_features/len(classes)) makes memory usage constant for the entire classifier
        self._vec = HashingVectorizer(n_features=int(n_features / len(classes)))
        self._clf = PassiveAggressiveClassifier(average=average, n_jobs=-1)
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

