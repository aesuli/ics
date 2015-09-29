import shelve
import os.path
from classifier.online_classifier import OnlineClassifier

__author__ = 'Andrea Esuli'


class ClassifierCollection(object):
    def __init__(self, name, path=os.path.curdir):
        self._name = name
        self._shelf = shelve.open(os.path.join(path, name))

    def close(self):
        self._shelf.close()

    def create(self, name, classes, overwrite=False):
        clf = OnlineClassifier(name, classes, average=20)
        if not overwrite:
            if name in self._shelf:
                raise KeyError('Key \'%s\' already in the collection' % name)
        self._shelf[name] = clf

    def list(self):
        return self._shelf.keys()

    def update(self, name, X, y):
        clf = self._shelf[name]
        clf.partial_fit(X, y)
        self._shelf[name] = clf

    def classes(self, name):
        clf = self._shelf[name]
        return clf.classes()

    def classify(self, name, X):
        clf = self._shelf[name]
        return clf.predict(X)

    def score(self, name, X):
        clf = self._shelf[name]
        scores = clf.decision_function(X)
        classes = clf.classes()
        if classes.shape[0] == 2:
            return [dict(zip(classes, [-value, value])) for value in scores]
        else:
            return [dict(zip(classes, values)) for values in scores]

    def delete(self, name):
        del self._shelf[name]


if __name__ == "__main__":
    text = 'one two three'
    cc = ClassifierCollection('test')
    cc.create('test', ['yes', 'no'], True)
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['one two three'], ['yes'])
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['two three'], ['yes'])
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['two two'], ['yes'])
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['one two', 'four three'], ['yes', 'no'])
    print(cc.classify('test', [text, text]))
    print(cc.scores('test', [text, text]))
    cc.create('test', ['yes', 'no', 'boh'], True)
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['one two three'], ['yes'])
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['two three'], ['yes'])
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['two two'], ['yes'])
    print(cc.classify('test', [text]))
    print(cc.scores('test', [text]))
    cc.update('test', ['one two', 'four three', 'one'], ['yes', 'no', 'boh'])
    print(cc.classify('test', [text, text]))
    print(cc.scores('test', [text, text]))
    cc.close()