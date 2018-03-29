from abc import ABCMeta, abstractmethod


class Classifier(object, metaclass=ABCMeta):
    @abstractmethod
    def partial_fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def decision_function(self, X):
        pass

    @abstractmethod
    def rename_class(self, label_name, new_name):
        pass

    @abstractmethod
    def classes(self):
        pass