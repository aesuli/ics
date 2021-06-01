from abc import abstractmethod, ABC

__author__ = 'Andrea Esuli'

YES_LABEL = 'yes'
NO_LABEL = 'no'
BINARY_LABELS = [NO_LABEL, YES_LABEL]


def create_classifier_model(name, labels):
    from ics.classifier.lri_online_classifier import LRIOnlineClassifier
    return LRIOnlineClassifier(name, labels)


class Classifier(ABC):

    @abstractmethod
    def learn(self, X, y):
        pass

    @abstractmethod
    def classify(self, X, multilabel=True):
        pass

    @abstractmethod
    def decision_function(self, X):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def rename(self, new_name):
        pass

    @abstractmethod
    def labels(self):
        pass

    @abstractmethod
    def rename_label(self, label_name, new_name):
        pass

    @abstractmethod
    def delete_label(self, label_name):
        pass

    @abstractmethod
    def add_label(self, label_name):
        pass

    @abstractmethod
    def get_label_classifier(self, label_name):
        pass
