from abc import ABCMeta, abstractmethod

_CLASSIFIER_TYPES = ['Statistical', 'Neural', 'Custom']


def get_classifier_model(type, name, labels):
    if type == _CLASSIFIER_TYPES[0]:
        from classifier.online_classifier import OnlineClassifier
        return OnlineClassifier(name, labels)
    elif type == _CLASSIFIER_TYPES[1]:
        from classifier.pytorch_classifier import LSTMClassifier
        return LSTMClassifier(name, labels)
    else:
        raise ValueError('Unknown classifier type')


def get_classifier_type_from_model(model):
    from classifier.online_classifier import OnlineClassifier
    from classifier.pytorch_classifier import LSTMClassifier
    if isinstance(model, OnlineClassifier):
        classifier_type = _CLASSIFIER_TYPES[0]
    elif isinstance(model, LSTMClassifier):
        classifier_type = _CLASSIFIER_TYPES[1]
    else:
        classifier_type = _CLASSIFIER_TYPES[-1]
    return classifier_type


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

