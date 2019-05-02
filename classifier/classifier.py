from abc import ABCMeta, abstractmethod

SINGLE_LABEL = 'Single-label'
MULTI_LABEL = 'Multi-label'
CLASSIFIER_TYPES = [MULTI_LABEL, SINGLE_LABEL]


def get_classifier_model(type, name, labels):
    if type == SINGLE_LABEL:
        from classifier.online_classifier import OnlineClassifier
        return OnlineClassifier(name, labels)
    elif type == MULTI_LABEL:
        from classifier.multilabel_online_classifier import MultiLabelOnlineClassifier
        return MultiLabelOnlineClassifier(name, labels)
    else:
        raise ValueError('Unknown classifier type')


def get_classifier_type_from_model(model):
    from classifier.online_classifier import OnlineClassifier
    from classifier.multilabel_online_classifier import MultiLabelOnlineClassifier
    if isinstance(model, OnlineClassifier):
        classifier_type = SINGLE_LABEL
    elif isinstance(model, MultiLabelOnlineClassifier):
        classifier_type = MULTI_LABEL
    else:
        classifier_type = 'Custom'
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

