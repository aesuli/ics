import cherrypy
import numpy
from classifier.online_classifier import OnlineClassifier
from db.sqlalchemydb import SQLAlchemyDB

__author__ = 'Andrea Esuli'


class WebClassifierCollection(object):
    def __init__(self, db_connection_string):
        self._db_connection_string = db_connection_string
        try:
            self._db = SQLAlchemyDB(db_connection_string)
        except Exception as e:
            print(e)
            raise e

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        result = []
        for name in self._db.classifier_names():
            classifier_info = dict()
            classifier_info['name'] = name
            classifier_info['classes'] = self._db.get_classifier_classes(name)
            classifier_info['created'] = self._db.get_classifier_creation_time(name)
            classifier_info['updated'] = self._db.get_classifier_last_update_time(name)
            result.append(classifier_info)
        return result

    @cherrypy.expose
    def create(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            classes = data['classes']
        except KeyError:
            try:
                classes = data['classes[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify the classes'
        if type(classes) is not list:
            cherrypy.response.status = 400
            return 'Must specify at least two classes'
        classes = list(set(classes))
        if len(classes) < 2:
            cherrypy.response.status = 400
            return 'Must specify at least two classes'
        try:
            overwrite = data['overwrite']
        except KeyError:
            overwrite = False
        if not overwrite:
            if self._db.classifier_exists(name):
                cherrypy.response.status = 403
                return '%s is already in the collection' % name
        clf = OnlineClassifier(name, classes, average=20)
        self._db.create_classifier(name, classes, clf)
        return 'Ok'

    @cherrypy.expose
    def update(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            X = data['X']
        except KeyError:
            try:
                X = data['X[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify a vector of strings (X)'
        try:
            y = data['y']
        except KeyError:
            try:
                y = data['y[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify a vector of assigned classes (y)'
        X = numpy.atleast_1d(X)
        y = numpy.atleast_1d(y)

        if len(X) != len(y):
            cherrypy.response.status = 400
            return 'Must specify the same numbers of strings and labels'

        for (content, label) in zip(X, y):
            self._db.create_training_example(name, content, label)

        clf = self._db.get_classifier_model(name)
        clf.partial_fit(X, y)
        self._db.update_classifier_model(name, clf)

        return 'Ok'

    @cherrypy.expose
    def delete(self, name):
        try:
            self._db.delete_classifier_model(name)
        except KeyError as e:
            cherrypy.response.status = 404
            return '%s does not exits' % name
        else:
            return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classes(self, name):
        clf = self._db.get_classifier_model(name)
        return list(clf.classes())

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classify(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            X = data['X']
        except KeyError:
            try:
                X = data['X[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify a vector of strings (X)'
        X = numpy.atleast_1d(X)
        clf = self._db.get_classifier_model(name)
        return [[y] for y in [self._classify(name, clf, x) for x in X]]

    def _classify(self, classifier_name, model, x):
        label = self._db.get_label(classifier_name, x)
        if label is not None:
            return label
        return model.predict([x])[0]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def score(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            X = data['X']
        except KeyError:
            try:
                X = data['X[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify a vector of strings (X)'
        X = numpy.atleast_1d(X)
        clf = self._db.get_classifier_model(name)
        scores = clf.decision_function(X)
        classes = clf.classes()
        if classes.shape[0] == 2:
            return [dict(zip(classes, [-value, value])) for value in scores]
        else:
            return [dict(zip(classes, values)) for values in scores]

    @cherrypy.expose
    def version(self):
        return "0.1.0"


if __name__ == "__main__":
    with WebClassifierCollection('sqlite:///%s' % 'test.db') as wcc:
        cherrypy.quickstart(wcc, '/service')
