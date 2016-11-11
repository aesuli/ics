import csv
import json
import logging
import os
import shutil
from collections import defaultdict
from uuid import uuid4
import cherrypy
from chardet.universaldetector import UniversalDetector
from cherrypy.lib.static import serve_file
import numpy
from classifier.online_classifier import OnlineClassifier
from db.sqlalchemydb import SQLAlchemyDB, DBLock
from util.util import get_fully_portable_file_name, logged_call, logged_call_with_args
from webservice.background_processor import BackgroundProcessor

__author__ = 'Andrea Esuli'

DOWNLOAD_DIR = os.path.join(os.path.abspath('.'), 'downloads')
UPLOAD_DIR = os.path.join(os.path.abspath('.'), 'uploads')

MAX_BATCH_SIZE = 1000


class WebClassifierCollection(object):
    def __init__(self, db_connection_string, background_processor):
        self._db_connection_string = db_connection_string
        self._db = SQLAlchemyDB(db_connection_string)
        self._background_processor = background_processor

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
            classifier_info['created'] = str(self._db.get_classifier_creation_time(name))
            classifier_info['updated'] = str(self._db.get_classifier_last_update_time(name))
            classifier_info['size'] = self._db.get_classifier_examples_count(name)
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
        name = str.strip(name)
        if len(name) < 1:
            cherrypy.response.status = 400
            return 'Classifier name too short'
        classes = map(str.strip, classes)
        classes = list(set(classes))
        if len(classes) < 2:
            cherrypy.response.status = 400
            return 'Must specify at least two classes'
        for class_name in classes:
            if len(class_name) < 1:
                cherrypy.response.status = 400
                return 'Class name too short'
        try:
            overwrite = data['overwrite']
        except KeyError:
            overwrite = False
        with _lock_trainingset(self._db, name), _lock_model(self._db, name):
            if not self._db.classifier_exists(name):
                self._db.create_classifier(name, classes)
            elif not overwrite:
                cherrypy.response.status = 403
                return '%s is already in the collection' % name
            else:
                self._db.update_classifier_model(name, None)
        return 'Ok'

    @cherrypy.expose
    def duplicate(self, name, new_name, overwrite=False):
        name = str.strip(name)
        new_name = str.strip(new_name)
        if len(new_name) < 1:
            cherrypy.response.status = 400
            return 'Classifier name too short'
        with _lock_trainingset(self._db, new_name), _lock_model(self._db, new_name):
            if not self._db.classifier_exists(new_name):
                self._db.create_classifier(new_name, self._db.get_classifier_classes(name))
            elif not overwrite:
                cherrypy.response.status = 403
                return '%s is already in the collection' % name
            else:
                self._db.update_classifier_model(name, None)
        self._background_processor.put(_duplicate_model, (self._db_connection_string, name, new_name),
                                       description='duplicate model \'%s\' to \'%s\'' % (name, new_name))
        self._background_processor.put(_duplicate_trainingset, (self._db_connection_string, name, new_name),
                                       description='duplicate training set \'%s\' to \'%s\'' % (name, new_name))
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

        name = str.strip(name)
        X = numpy.asanyarray([x.strip() for x in X])
        y = numpy.asanyarray([label.strip() for label in y])

        if len(X) != len(y):
            cherrypy.response.status = 400
            return 'Must specify the same numbers of strings and labels'

        self._background_processor.put(_update_model, (self._db_connection_string, name, X, y),
                                       description='update model')
        self._background_processor.put(_update_trainingset, (self._db_connection_string, name, X, y),
                                       description='update training set')

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
        return self._db.get_classifier_classes(name)

    @cherrypy.expose
    def download(self, name):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        filename = 'training data %s %s.csv' % (name, str(self._db.get_classifier_last_update_time(name)))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(DOWNLOAD_DIR, filename)
        if not os.path.isfile(fullpath):
            header = dict()
            header['name'] = name
            header['classes'] = self._db.get_classifier_classes(name)
            try:
                with open(fullpath, 'w') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow([json.dumps([header])])
                    counter = 0
                    for classification in self._db.get_classifier_examples(name):
                        writer.writerow([counter, classification.document.text,
                                         '%s:%s' % (name, classification.label.name)])
                        counter += 1
            except:
                os.unlink(filename)

        return serve_file(fullpath, "text/csv", "attachment")

    @cherrypy.expose
    def upload(self, **data):
        try:
            file = data['file']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must upload a file'

        filename = 'examples %s.csv' % (uuid4())
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(UPLOAD_DIR, filename)
        with open(fullpath, 'wb') as outfile:
            shutil.copyfileobj(file.file, outfile)

        detector = UniversalDetector()
        with open(fullpath, 'rb') as file:
            for line in file:
                detector.feed(line)
                if detector.done:
                    break
        encoding = detector.result['encoding']
        cherrypy.log('Encode guessing for uploaded file ' + json.dumps(detector.result), severity=logging.INFO)

        classifiers_definition = defaultdict(set)
        with open(fullpath, 'r', encoding=encoding) as file:
            try:
                reader = csv.reader(file)
                header = json.loads(next(reader)[0])
                for classifier_definition in header:
                    classifier_name = classifier_definition['name']
                    classifiers_definition[classifier_name].add(classifier_definition['classes'])
            except Exception as e:
                cherrypy.log(
                    'No JSON header in uploaded file, scanning classes from the whole file. Exception:' + repr(e),
                    severity=logging.INFO)

        with open(fullpath, 'r', encoding=encoding) as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 0:
                    continue
                first = row[0].strip()
                if len(first) > 0 and first[0] == '#':
                    continue

                if len(row) < 3:
                    continue
                for classifier_label in row[2:]:
                    try:
                        classifier_name, label = classifier_label.split(':')
                    except:
                        continue
                    classifier_name = classifier_name.strip()
                    label = label.strip()
                    classifiers_definition[classifier_name].add(label)

        for classifier_name in classifiers_definition:
            classes = classifiers_definition[classifier_name]
            if not self._db.classifier_exists(classifier_name):
                if len(classes) < 2:
                    cherrypy.response.status = 400
                    return 'Must specify at least two classes for classifier \'%s\'' % classifier_name
                self._background_processor.put(_create_model,
                                               (self._db_connection_string, classifier_name, classes),
                                               description='create model \'%s\'' % classifier_name)
            else:
                if not len(set(self._db.get_classifier_classes(classifier_name)).intersection(classes)) == len(
                        classes):
                    cherrypy.response.status = 400
                    return 'Existing classifier \'%s\' uses a different set of classes than input file' % classifier_name

        for classifier_name in classifiers_definition:
            self._background_processor.put(_update_from_file,
                                           (_update_model, encoding, self._db_connection_string, fullpath,
                                            classifier_name),
                                           description='update model \'%s\' from file' % classifier_name)
            self._background_processor.put(_update_from_file,
                                           (_update_trainingset, encoding, self._db_connection_string, fullpath,
                                            classifier_name),
                                           description='update training set \'%s\' from file' % classifier_name)

        return 'Ok'

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
        return self._db.classify(name, X)

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
    def extract(self, **data):
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
        if type(classes) is str:
            classes = [classes]
        if type(classes) is not list:
            cherrypy.response.status = 400
            return 'Must specify at least a class'
        name = str.strip(name)
        if len(name) < 1:
            cherrypy.response.status = 400
            return 'Classifier name too short'
        classes = map(str.strip, classes)
        classes = list(set(classes))
        if len(classes) < 1:
            cherrypy.response.status = 400
            return 'Must specify at least a class'
        for class_name in classes:
            if len(class_name) < 1:
                cherrypy.response.status = 400
                return 'Class name too short'
        try:
            overwrite = data['overwrite']
        except KeyError:
            overwrite = False
        for class_name in classes:
            # TODO retrain or extract module?
            with _lock_trainingset(self._db, class_name), _lock_model(self._db, class_name):
                if not self._db.classifier_exists(class_name):
                    self._db.create_classifier(class_name, ['yes','no'])
                elif not overwrite:
                    cherrypy.response.status = 403
                    return '%s is already in the collection' % class_name
                else:
                    self._db.update_classifier_model(class_name, None)
        return 'Ok'


    @cherrypy.expose
    def compose(self, **data):
        # TODO compose binary classifiers into a multilabel classifier
        pass

    @cherrypy.expose
    def version(self):
        return "0.4.0 (db: %s)" % self._db.version()


@logged_call
def _update_trainingset(db_connection_string, name, X, y):
    with SQLAlchemyDB(db_connection_string) as db:
        with _lock_trainingset(db, name):
            for (content, label) in zip(X, y):
                db.create_training_example(name, content, label)


@logged_call
def _update_model(db_connection_string, name, X, y):
    with SQLAlchemyDB(db_connection_string) as db:
        with _lock_model(db, name):
            clf = db.get_classifier_model(name)
            if clf is None:
                clf = OnlineClassifier(name, db.get_classifier_classes(name), average=20)
            clf.partial_fit(X, y)
            db.update_classifier_model(name, clf)


@logged_call_with_args
def _update_from_file(update_function, encoding, db_connection_string, filename, classifier_name):
    with open(filename, encoding=encoding) as file:
        reader = csv.reader(file)
        next(reader)
        X = []
        y = []
        for row in reader:
            if len(row) < 3:
                continue
            text = row[1]
            classifiers_labels = row[2:]
            for classifier_label in classifiers_labels:
                try:
                    example_classifier_name, label = classifier_label.split(':')
                except:
                    continue
                example_classifier_name = example_classifier_name.strip()
                label = label.strip()
                if example_classifier_name is not None and example_classifier_name == classifier_name and \
                                label is not None and len(label) > 0:
                    X.append(text)
                    y.append(label)
                    break
            if len(X) >= MAX_BATCH_SIZE:
                update_function(db_connection_string, classifier_name, X, y)
                X = []
                y = []
        if len(X) > 0:
            update_function(db_connection_string, classifier_name, X, y)


@logged_call_with_args
def _create_model(db_connection_string, name, classes):
    with SQLAlchemyDB(db_connection_string) as db:
        with _lock_trainingset(db, name), _lock_model(db, name):
            if not db.classifier_exists(name):
                clf = OnlineClassifier(name, classes, average=20)
                db.create_classifier(name, classes, clf)


@logged_call_with_args
def _duplicate_model(db_connection_string, name, new_name):
    with SQLAlchemyDB(db_connection_string) as db:
        with _lock_model(db, new_name):
            if not db.classifier_exists(new_name):
                clf = db.get_classifier_model(name)
                clf.name = new_name
                classes = db.get_classifier_classes(name)
                db.create_classifier(new_name, classes, clf)
            elif db.get_classifier_model(new_name) is None:
                clf = db.get_classifier_model(name)
                clf.name = new_name
                db.update_classifier_model(new_name, clf)


@logged_call_with_args
def _duplicate_trainingset(db_connection_string, name, new_name):
    with SQLAlchemyDB(db_connection_string) as db:
        with _lock_trainingset(db, new_name):
            batchsize = 1000
            block = 0
            batch = list()
            added = 1
            while added > 0:
                added = 0
                for example in db.get_classifier_examples(name, block * batchsize, batchsize):
                    batch.append((example.document.text, example.label.name))
                    added += 1
                for text, label in batch:
                    db.create_training_example(new_name, text, label)
                block += 1


def _lock_model(db, name):
    return DBLock(db, '%s %s' % (name, 'model'))


def _lock_trainingset(db, name):
    return DBLock(db, '%s %s' % (name, 'trainingset'))


if __name__ == "__main__":
    with WebClassifierCollection('sqlite:///%s' % 'test.db', BackgroundProcessor('sqlite:///%s' % 'test.db')) as wcc:
        cherrypy.quickstart(wcc, '/service')
