import csv
import json
import logging
import os
import pickle
import random
import shutil
from collections import defaultdict
from uuid import uuid4

import cherrypy
import numpy
from cherrypy.lib.static import serve_file

from db import sqlalchemydb
from db.sqlalchemydb import SQLAlchemyDB, DBLock
from util.util import get_fully_portable_file_name

__author__ = 'Andrea Esuli'

MAX_BATCH_SIZE = 1000

YES_LABEL = 'yes'
NO_LABEL = 'no'
BINARY_LABELS = [YES_LABEL, NO_LABEL]
CSV_LARGE_FIELD = 1024 * 1024 * 10


class ClassifierCollectionService(object):
    def __init__(self, db_connection_string, data_dir):
        self._db_connection_string = db_connection_string
        self._db = SQLAlchemyDB(db_connection_string)
        self._download_dir = os.path.join(data_dir, 'downloads')
        self._upload_dir = os.path.join(data_dir, 'uploads')

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def info(self, page=0, page_size=50):
        result = []
        names = self._db.classifier_names()[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
        for name in names:
            classifier_info = dict()
            classifier_info['name'] = name
            classifier_info['type'] = self._db.get_classifier_type(name)
            classifier_info['labels'] = self._db.get_classifier_labels(name)
            classifier_info['description'] = self._db.get_classifier_description(name)
            classifier_info['created'] = str(self._db.get_classifier_creation_time(name))
            classifier_info['updated'] = str(self._db.get_classifier_last_update_time(name))
            classifier_info['size'] = self._db.get_classifier_examples_count(name, True)
            result.append(classifier_info)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def count(self):
        return str(len(list(self._db.classifier_names())))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classifier_types(self):
        return sqlalchemydb._CLASSIFIER_TYPES[:-1]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def create(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            classifier_type = data['type']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a classifier type'
        if not classifier_type in sqlalchemydb._CLASSIFIER_TYPES[:-1]:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(sqlalchemydb._CLASSIFIER_TYPES[:-1])

        try:
            labels = data['labels']
        except KeyError:
            try:
                labels = data['labels[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify the labels'
        if not (type(labels) is list or type(labels) is set):
            cherrypy.response.status = 400
            return 'Must specify at least two labels'
        name = str.strip(name)
        if len(name) < 1:
            cherrypy.response.status = 400
            return 'Classifier name too short'
        labels = map(str.strip, labels)
        labels = list(set(labels))
        if len(labels) < 2:
            cherrypy.response.status = 400
            return 'Must specify at least two labels'
        for label in labels:
            if len(label) < 1:
                cherrypy.response.status = 400
                return 'Label name too short'
        try:
            overwrite = data['overwrite']
            if overwrite == 'false' or overwrite == 'False':
                overwrite = False
        except KeyError:
            overwrite = False

        with _lock_trainingset(self._db, name), _lock_model(self._db, name):
            if not self._db.classifier_exists(name):
                self._db.create_classifier(name, labels, classifier_type)
            elif not overwrite:
                cherrypy.response.status = 403
                return '%s is already in the collection' % name
            else:
                self.delete(name)
                self._db.create_classifier(name, labels, classifier_type)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def duplicate(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            new_name = data['new_name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a new name'
        try:
            classifier_type = data['type']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a classifier type'
        if not classifier_type in sqlalchemydb._CLASSIFIER_TYPES[:-1]:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(sqlalchemydb._CLASSIFIER_TYPES[:-1])
        try:
            overwrite = data['overwrite']
            if overwrite == 'false' or overwrite == 'False':
                overwrite = False
        except KeyError:
            overwrite = False

        name = str.strip(name)
        new_name = str.strip(new_name)
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exist' % name
        if len(new_name) < 1:
            cherrypy.response.status = 400
            return 'Classifier name too short'
        if overwrite == 'false' or overwrite == 'False':
            overwrite = False
        with _lock_trainingset(self._db, new_name), _lock_model(self._db, new_name):
            if not self._db.classifier_exists(new_name):
                self._db.create_classifier(new_name, self._db.get_classifier_labels(name), classifier_type)
            elif not overwrite:
                cherrypy.response.status = 403
                return '%s is already in the collection' % new_name
            else:
                self.delete(new_name)
                self._db.create_classifier(new_name, self._db.get_classifier_labels(name), classifier_type)
        job_id_model = self._db.create_job(_duplicate_model, (self._db_connection_string, name, new_name),
                                           description='duplicate model \'%s\' to \'%s\'' % (name, new_name))
        job_id_training = self._db.create_job(_duplicate_trainingset, (self._db_connection_string, name, new_name),
                                              description='duplicate training set \'%s\' to \'%s\'' % (name, new_name))
        return [job_id_model, job_id_training]

    @cherrypy.expose
    @cherrypy.tools.json_out()
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
                return 'Must specify a vector of assigned labels (y)'

        try:
            synchro = data['synchro']
            if synchro == 'false' or synchro == 'False':
                synchro = False
        except KeyError:
            synchro = False

        X = numpy.atleast_1d(X)
        y = numpy.atleast_1d(y)

        name = str.strip(name)
        X = numpy.asanyarray([x.strip() for x in X])
        y = numpy.asanyarray([label.strip() for label in y])

        if len(X) != len(y):
            cherrypy.response.status = 400
            return 'Must specify the same numbers of strings and labels'

        if synchro:
            _update_trainingset(self._db_connection_string, name, X, y)
            _update_model(self._db_connection_string, name, X, y)
            return []
        else:
            job_id_model = self._db.create_job(_update_model, (self._db_connection_string, name, X, y),
                                               description='update model')
            job_id_training = self._db.create_job(_update_trainingset, (self._db_connection_string, name, X, y),
                                                  description='update training set')

            return [job_id_model, job_id_training]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def hide(self, **data):
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

        name = str.strip(name)
        X = numpy.asanyarray([x.strip() for x in X])
        for x in X:
            self._db.mark_classifier_text_as_hidden(name, x)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def set_description(self, name, description):
        if description is not None:
            self._db.set_classifier_description(name, description)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def rename(self, name, new_name, overwrite=False):
        if overwrite == 'false' or overwrite == 'False':
            overwrite = False
        try:
            self._db.rename_classifier(name, new_name, overwrite)
        except KeyError:
            cherrypy.response.status = 404
            return '%s does not exist' % name
        except Exception as e:
            cherrypy.response.status = 500
            return 'Error (%s)' % str(e)
        else:
            return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def delete(self, name):
        try:
            self._db.delete_classifier(name)
        except KeyError as e:
            cherrypy.response.status = 404
            return '%s does not exits' % name
        else:
            return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def label_info(self, name):
        return self._db.get_classifier_labels(name)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def label_rename(self, name, label_name, new_label_name):
        try:
            self._db.rename_classifier_label(name, label_name, new_label_name)
        except KeyError:
            cherrypy.response.status = 404
            return '%s does not exits in %s' % (label_name, name)
        except Exception as e:
            cherrypy.response.status = 500
            return 'Error (%s)' % str(e)
        else:
            return 'Ok'

    @cherrypy.expose
    def download_training_data(self, name):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        filename = 'training data %s %s.csv' % (name, str(self._db.get_classifier_last_update_time(name)))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)
        if not os.path.isfile(fullpath):
            header = list()
            header.append('#id')
            header.append('text')
            header.append(name + ' = (' + ', '.join(self._db.get_classifier_labels(name)) + ')')
            try:
                with open(fullpath, 'w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow(header)
                    for i, classification in enumerate(self._db.get_classifier_examples(name, False)):
                        writer.writerow([i, classification.document.text,
                                         '%s:%s' % (name, classification.label.name)])
            except:
                os.unlink(fullpath)

        return serve_file(fullpath, "text/csv", "attachment")

    @cherrypy.expose
    def download_model(self, name):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        filename = 'model %s %s.modeldata' % (name, str(self._db.get_classifier_last_update_time(name)))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)
        if not os.path.isfile(fullpath):
            try:
                with open(fullpath, 'wb') as file:
                    model = self._db.get_classifier_model(name)
                    pickle.dump(model, file)
            except:
                os.unlink(fullpath)

        return serve_file(fullpath, "application/x-download", "attachment")

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def upload_training_data(self, **data):
        try:
            file = data['file']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must upload a file'
        try:
            classifier_type = data['type']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a classifier type'
        if not classifier_type in sqlalchemydb._CLASSIFIER_TYPES[:-1]:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(sqlalchemydb._CLASSIFIER_TYPES[:-1])

        filename = 'examples %s.csv' % (uuid4())
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._upload_dir, filename)
        with open(fullpath, 'wb') as outfile:
            shutil.copyfileobj(file.file, outfile)

        if csv.field_size_limit() < CSV_LARGE_FIELD:
            csv.field_size_limit(CSV_LARGE_FIELD)

        classifiers_definition = defaultdict(set)

        with open(fullpath, 'r', encoding='utf-8', errors='ignore') as file:
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

        jobs = list()

        for classifier_name in classifiers_definition:
            labels = classifiers_definition[classifier_name]
            if not self._db.classifier_exists(classifier_name):
                if len(labels) < 2:
                    cherrypy.response.status = 400
                    return 'Must specify at least two labels for classifier \'%s\'' % classifier_name
                self.create(**{'name': classifier_name, 'labels': labels, 'type': classifier_type})
            else:
                if not len(set(self._db.get_classifier_labels(classifier_name)).intersection(labels)) == len(
                        labels):
                    cherrypy.response.status = 400
                    return 'Existing classifier \'%s\' uses a different set of labels than input file' % classifier_name

        for classifier_name in classifiers_definition:
            jobs.append(self._db.create_job(_update_from_file,
                                            (_update_model, 'utf-8', self._db_connection_string, fullpath,
                                             classifier_name),
                                            description='update model \'%s\' from file' % classifier_name))
            jobs.append(self._db.create_job(_update_from_file,
                                            (_update_trainingset, 'utf-8', self._db_connection_string, fullpath,
                                             classifier_name),
                                            description='update training set \'%s\' from file' % classifier_name))

        return jobs

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def upload_model(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must provide a name'

        try:
            file = data['file']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must upload a file'

        try:
            overwrite = data['overwrite']
            if overwrite == 'false' or overwrite == 'False':
                overwrite = False
        except KeyError:
            overwrite = False

        filename = 'model %s %s.pickle' % (name, uuid4())
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._upload_dir, filename)
        with open(fullpath, 'wb') as outfile:
            shutil.copyfileobj(file.file, outfile)

        with _lock_trainingset(self._db, name), _lock_model(self._db, name):
            if self._db.classifier_exists(name):
                if overwrite:
                    self._db.delete_classifier(name)
                else:
                    cherrypy.response.status = 403
                    return 'A classifier with name %s is already in the collection' % name

            with open(fullpath, 'rb') as infile:
                model = pickle.load(infile)
                labels = list(model.labels())
                classifier_type = sqlalchemydb.get_classifier_type_from_model(model)
                self._db.create_classifier(name, labels, classifier_type)
                self._db.update_classifier_model(name, model)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classify(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        if len(name.strip()) == 0:
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
        cherrypy.log('ClassifierCollectionService.classify(name="' + name + '", X="' + str(X) + '")')
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
        cherrypy.log('ClassifierCollectionService.score(name="' + name + '", X="' + str(X) + '")')
        X = numpy.atleast_1d(X)
        return self._db.score(name, X)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def extract(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            classifier_type = data['type']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a classifier type'
        if not classifier_type in sqlalchemydb._CLASSIFIER_TYPES[:-1]:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(sqlalchemydb._CLASSIFIER_TYPES[:-1])
        try:
            labels = data['labels']
        except KeyError:
            try:
                labels = data['labels[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify the labels'
        if type(labels) is str:
            labels = [labels]
        if type(labels) is not list:
            cherrypy.response.status = 400
            return 'Must specify at least a label'
        name = str.strip(name)
        if len(name) < 1:
            cherrypy.response.status = 400
            return 'Classifier name too short'
        labels = map(str.strip, labels)
        labels = list(set(labels))
        if len(labels) < 1:
            cherrypy.response.status = 400
            return 'Must specify at least a label'
        for label in labels:
            if len(label) < 1:
                cherrypy.response.status = 400
                return 'Label name too short'
        try:
            overwrite = data['overwrite']
            if overwrite == 'false' or overwrite == 'False':
                overwrite = False
        except KeyError:
            overwrite = False
        for label in labels:
            with _lock_trainingset(self._db, label), _lock_model(self._db, label):
                if self._db.classifier_exists(label) and not overwrite:
                    cherrypy.response.status = 403
                    return 'A classifier with name %s is already in the collection' % label

        jobs = list()
        for label in labels:
            with _lock_trainingset(self._db, label), _lock_model(self._db, label):
                if not self._db.classifier_exists(label):
                    self._db.create_classifier(label, BINARY_LABELS, classifier_type)
                elif not overwrite:
                    cherrypy.response.status = 403
                    return 'A classifier with name %s is already in the collection' % label
                else:
                    self.delete(label)
                    self._db.create_classifier(label, BINARY_LABELS, classifier_type)
                jobs.append(self._db.create_job(_extract_binary_trainingset, (self._db_connection_string, name, label),
                                                description='extract binary classifier from \'%s\' to \'%s\'' % (
                                                    name, label)))
        return jobs

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def combine(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name for the new classifier'
        try:
            classifier_type = data['type']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a classifier type'
        if not classifier_type in sqlalchemydb._CLASSIFIER_TYPES[:-1]:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(sqlalchemydb._CLASSIFIER_TYPES[:-1])
        try:
            sources = data['sources']
        except KeyError:
            try:
                sources = data['sources[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify the source binary classifiers'
        if type(sources) is str:
            sources = [sources]
        if type(sources) is not list:
            cherrypy.response.status = 400
            return 'Unknown type for the sources'
        name = str.strip(name)
        if len(name) < 1:
            cherrypy.response.status = 400
            return 'New classifier name is too short'
        sources = map(str.strip, sources)
        sources = list(set(sources))
        if len(sources) < 2:
            cherrypy.response.status = 400
            return 'Must specify at least two classifiers'
        for source_name in sources:
            if len(source_name) < 1:
                cherrypy.response.status = 400
                return 'Source name too short'
        try:
            overwrite = data['overwrite']
            if overwrite == 'false' or overwrite == 'False':
                overwrite = False
        except KeyError:
            overwrite = False

        labels = set()
        for source_name in sources:
            if set(self._db.get_classifier_labels(source_name)) == set(BINARY_LABELS):
                labels.add(source_name)
            else:
                labels.update(self._db.get_classifier_labels(source_name))

        labels = list(labels)

        with _lock_trainingset(self._db, name), _lock_model(self._db, name):
            if not self._db.classifier_exists(name):
                self._db.create_classifier(name, labels, classifier_type)
            elif not overwrite:
                cherrypy.response.status = 403
                return '%s is already in the collection' % name
            else:
                self.delete(name)
                self._db.create_classifier(name, labels, classifier_type)
            job_id = self._db.create_job(_combine_classifiers, (self._db_connection_string, name, sources),
                                         description='combining classifiers from \'%s\' to \'%s\'' % (
                                             ', '.join(sources), name))
        return [job_id]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def version(self):
        return "3.1.1 (db: %s)" % self._db.version()


def _update_trainingset(db_connection_string, name, X, y):
    cherrypy.log('ClassifierCollectionService._update_trainingset(name="' + name + '", len(X)="' + str(len(X)) + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        with _lock_trainingset(db, name):
            db.create_training_examples(name, list(zip(X, y)))


def _update_model(db_connection_string, name, X, y):
    if len(X) > 0:
        cherrypy.log('ClassifierCollectionService._update_model(name="' + name + '", len(X)="' + str(len(X)) + '")')
        with SQLAlchemyDB(db_connection_string) as db:
            with _lock_model(db, name):
                model = db.get_classifier_model(name)
                model.partial_fit(X, y)
                db.update_classifier_model(name, model)


def _update_from_file(update_function, encoding, db_connection_string, filename, classifier_name):
    cherrypy.log(
        'ClassifierCollectionService._update_from_file(filename="' + filename + '", classifier_name="' + classifier_name + '")')
    if csv.field_size_limit() < CSV_LARGE_FIELD:
        csv.field_size_limit(CSV_LARGE_FIELD)
    with open(filename, encoding=encoding, errors='ignore') as file:
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


def _duplicate_model(db_connection_string, name, new_name):
    cherrypy.log('ClassifierCollectionService._duplicate_model(name="' + name + '", new_name="' + new_name + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        source_type = db.get_classifier_type(name)
        new_type = db.get_classifier_type(new_name)
        if source_type == new_type:
            with _lock_model(db, new_name):
                model = db.get_classifier_model(name)
                model.rename(new_name)
                db.update_classifier_model(new_name, model)
        else:
            block = 0
            batchsize = MAX_BATCH_SIZE

            added = True
            while added:
                batchX = list()
                batchy = list()
                added = False
                example_numerator = db.get_classifier_examples(name, False, block * batchsize, batchsize)
                for example in example_numerator:
                    batchX.append(example.document.text)
                    batchy.append(example.label.name)
                    added = True
                block += 1
                if len(batchX) > 0:
                    pairs = list(zip(batchX, batchy))
                    random.shuffle(pairs)
                    batchX, batchy = zip(*pairs)
                    _update_model(db_connection_string, new_name, batchX, batchy)


def _duplicate_trainingset(db_connection_string, name, new_name):
    cherrypy.log('ClassifierCollectionService._duplicate_trainingset(name="' + name + '", new_name="' + new_name + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        batchsize = MAX_BATCH_SIZE
        block = 0
        added = True
        while added:
            batch = list()
            added = False
            for example in db.get_classifier_examples(name, True, block * batchsize, batchsize):
                batch.append((example.document.text, example.label.name))
                added = True
            if len(batch) > 0:
                with _lock_trainingset(db, new_name):
                    db.create_training_examples(new_name, batch)
            block += 1


def _extract_binary_trainingset(db_connection_string, classifier, label_to_extract):
    cherrypy.log(
        'ClassifierCollectionService._extract_binary_trainingset(classifier="' + classifier + '", label="' + label_to_extract + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        batchsize = MAX_BATCH_SIZE
        block = 0
        added = True
        while added:
            batch = list()
            added = False
            for example in db.get_classifier_examples(classifier, False, block * batchsize, batchsize):
                if example.label.name == label_to_extract:
                    label = YES_LABEL
                else:
                    label = NO_LABEL
                batch.append((example.document.text, label))

                added = True
            if len(batch) > 0:
                with _lock_trainingset(db, label_to_extract):
                    db.create_training_examples(label_to_extract, batch)
                batchX, batchy = list(zip(*batch))
                _update_model(db_connection_string, label_to_extract, batchX, batchy)
            block += 1


def _combine_classifiers(db_connection_string, name, sources):
    cherrypy.log(
        'ClassifierCollectionService._combine_classifiers(name="' + name + '", sources="' + str(sources) + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        binary_sources = set()
        for source_name in sources:
            if set(db.get_classifier_labels(source_name)) == set(BINARY_LABELS):
                binary_sources.add(source_name)
        sizes = list()
        for source in sources:
            if source in binary_sources:
                sizes.append(db.get_classifier_examples_with_label_count(source, YES_LABEL))
            else:
                sizes.append(db.get_classifier_examples_count(source, False))
        max_size = max(sizes)
        paddings = list()
        for size in sizes:
            paddings.append(size - max_size)

        batchsize = MAX_BATCH_SIZE // len(sources)
        added = True
        while added:
            added = False
            batchX = list()
            batchy = list()
            for i, source in enumerate(sources):
                if paddings[i] < 0:
                    paddings[i] += batchsize
                    paddings[i] = min(paddings[i], 0)
                    continue
                if source in binary_sources:
                    example_numerator = db.get_classifier_examples_with_label(source, YES_LABEL, paddings[i], batchsize)
                    for example in example_numerator:
                        batchX.append(example.document.text)
                        batchy.append(source)
                        added = True
                else:
                    example_numerator = db.get_classifier_examples(source, False, paddings[i], batchsize)
                    for example in example_numerator:
                        batchX.append(example.document.text)
                        batchy.append(example.label.name)
                        added = True
                paddings[i] += batchsize
            if len(batchX)>0:
                pairs = list(zip(batchX,batchy))
                random.shuffle(pairs)
                batchX, batchy = zip(*pairs)

                _update_trainingset(db_connection_string, name, batchX, batchy)
                _update_model(db_connection_string, name, batchX, batchy)


def _lock_model(db, name):
    return DBLock(db, '%s %s' % (name, 'model'))


def _lock_trainingset(db, name):
    return DBLock(db, '%s %s' % (name, 'trainingset'))
