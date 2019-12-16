import csv
import os
import pickle
import random
import shutil
from collections import defaultdict
from uuid import uuid4

import cherrypy
import numpy
from cherrypy.lib.static import serve_file

from classifier.classifier import SINGLE_LABEL, MULTI_LABEL
from db import sqlalchemydb
from db.sqlalchemydb import SQLAlchemyDB, DBLock, _CLASSIFIER_LABEL_SEP
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
    def info(self, page=None, page_size=50):
        result = []
        if page is None:
            names = self._db.classifier_names()
        else:
            names = self._db.classifier_names()[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
        for name in names:
            classifier_info = dict()
            classifier_info['name'] = name
            classifier_info['type'] = self._db.get_classifier_type(name)
            classifier_info['labels'] = self._db.get_classifier_labels(name)
            classifier_info['description'] = self._db.get_classifier_description(name)
            classifier_info['created'] = str(self._db.get_classifier_creation_time(name))
            classifier_info['updated'] = str(self._db.get_classifier_last_update_time(name))
            classifier_info['size'] = self._db.get_classifier_examples_count(name)
            result.append(classifier_info)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def count(self):
        return str(len(list(self._db.classifier_names())))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classifier_types(self):
        return sqlalchemydb.CLASSIFIER_TYPES

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
        if not classifier_type in sqlalchemydb.CLASSIFIER_TYPES:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(sqlalchemydb.CLASSIFIER_TYPES)

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

        if isinstance(X, str):
            X = [X]

        if isinstance(y, str):
            y = [y]

        if len(X) != len(y):
            cherrypy.response.status = 400
            return 'Must specify the same numbers of strings and labels'

        if synchro:
            _update_trainingset(self._db, name, X, y)
            _update_model(self._db, name, X, y)
            return []
        else:
            job_id_model = self._db.create_job(_update_model, (self._db_connection_string, name, X, y),
                                               description='update model')
            job_id_training = self._db.create_job(_update_trainingset, (self._db_connection_string, name, X, y),
                                                  description='update training set')

            return [job_id_model, job_id_training]

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
    def classifier_type(self, name):
        return self._db.get_classifier_type(name)

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
    @cherrypy.tools.json_out()
    def get_training_data_size(self, name):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        return str(self._db.get_classifier_examples_count(name))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_training_data(self, name, page=None, page_size=50, filter=None):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        page_size = int(page_size)
        if page is None:
            offset = 0
        else:
            offset = int(page) * page_size
        limit = page_size
        batch = list()
        classifier_type = self._db.get_classifier_type(name)
        if classifier_type == SINGLE_LABEL:
            for classification in self._db.get_classifier_examples(name, offset, limit, filter):
                batch.append({'id': classification.id, 'update': str(classification.last_updated),
                              'text': classification.document.text,
                              'label': '%s:%s' % (name, classification.label.name)})
        elif classifier_type == MULTI_LABEL:
            for classification in self._db.get_classifier_examples(name, offset, limit, filter):
                batch.append({'id': classification.id, 'doc_id': classification.document_id,
                              'update': str(classification.last_updated),
                              'text': classification.document.text})
            for doc in batch:
                doc['labels'] = [name + ':' + label for label in
                                 self._db.get_label_from_training_id(name, doc['doc_id'])]
                del doc['doc_id']
        return batch

    @cherrypy.expose
    def delete_classifier_example(self, name, id):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        self._db.delete_classifier_example(name, id)

    @cherrypy.expose
    def download_training_data(self, name):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        filename = 'training data %s %s.csv' % (name, str(self._db.get_classifier_last_update_time(name)))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)
        if not os.path.isfile(fullpath):
            classifier_type = self._db.get_classifier_type(name)
            header = list()
            header.append('#id')
            header.append('text')
            header.append(name + ' = (' + ', '.join(self._db.get_classifier_labels(name)) + ')')
            try:
                with open(fullpath, 'w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow(header)
                    if classifier_type == SINGLE_LABEL:
                        for i, classification in enumerate(self._db.get_classifier_examples(name)):
                            writer.writerow([i, classification.document.text,
                                             '%s:%s' % (name, classification.label.name)])
                    elif classifier_type == MULTI_LABEL:
                        added = True
                        block_count = 0
                        block_size = MAX_BATCH_SIZE
                        while added:
                            offset = block_count * block_size
                            docs = [(i + offset, classification.document.id, classification.document.text) for
                                    i, classification in
                                    enumerate(
                                        self._db.get_classifier_examples(name, offset, block_size))]
                            for i, id, text in docs:
                                label_assignment = [name + ':' + label for label in
                                                    self._db.get_label_from_training_id(name, id)]
                                writer.writerow([i, text] + label_assignment)
                            block_count += 1
                            added = len(docs) > 0
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

        filename = 'examples %s.csv' % (uuid4())
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._upload_dir, filename)
        with open(fullpath, 'wb') as outfile:
            shutil.copyfileobj(file.file, outfile)

        if csv.field_size_limit() < CSV_LARGE_FIELD:
            csv.field_size_limit(CSV_LARGE_FIELD)

        classifiers_definition = defaultdict(set)
        classifiers_type = dict()

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
                    split_values = classifier_label.split(':')
                    if len(split_values) == 2:
                        classifier_name, label = split_values
                        classifier_name = classifier_name.strip()
                        label = label.strip()
                        classifier_type = classifiers_type.get(classifier_name, None)
                        if classifier_type:
                            if classifier_type != SINGLE_LABEL:
                                cherrypy.response.status = 400
                                return 'Inconsistent single-label and multi-label labeling for classifier \'%s\'' % classifier_name
                        else:
                            classifiers_type[classifier_name] = SINGLE_LABEL
                    elif len(split_values) == 3:
                        classifier_name, label, value = split_values
                        classifier_name = classifier_name.strip()
                        label = label.strip()
                        value = value.strip()
                        classifier_type = classifiers_type.get(classifier_name, None)
                        if classifier_type:
                            if classifier_type != MULTI_LABEL:
                                cherrypy.response.status = 400
                                return 'Inconsistent single-label and multi-label labeling for classifier \'%s\'' % classifier_name
                        else:
                            classifiers_type[classifier_name] = MULTI_LABEL
                        if value != NO_LABEL and value != YES_LABEL:
                            cherrypy.response.status = 400
                            return 'Unknown label value \'%s\'' % value
                    classifiers_definition[classifier_name].add(label)

        jobs = list()

        for classifier_name in classifiers_definition:
            labels = classifiers_definition[classifier_name]
            if not self._db.classifier_exists(classifier_name):
                if len(labels) < 2 and classifiers_type[classifier_name] == SINGLE_LABEL:
                    cherrypy.response.status = 400
                    return 'Must specify at least two labels for classifier \'%s\'' % classifier_name
                self.create(
                    **{'name': classifier_name, 'labels': labels, 'type': classifiers_type[classifier_name]})
            else:
                if not len(set(self._db.get_classifier_labels(classifier_name)).intersection(labels)) == len(
                        labels):
                    cherrypy.response.status = 400
                    return 'Existing classifier \'%s\' uses a different set of labels than input file' % classifier_name
                classifier_type = self._db.get_classifier_type(classifier_name)
                if classifier_type != classifiers_type[classifier_name]:
                    cherrypy.response.status = 400
                    return 'Existing classifier \'%s\' uses a different labeling model: \'%s\'' % (
                        classifier_name, classifier_type)

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
        if isinstance(X, str):
            X = [X]

        mark_human_labels = False
        requester = cherrypy.request.login
        if requester is not None:
            mark_human_labels = True

        return self._db.classify(name, X, mark_human_labels=mark_human_labels)

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
        new_name = dict()
        for label in labels:
            new_name[label] = name + '_' + label
            with _lock_trainingset(self._db, new_name[label]), _lock_model(self._db, new_name[label]):
                if self._db.classifier_exists(new_name[label]) and not overwrite:
                    cherrypy.response.status = 403
                    return 'A classifier with name %s is already in the collection' % new_name[label]

        jobs = list()
        for label in labels:
            with _lock_trainingset(self._db, new_name[label]), _lock_model(self._db, new_name[label]):
                if not self._db.classifier_exists(new_name[label]):
                    self._db.create_classifier(new_name[label], BINARY_LABELS, SINGLE_LABEL)
                elif not overwrite:
                    cherrypy.response.status = 403
                    return 'A classifier with name %s is already in the collection' % new_name[label]
                else:
                    self.delete(label)
                    self._db.create_classifier(new_name[label], BINARY_LABELS, SINGLE_LABEL)
                jobs.append(self._db.create_job(_extract_binary_trainingset,
                                                (self._db_connection_string, name, label, new_name[label]),
                                                description='extract binary classifier \'%s\'' %
                                                            new_name[label]))
        return jobs

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def merge(self, **data):
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
        if not classifier_type in sqlalchemydb.CLASSIFIER_TYPES:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(sqlalchemydb.CLASSIFIER_TYPES)
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
        if len(sources) < 1:
            cherrypy.response.status = 400
            return 'Must specify at least a source classifier'
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

        try:
            binary_by_name = data['binary_by_name']
            if binary_by_name == 'false' or binary_by_name == 'False':
                binary_by_name = False
        except KeyError:
            binary_by_name = False

        labels = set()
        for source_name in sources:
            if set(self._db.get_classifier_labels(source_name)) == set(BINARY_LABELS) and binary_by_name:
                labels.add(source_name)
            else:
                labels.update(self._db.get_classifier_labels(source_name))

        labels = list(labels)
        if len(labels) < 2:
            cherrypy.response.status = 400
            return f'This merge operation would produce a classifier with a single label ("{list(labels)[0]}"), while two at least are required.'

        with _lock_trainingset(self._db, name), _lock_model(self._db, name):
            if not self._db.classifier_exists(name):
                self._db.create_classifier(name, labels, classifier_type)
            elif not overwrite:
                cherrypy.response.status = 403
                return '%s is already in the collection' % name
            else:
                self.delete(name)
                self._db.create_classifier(name, labels, classifier_type)
            model_merged = False
            if len(sources) == 1:
                source_type = self._db.get_classifier_type(sources[0])
                new_type = self._db.get_classifier_type(name)
                if source_type == new_type:
                    model_merged = True
                    model = self._db.get_classifier_model(sources[0])
                    model.rename(name)
                    self._db.update_classifier_model(name, model)

            jobs = list()
            jobs.append(self._db.create_job(_merge,
                                            (self._db_connection_string, _update_trainingset, name, sources,
                                             binary_by_name),
                                            description=f'merging classifiers from {sources} to "{name}" (binary_by_name={binary_by_name})'))
            if not model_merged:
                jobs.append(self._db.create_job(_merge,
                                                (self._db_connection_string, _update_model, name, sources,
                                                 binary_by_name),
                                                description=f'merging models from {sources} to "{name}" (binary_by_name={binary_by_name})'))
        return jobs

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def version(self):
        return "5.1.1 (db: %s)" % self._db.version()


def _update_trainingset(db_connection, name, X, y):
    cherrypy.log(
        'ClassifierCollectionService._update_trainingset(name="' + name + '", len(X)="' + str(len(X)) + '")')
    if type(db_connection) == str:
        with SQLAlchemyDB(db_connection) as db:
            with _lock_trainingset(db, name):
                db.create_training_examples(name, list(zip(X, y)))
    else:
        with _lock_trainingset(db_connection, name):
            db_connection.create_training_examples(name, list(zip(X, y)))


def _update_model(db_connection, name, X, y):
    if len(X) > 0:
        cherrypy.log(
            'ClassifierCollectionService._update_model(name="' + name + '", len(X)="' + str(len(X)) + '")')
        if type(db_connection) == str:
            with SQLAlchemyDB(db_connection) as db:
                with _lock_model(db, name):
                    model = db.get_classifier_model(name)
                    model.partial_fit(X, y)
                    db.update_classifier_model(name, model)
        else:
            with _lock_model(db_connection, name):
                model = db_connection.get_classifier_model(name)
                model.partial_fit(X, y)
                db_connection.update_classifier_model(name, model)


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
                    example_classifier_name, label = classifier_label.split(':', maxsplit=1)
                except:
                    continue
                example_classifier_name = example_classifier_name.strip()
                label = label.strip()
                if example_classifier_name is not None and example_classifier_name == classifier_name and \
                        label is not None and len(label) > 0:
                    X.append(text)
                    y.append(label)
            if len(X) >= MAX_BATCH_SIZE:
                update_function(db_connection_string, classifier_name, X, y)
                X = []
                y = []
        if len(X) > 0:
            update_function(db_connection_string, classifier_name, X, y)


def _extract_binary_trainingset(db_connection_string, classifier, label_to_extract, new_name):
    cherrypy.log(
        'ClassifierCollectionService._extract_binary_trainingset(classifier="' + classifier + '", label="' + label_to_extract + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        source_classifier_type = db.get_classifier_type(classifier)
        batchsize = MAX_BATCH_SIZE
        block = 0
        added = True
        while added:
            batch = list()
            added = False
            if source_classifier_type == SINGLE_LABEL:
                for example in db.get_classifier_examples(classifier, block * batchsize, batchsize):
                    if example.label.name == label_to_extract:
                        label = YES_LABEL
                    else:
                        label = NO_LABEL
                    batch.append((example.document.text, label))
                    added = True
            elif source_classifier_type == MULTI_LABEL:
                offset = block * batchsize
                docs = [classification.document.text for classification in
                        db.get_classifier_examples(classifier, offset, batchsize)]
                for text in docs:
                    known_labels_assignment = db.get_label(classifier, text)
                    for known_label in known_labels_assignment:
                        split = known_label.split(':')
                        if split[0] == label_to_extract:
                            batch.append((text, split[1]))
                added = len(docs) > 0
            if len(batch) > 0:
                with _lock_trainingset(db, new_name):
                    db.create_training_examples(new_name, batch)
                batchX, batchy = list(zip(*batch))
                _update_model(db, new_name, batchX, batchy)
            block += 1


def _merge(db_connection_string, merge_function, name, sources, binary_by_name):
    sources = list(sources)
    cherrypy.log(
        f'ClassifierCollectionService._merge_classifiers(merge_function={merge_function}, name="{name}", sources={sources}, binary_by_name={binary_by_name})')
    with SQLAlchemyDB(db_connection_string) as db:
        source_types = dict()
        source_to_label = dict()
        for source in sources:
            source_types[source] = db.get_classifier_type(source)
            source_to_label[source] = source
        target_type = db.get_classifier_type(name)

        binary_sources = set()
        multi_binary_sources = set()
        for source in sources:
            if source_types[source] == SINGLE_LABEL and set(db.get_classifier_labels(source)) == set(
                    BINARY_LABELS) and binary_by_name:
                binary_sources.add(source)
            if source_types[source] == MULTI_LABEL:
                multi_binary_sources.add(source)
        sizes = list()
        for multi_binary_source in multi_binary_sources:
            del source_to_label[source]
            sources.remove(multi_binary_source)
            for label in db.get_classifier_labels(multi_binary_source):
                sources.append(multi_binary_source + _CLASSIFIER_LABEL_SEP + label)
                binary_sources.add(multi_binary_source + _CLASSIFIER_LABEL_SEP + label)
                source_to_label[multi_binary_source + _CLASSIFIER_LABEL_SEP + label] = label

        for source in sources:
            if source in binary_sources:
                sizes.append(db.get_classifier_examples_with_label_count(source, YES_LABEL))
            else:
                sizes.append(db.get_classifier_examples_count(source))
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
                    if target_type == SINGLE_LABEL:
                        example_numerator = db.get_classifier_examples_with_label(source, YES_LABEL,
                                                                                  paddings[i],
                                                                                  batchsize)
                        for example in example_numerator:
                            batchX.append(example.document.text)
                            batchy.append(source_to_label[source])
                            added = True
                    elif target_type == MULTI_LABEL:
                        example_numerator = db.get_classifier_examples(source, paddings[i], batchsize)
                        for example in example_numerator:
                            batchX.append(example.document.text)
                            batchy.append(source_to_label[source] + ':' + example.label.name)
                            added = True
                else:
                    if target_type == SINGLE_LABEL:
                        example_numerator = db.get_classifier_examples(source, paddings[i], batchsize)
                        for example in example_numerator:
                            batchX.append(example.document.text)
                            batchy.append(example.label.name)
                            added = True
                    elif target_type == MULTI_LABEL:
                        labels = db.get_classifier_labels(source)
                        for example in db.get_classifier_examples(source, paddings[i], batchsize):
                            assigned = example.label.name
                            for label in labels:
                                batchX.append(example.document.text)
                                if label == assigned:
                                    batchy.append(label + ':yes')
                                else:
                                    batchy.append(label + ':no')
                            added = True
                paddings[i] += batchsize
            if len(batchX) > 0:
                pairs = list(zip(batchX, batchy))
                random.shuffle(pairs)
                batchX, batchy = zip(*pairs)
                merge_function(db, name, batchX, batchy)


def _lock_model(db, name):
    return DBLock(db, '%s %s' % (name, 'model'))


def _lock_trainingset(db, name):
    return DBLock(db, '%s %s' % (name, 'trainingset'))
