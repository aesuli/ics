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

from classifier.classifier import BINARY_LABELS, YES_LABEL, NO_LABEL
from db.sqlalchemydb import SQLAlchemyDB, ClassificationMode, HUMAN_LABEL, MACHINE_LABEL
from util.util import get_fully_portable_file_name, bool_to_string

__author__ = 'Andrea Esuli'

MAX_BATCH_SIZE = 1000

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
            classifier_info['mode'] = self._db.get_preferred_classification_mode(name).value
            classifier_info['labels'] = self._db.get_classifier_labels(name)
            classifier_info['description'] = self._db.get_classifier_description(name)
            classifier_info['created'] = str(self._db.get_classifier_creation_time(name))
            classifier_info['updated'] = str(self._db.get_classifier_last_update_time(name))
            classifier_info['size'] = self._db.get_training_document_count(name)
            result.append(classifier_info)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def count(self):
        return str(len(list(self._db.classifier_names())))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classification_modes(self):
        return [mode.value for mode in ClassificationMode]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def create(self, **data):
        try:
            name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'

        try:
            classification_mode = data['mode']
        except KeyError:
            classification_mode = ClassificationMode.SINGLE_LABEL

        try:
            classification_mode = ClassificationMode(classification_mode)
        except ValueError:
            cherrypy.response.status = 400
            return f'Classifier type must be one of {self.classification_modes()}'

            labels = []
        try:
            labels = data['labels']
        except KeyError:
            try:
                labels = data['labels[]']
            except KeyError:
                pass
        if type(labels) is str:
            if len(labels) == 0:
                labels = []
            else:
                labels = [labels]
        if not (type(labels) is list or type(labels) is set):
            cherrypy.response.status = 400
            return f'Unknown label specification {labels}, must be a list of strings.'

        name = str.strip(name)
        if len(name) < 1:
            cherrypy.response.status = 400
            return 'Empty classifier name'

        labels = map(str.strip, labels)
        labels = list(set(labels))
        for label in labels:
            if len(label) < 1:
                cherrypy.response.status = 400
                return 'Empty label name'

        try:
            overwrite = data['overwrite']
            if overwrite == 'false' or overwrite == 'False':
                overwrite = False
        except KeyError:
            overwrite = False

        with self._db._lock_classifier_training_documents(name), self._db._lock_classifier_model(name):
            if not self._db.classifier_exists(name):
                self._db.create_classifier(name, labels, classification_mode)
            elif not overwrite:
                cherrypy.response.status = 403
                return '%s is already in the collection' % name
            else:
                self.delete(name)
                self._db.create_classifier(name, labels, classification_mode)
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
                try:
                    idx = 0
                    y = []
                    while True:
                        y.append(data['y[' + str(idx) + '][]'])
                        idx += 1
                except KeyError:
                    if idx == 0:
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
    def preferred_classification_mode(self, name):
        return self._db.get_preferred_classification_mode(name).value

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def delete(self, name):
        try:
            self._db.delete_classifier(name)
        except KeyError as e:
            cherrypy.response.status = 404
            return '%s does not exist' % name
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
            return '%s does not exist in %s' % (label_name, name)
        except Exception as e:
            cherrypy.response.status = 500
            return 'Error (%s)' % str(e)
        else:
            return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def label_add(self, name, label_name):
        try:
            self._db.add_classifier_label(name, label_name)
        except KeyError:
            cherrypy.response.status = 404
            return '%s does not exist' % (name)
        except Exception as e:
            cherrypy.response.status = 500
            return 'Error (%s)' % str(e)
        else:
            return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def label_delete(self, name, label_name):
        try:
            self._db.delete_classifier_label(name, label_name)
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
            return '%s does not exist' % name
        return str(self._db.get_training_document_count(name))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_training_data(self, name, page=None, page_size=50, filter=None):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exist' % name
        page_size = int(page_size)
        if page is None:
            offset = 0
        else:
            offset = int(page) * page_size
        limit = page_size
        batch = list()
        classification_mode = self._db.get_preferred_classification_mode(name)

        for trainingdocument in self._db.get_training_documents(name, offset, limit, filter):
            batch.append({'id': trainingdocument.id, 'text': trainingdocument.text})

        for entry in batch:
            id_ = entry['id']
            labels = self._db.get_labels_of_training_id(name, id_)
            entry['update'] = str(self._db.get_training_id_last_update_time(name, id_))
            if classification_mode == ClassificationMode.SINGLE_LABEL:
                entry['labels'] = []
                for label, assigned in labels:
                    if assigned:
                        entry['labels'] = [(name, label)]
                        break
            elif classification_mode == ClassificationMode.MULTI_LABEL:
                entry['labels'] = [(name, (label, assigned)) for label, assigned in labels]

        return batch

    @cherrypy.expose
    def delete_training_example(self, classifier_name, training_document_id):
        if not self._db.classifier_exists(classifier_name):
            cherrypy.response.status = 404
            return '%s does not exist' % classifier_name
        self._db.delete_training_example(classifier_name, training_document_id)

    @cherrypy.expose
    def download_training_data(self, name):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exist' % name
        filename = 'training data %s %s.csv' % (name, str(self._db.get_classifier_last_update_time(name)))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)
        if not os.path.isfile(fullpath):
            classification_mode = self._db.get_preferred_classification_mode(name)
            header = list()
            header.append('#id')
            header.append('text')
            header.append(
                f'{name} = {classification_mode.value}, ({", ".join(self._db.get_classifier_labels(name))})')
            try:
                with open(fullpath, 'w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow(header)
                    added = True
                    block_count = 0
                    block_size = MAX_BATCH_SIZE
                    while added:
                        offset = block_count * block_size
                        block_count += 1
                        rows = list()
                        for trainingdocument in self._db.get_training_documents(name, offset, MAX_BATCH_SIZE):
                            rows.append([trainingdocument.id, trainingdocument.text])

                        for id_, text in rows:
                            labels = self._db.get_labels_of_training_id(name, id_)
                            row = [id_, text]
                            if classification_mode == ClassificationMode.SINGLE_LABEL:
                                for label, assigned in labels:
                                    if assigned:
                                        row.append(f'{name}:{label}{HUMAN_LABEL}')
                                        break
                            else:
                                for label, assigned in labels:
                                    row.append(
                                        f'{name}:{label}:{bool_to_string(assigned, YES_LABEL, NO_LABEL)}{HUMAN_LABEL}')
                            writer.writerow(row)
                        added = len(rows) > 0
            except:
                os.unlink(fullpath)

        return serve_file(fullpath, "text/csv", "attachment")

    @cherrypy.expose
    def download_model(self, name):
        if not self._db.classifier_exists(name):
            cherrypy.response.status = 404
            return '%s does not exist' % name
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
        classifiers_mode = dict()

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
                    if classifier_label.endswith(HUMAN_LABEL):
                        classifier_label = classifier_label[:-len(HUMAN_LABEL)]
                    elif classifier_label.endswith(MACHINE_LABEL):
                        classifier_label = classifier_label[:-len(MACHINE_LABEL)]
                    split_values = classifier_label.split(':')
                    if len(split_values) == 2:
                        classifier_name, label = split_values
                        classifier_name = classifier_name.strip()
                        label = label.strip()
                        classifier_mode = classifiers_mode.get(classifier_name, None)
                        if classifier_mode:
                            if classifier_mode != ClassificationMode.SINGLE_LABEL:
                                cherrypy.response.status = 400
                                return f'Inconsistent single-label and multi-label labeling for classifier "classifier_name"'
                        else:
                            classifiers_mode[classifier_name] = ClassificationMode.SINGLE_LABEL
                    elif len(split_values) == 3:
                        classifier_name, label, value = split_values
                        classifier_name = classifier_name.strip()
                        label = label.strip()
                        value = value.strip()
                        classifier_mode = classifiers_mode.get(classifier_name, None)
                        if classifier_mode:
                            if classifier_mode != ClassificationMode.MULTI_LABEL:
                                cherrypy.response.status = 400
                                return f'Inconsistent single-label and multi-label labeling for classifier "classifier_name"'
                        else:
                            classifiers_mode[classifier_name] = ClassificationMode.MULTI_LABEL
                        if value != NO_LABEL and value != YES_LABEL:
                            cherrypy.response.status = 400
                            return f'Unsupported label assignment value "{value}" for label ' \
                                   f'"{label}" for classifier "classifier_name" (must be either "{YES_LABEL}" or "{NO_LABEL}").'
                    classifiers_definition[classifier_name].add(label)

        jobs = list()

        for classifier_name in classifiers_definition:
            labels = classifiers_definition[classifier_name]
            if not self._db.classifier_exists(classifier_name):
                self.create(**{'name': classifier_name, 'labels': labels, 'mode': classifiers_mode[classifier_name]})
            else:
                if len(set(labels).difference(set(self._db.get_classifier_labels(classifier_name)))) > 0:
                    cherrypy.response.status = 400
                    return 'Existing classifier \'%s\' uses a different set of labels than input file' % classifier_name
                classifier_mode = self._db.get_preferred_classification_mode(classifier_name)
                if classifier_mode != classifiers_mode[classifier_name]:
                    cherrypy.response.status = 400
                    return f'Existing classifier "{classifier_name}" uses a different labeling mode "{classifier_mode}"'

        for classifier_name in classifiers_definition:
            jobs.append(self._db.create_job(_update_from_file, (
                _update_model, 'utf-8', self._db_connection_string, fullpath, classifier_name),
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
            classification_mode = data['mode']
        except KeyError:
            classification_mode = ClassificationMode.SINGLE_LABEL

        try:
            classification_mode = ClassificationMode(classification_mode)
        except ValueError:
            cherrypy.response.status = 400
            return 'Classifier type must be one of ' + str(self.classification_modes())

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

        with self._db._lock_classifier_training_documents(name), self._db._lock_classifier_model(name):
            if self._db.classifier_exists(name):
                if overwrite:
                    self._db.delete_classifier(name)
                else:
                    cherrypy.response.status = 403
                    return f'A classifier with name {name} is already in the collection'

            with open(fullpath, 'rb') as infile:
                model = pickle.load(infile)
                labels = list(model.labels())
                self._db.create_classifier(name, labels, classification_mode)
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

        try:
            classification_mode = ClassificationMode(data['mode'])
        except KeyError:
            classification_mode = self._db.get_preferred_classification_mode(name)

        X = numpy.atleast_1d(X)
        cherrypy.log(
            f'ClassifierCollectionService.classify(name="{name}", len(X)="{len(X)}", classification_mode="{classification_mode.value}")')

        show_gold_labels = cherrypy.request.login is not None

        return self._db.classify(name, X, classification_mode=classification_mode, show_gold_labels=show_gold_labels)

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
        cherrypy.log(f'ClassifierCollectionService.score(name="{name}", X="{X}")')
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
            with self._db._lock_classifier_training_documents(new_name[label]), self._db._lock_classifier_model(
                    new_name[label]):
                if self._db.classifier_exists(new_name[label]) and not overwrite:
                    cherrypy.response.status = 403
                    return f'A classifier with name {new_name[label]} is already in the collection'

        jobs = list()
        for label in labels:
            with self._db._lock_classifier_training_documents(new_name[label]), self._db._lock_classifier_model(
                    new_name[label]):
                if not self._db.classifier_exists(new_name[label]):
                    self._db.create_classifier(new_name[label], [new_name[label]], ClassificationMode.MULTI_LABEL)
                elif not overwrite:
                    cherrypy.response.status = 403
                    return f'A classifier with name {new_name[label]} is already in the collection'

                else:
                    self.delete(label)
                    self._db.create_classifier(new_name[label], [new_name[label]], ClassificationMode.MULTI_LABEL)
                jobs.append(self._db.create_job(_extract_label_classifier,
                                                (self._db_connection_string, name, label, new_name[label]),
                                                description=f'extract label classifier \'{new_name[label]}\''))
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
            classification_mode = ClassificationMode(data['mode'])
        except KeyError:
            classification_mode = self._db.get_preferred_classification_mode(name)

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

        with self._db._lock_classifier_training_documents(name), self._db._lock_classifier_model(name):
            if not self._db.classifier_exists(name):
                self._db.create_classifier(name, labels, classification_mode)
            elif not overwrite:
                cherrypy.response.status = 403
                return f'{name} is already in the collection'
            else:
                self.delete(name)
                self._db.create_classifier(name, labels, classification_mode)
            model_merged = False
            if len(sources) == 1:
                model_merged = True
                model = self._db.get_classifier_model(sources[0])
                model.rename(name)
                self._db.update_classifier_model(name, model)

            jobs = list()
            jobs.append(self._db.create_job(_merge, (
                self._db_connection_string, _update_trainingset, name, sources, binary_by_name),
                                            description=f'merging classifiers from {sources} to "{name}" (binary_by_name={binary_by_name})'))
            if not model_merged:
                jobs.append(self._db.create_job(_merge, (
                    self._db_connection_string, _update_model, name, sources, binary_by_name),
                                                description=f'merging models from {sources} to "{name}" (binary_by_name={binary_by_name})'))
        return jobs

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def version(self):
        return f"6.2.1 (db: {self._db.version()})"


def _update_trainingset(db_connection, name, X, y):
    cherrypy.log(
        'ClassifierCollectionService._update_trainingset(name="' + name + '", len(X)="' + str(len(X)) + '")')
    if type(db_connection) == str:
        with SQLAlchemyDB(db_connection) as db:
            with db._lock_classifier_training_documents(name):
                db.create_training_examples(name, list(zip(X, y)))
    else:
        with db_connection._lock_classifier_model(name):
            db_connection.create_training_examples(name, list(zip(X, y)))


def _update_model(db_connection, name, X, y):
    if len(X) > 0:
        cherrypy.log(
            'ClassifierCollectionService._update_model(name="' + name + '", len(X)="' + str(len(X)) + '")')
        if type(db_connection) == str:
            with SQLAlchemyDB(db_connection) as db:
                with db._lock_classifier_model(name):
                    model = db.get_classifier_model(name)
                    model.learn(X, y)
                    db.update_classifier_model(name, model)
        else:
            with db_connection._lock_classifier_model(name):
                model = db_connection.get_classifier_model(name)
                model.learn(X, y)
                db_connection.update_classifier_model(name, model)


def _update_from_file(update_function, encoding, db_connection_string, filename, classifier_name):
    cherrypy.log(
        f'ClassifierCollectionService._update_from_file(filename="{filename}", classifier_name="{classifier_name}")')
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
                if classifier_label.endswith(HUMAN_LABEL):
                    classifier_label = classifier_label[:-len(HUMAN_LABEL)]
                elif classifier_label.endswith(MACHINE_LABEL):
                    classifier_label = classifier_label[:-len(MACHINE_LABEL)]

                try:
                    example_fields = classifier_label.split(':')
                except:
                    continue
                example_classifier_name = example_fields[0].strip()
                if len(example_fields) == 2:
                    label = example_fields[1].strip()
                else:
                    label = (example_fields[1].strip(), example_fields[2].strip())
                if example_classifier_name is not None and example_classifier_name == classifier_name:
                    X.append(text)
                    y.append(label)
            if len(X) >= MAX_BATCH_SIZE:
                update_function(db_connection_string, classifier_name, X, y)
                X = []
                y = []
        if len(X) > 0:
            update_function(db_connection_string, classifier_name, X, y)


def _extract_label_classifier(db_connection_string, classifier, label_to_extract, new_name):
    cherrypy.log(
        f'ClassifierCollectionService._extract_label_classifier(classifier="{classifier}", label="{label_to_extract}")')
    with SQLAlchemyDB(db_connection_string) as db:
        model = db.get_classifier_model(classifier)
        db.update_classifier_model(label_to_extract, model.get_label_classifier(label_to_extract))
        batchsize = MAX_BATCH_SIZE
        block = 0
        added = True
        while added:
            batch = list()
            added = False
            for example in db.get_classifier_examples_with_label(classifier, label_to_extract, block * batchsize,
                                                                 batchsize):
                batch.append((example.document.text, label_to_extract, example.assigned))
                added = True
            if len(batch) > 0:
                with db._lock_classifier_training_documents(new_name):
                    db.create_training_examples(new_name, batch)
            block += 1


def _merge(db_connection_string, merge_function, name, sources, binary_by_name):
    sources = list(sources)
    cherrypy.log(
        f'ClassifierCollectionService._merge_classifiers(merge_function={merge_function}, name="{name}", sources={sources}, binary_by_name={binary_by_name})')
    with SQLAlchemyDB(db_connection_string) as db:
        target_mode = db.get_preferred_classification_mode(name)

        binary_sources = set()
        for source in sources:
            if db.get_preferred_classification_mode(source) == ClassificationMode.SINGLE_LABEL and set(
                    db.get_classifier_labels(source)) == set(BINARY_LABELS) and binary_by_name:
                binary_sources.add(source)
        sizes = list()

        for source in sources:
            if target_mode == ClassificationMode.SINGLE_LABEL and source in binary_sources:
                sizes.append(db.get_classifier_examples_with_label_count(source, YES_LABEL))
            else:
                sizes.append(db.get_training_document_count(source))
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
                    if target_mode == ClassificationMode.SINGLE_LABEL:
                        for example in db.get_classifier_examples_with_label(source, YES_LABEL, paddings[i], batchsize):
                            batchX.append(example.document.text)
                            batchy.append(source)
                            added = True
                    elif target_mode == ClassificationMode.MULTI_LABEL:
                        text_and_id = [(example.text, example.id) for example in
                                       db.get_training_documents(source, paddings[i], batchsize)]
                        for text, id in text_and_id:
                            for label, assigned in db.get_labels_of_training_id(source, id):
                                if label == YES_LABEL:
                                    batchX.append(text)
                                    if assigned:
                                        batchy.append((source, YES_LABEL))
                                    else:
                                        batchy.append((source, NO_LABEL))
                                    added = True
                else:
                    if target_mode == ClassificationMode.SINGLE_LABEL:
                        text_and_id = [(example.text, example.id) for example in
                                       db.get_training_documents(source, paddings[i], batchsize)]
                        for text, id in text_and_id:
                            for label, assigned in db.get_labels_of_training_id(source, id, by_last_update=True):
                                if assigned:
                                    batchX.append(text)
                                    batchy.append(label)
                                    added = True
                                    break
                    elif target_mode == ClassificationMode.MULTI_LABEL:
                        text_and_id = [(example.text, example.id) for example in
                                       db.get_training_documents(source, paddings[i], batchsize)]
                        for text, id in text_and_id:
                            for label, assigned in db.get_labels_of_training_id(source, id):
                                batchX.append(text)
                                if assigned:
                                    batchy.append((label, YES_LABEL))
                                else:
                                    batchy.append((label, NO_LABEL))
                                added = True
                paddings[i] += batchsize
            if len(batchX) > 0:
                pairs = list(zip(batchX, batchy))
                random.shuffle(pairs)
                batchX, batchy = zip(*pairs)
                merge_function(db, name, batchX, batchy)
