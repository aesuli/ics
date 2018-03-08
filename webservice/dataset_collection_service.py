import csv
import json
import os
import random
import shutil
from uuid import uuid4

import cherrypy
import numpy as np
from cherrypy.lib.static import serve_file

from db.sqlalchemydb import SQLAlchemyDB, Job, Label
from util.util import get_fully_portable_file_name, logged_call

__author__ = 'Andrea Esuli'

MAX_BATCH_SIZE = 1000
CSV_LARGE_FIELD = 1024 * 1024 * 10

QUICK_CLASSIFICATION_BATCH_SIZE = 1000


class DatasetCollectionService(object):
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
        names = self._db.dataset_names()[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
        for name in names:
            dataset_info = dict()
            dataset_info['name'] = name
            dataset_info['created'] = str(self._db.get_dataset_creation_time(name))
            dataset_info['updated'] = str(self._db.get_dataset_last_update_time(name))
            dataset_info['size'] = self._db.get_dataset_size(name)
            result.append(dataset_info)
        return result

    @cherrypy.expose
    def count(self):
        return str(len(list(self._db.dataset_names())))

    @cherrypy.expose
    def create(self, name):
        self._db.create_dataset(name)
        return 'Ok'

    @cherrypy.expose
    def add_document(self, dataset_name, document_name, document_content):
        if not self._db.dataset_exists(dataset_name):
            self._db.create_dataset(dataset_name)
        self._db.create_dataset_document(dataset_name, document_name, document_content)

    @cherrypy.expose
    def delete_document(self, dataset_name, document_name):
        if not self._db.dataset_exists(dataset_name):
            cherrypy.response.status = 404
            return '%s does not exits' % dataset_name
        self._db.delete_dataset_document(dataset_name, document_name)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def upload(self, **data):
        try:
            dataset_name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            file = data['file']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must upload a file'

        if not self._db.dataset_exists(dataset_name):
            self._db.create_dataset(dataset_name)

        filename = 'dataset %s %s.csv' % (dataset_name, uuid4())
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._upload_dir, filename)
        with open(fullpath, 'wb') as outfile:
            shutil.copyfileobj(file.file, outfile)

        job_id = self._db.create_job(_create_dataset_documents, (self._db_connection_string, dataset_name, fullpath),
                                     description='upload to dataset \'%s\'' % dataset_name)

        return [job_id]

    @cherrypy.expose
    def rename(self, name, newname):
        try:
            self._db.rename_dataset(name, newname)
        except KeyError:
            cherrypy.response.status = 404
            return '%s does not exits' % name
        except Exception as e:
            cherrypy.response.status = 500
            return 'Error (%s)' % str(e)
        else:
            return 'Ok'

    @cherrypy.expose
    def delete(self, name):
        try:
            self._db.delete_dataset(name)
        except KeyError:
            cherrypy.response.status = 404
            return '%s does not exits' % name
        else:
            return 'Ok'

    @cherrypy.expose
    def download(self, name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        filename = 'dataset %s %s.csv' % (name, str(self._db.get_dataset_last_update_time(name)))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)
        if not os.path.isfile(fullpath):
            try:
                with open(fullpath, 'w') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    for document in self._db.get_dataset_documents_by_name(name):
                        writer.writerow([document.external_id, document.text])
            except:
                os.unlink(fullpath)

        return serve_file(fullpath, "text/csv", "attachment")

    @cherrypy.expose
    def size(self, name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        return str(self._db.get_dataset_size(name))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def document_by_name(self, name, documentname):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        document = self._db.get_dataset_document_by_name(name, documentname)
        if document is not None:
            result = dict()
            result['external_id'] = document.external_id
            result['text'] = document.text
            result['created'] = str(document.creation)
            return result
        else:
            cherrypy.response.status = 404
            return 'Document with name \'%i\' does not exits in \'%s\'' % (documentname, name)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def document_by_position(self, name, position):
        position = int(position)
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        document = self._db.get_dataset_document_by_position(name, position)
        if document is not None:
            result = dict()
            result['external_id'] = document.external_id
            result['text'] = document.text
            result['created'] = str(document.creation)
            return result
        else:
            cherrypy.response.status = 404
            return 'Position %i does not exits in \'%s\'' % (position, name)

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def most_uncertain_document_id(self, dataset_name, classifier_name):
        dataset_size = self._db.get_dataset_size(dataset_name)
        offset = random.randint(0, dataset_size - QUICK_CLASSIFICATION_BATCH_SIZE)
        X = list()
        for doc in self._db.get_dataset_documents_by_position(dataset_name, offset, QUICK_CLASSIFICATION_BATCH_SIZE):
            X.append(doc.text)
        scores = self._db.score(classifier_name, X)
        positions_scores = list()
        for i, dict_ in enumerate(scores):
            probs = self._softmax(list(dict_.values()))
            probs.sort()
            diff = probs[-1] - probs[-2]
            positions_scores.append((i, diff))
        positions_scores.sort(key=lambda x: x[1])
        for position, score in positions_scores:
            text = X[position]
            if not self._db.classifier_has_example(classifier_name, text, True):
                return offset + position
        return random.randint(0, dataset_size - 1)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def most_certain_document_id(self, dataset_name, classifier_name):
        dataset_size = self._db.get_dataset_size(dataset_name)
        offset = random.randint(0, dataset_size - QUICK_CLASSIFICATION_BATCH_SIZE)
        X = list()
        for doc in self._db.get_dataset_documents_by_position(dataset_name, offset, QUICK_CLASSIFICATION_BATCH_SIZE):
            X.append(doc.text)
        scores = self._db.score(classifier_name, X)
        positions_scores = list()
        for i, dict_ in enumerate(scores):
            probs = self._softmax(list(dict_.values()))
            probs.sort()
            diff = probs[-1] - probs[-2]
            positions_scores.append((i, diff))
        positions_scores.sort(key=lambda x: -x[1])
        for position, score in positions_scores:
            text = X[position]
            if not self._db.classifier_has_example(classifier_name, text, True):
                return offset + position
        return random.randint(0, dataset_size - 1)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def random_hidden_document_id(self, dataset_name, classifier_name):
        document_ids = [document_id for document_id in
                        self._db.get_dataset_documents_with_label(dataset_name, classifier_name, Label.HIDDEN_LABEL)]
        if document_ids:
            document_id = random.choice(document_ids)
            position = self._db.get_dataset_document_position_by_id(dataset_name, document_id)
            return position
        else:
            cherrypy.response.status = 400
            return f'No hidden documents in dataset \'{dataset_name}\' for classifier \'{classifier_name}\''

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classify(self, **data):
        try:
            datasetname = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a dataset name'
        try:
            classifiers = data['classifiers']
        except KeyError:
            try:
                classifiers = data['classifiers[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify a vector of names of classifiers'
        classifiers = np.atleast_1d(classifiers).tolist()

        last_update_time = self._db.get_most_recent_classifier_update_time(classifiers)
        dataset_update_time = self._db.get_dataset_last_update_time(datasetname)
        if last_update_time is None or last_update_time < dataset_update_time:
            last_update_time = dataset_update_time

        filename = 'dataset %s classified %s %s.csv' % (
            datasetname, "-".join(classifiers), str(last_update_time))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)

        if self._db.classification_exists(fullpath):
            cherrypy.response.status = 409
            return 'An up-to-date classification is already available.'

        job_id = self._db.create_job(_classify,
                                     (self._db_connection_string, datasetname, classifiers, fullpath),
                                     description='classify dataset \'%s\' with %s' % (
                                         datasetname,
                                         ', '.join(['\'%s\'' % classifier for classifier in classifiers])))

        self._db.create_classification_job(datasetname, classifiers, job_id, fullpath)

        return [job_id]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_classification_jobs(self, name, page=0, page_size=50):
        got_deleted = True
        result = None
        while got_deleted:
            got_deleted = False
            result = list()
            to_delete = list()
            jobs = self._db.get_classification_jobs(name)[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
            for classification_job in jobs:
                classification_job_info = dict()
                classification_job_info['id'] = classification_job.id
                if (classification_job.filename is None or not os.path.exists(
                        classification_job.filename)) and classification_job.job.status == Job.status_done:
                    to_delete.append(classification_job.id)
                    got_deleted = True
                    continue
                classification_job_info['dataset'] = name
                classification_job_info['classifiers'] = classification_job.classifiers
                classification_job_info['status'] = classification_job.job.status
                classification_job_info['creation'] = str(classification_job.job.creation)
                classification_job_info['completion'] = str(classification_job.job.completion)
                result.append(classification_job_info)

            for id in to_delete:
                self._db.delete_classification_job(id)
        return result

    @cherrypy.expose
    def get_classification_jobs_count(self, name):
        return str(len(list(self._db.get_classification_jobs(name))))

    @cherrypy.expose
    def download_classification(self, id):
        filename = self._db.get_classification_job_filename(id)
        if filename is None or not os.path.exists(filename):
            cherrypy.response.status = 404
            return "File not found"
        return serve_file(filename, "text/csv", "attachment")

    @cherrypy.expose
    def delete_classification(self, id):
        filename = self._db.get_classification_job_filename(id)
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass
        self._db.delete_classification_job(id)
        return 'Ok'

    @cherrypy.expose
    def version(self):
        return "1.2.1 (db: %s)" % self._db.version()


@logged_call
def _classify(db_connection_string, datasetname, classifiers, fullpath):
    with SQLAlchemyDB(db_connection_string) as db:
        tempfile = fullpath + '.tmp'
        try:
            with open(tempfile, 'w') as file:
                writer = csv.writer(file, lineterminator='\n')
                header = list()
                for classifier in classifiers:
                    if db.classifier_exists(classifier):
                        classifiers_header = dict()
                        classifiers_header['name'] = classifier
                        classifiers_header['labels'] = db.get_classifier_labels(classifier)
                        header.append(classifiers_header)
                writer.writerow(['# ' + json.dumps({'classifiers': header})])

                X = list()
                id = list()
                for document in db.get_dataset_documents_by_name(datasetname):
                    id.append(document.external_id)
                    X.append(document.text)
                    if len(X) >= MAX_BATCH_SIZE:
                        _classify_and_write(db, id, X, classifiers, writer)
                        X = list()
                        id = list()
                if len(X) > 0:
                    _classify_and_write(db, id, X, classifiers, writer)
            try:
                os.unlink(fullpath)
            except FileNotFoundError:
                pass
            os.rename(tempfile, fullpath)
        except:
            try:
                os.unlink(tempfile)
            except FileNotFoundError:
                pass
            try:
                os.unlink(fullpath)
            except FileNotFoundError:
                pass


def _classify_and_write(db, id, X, classifiers, writer):
    cols = list()
    cols.append(id)
    cols.append(X)
    for classifier in classifiers:
        cols.append(['%s:%s' % (classifier, y) for y in db.classify(classifier, X)])
    for row in zip(*cols):
        writer.writerow(row)


@logged_call
def _create_dataset_documents(db_connection_string, dataset_name, filename):
    with SQLAlchemyDB(db_connection_string) as db:
        if not db.dataset_exists(dataset_name):
            db.create_dataset(dataset_name)
        if csv.field_size_limit() < CSV_LARGE_FIELD:
            csv.field_size_limit(CSV_LARGE_FIELD)
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
            reader = csv.reader(file)
            batch_size = 1000
            for row in reader:
                if len(row) > 1:
                    document_name = row[0]
                    content = row[1]
                    db.create_dataset_document(dataset_name, document_name, content)
