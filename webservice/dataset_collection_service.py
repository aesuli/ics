import csv
import json
import logging
import os
import shutil
from uuid import uuid4

import cherrypy
import numpy
from chardet.universaldetector import UniversalDetector
from cherrypy.lib.static import serve_file

from db.sqlalchemydb import SQLAlchemyDB, Job
from util.util import get_fully_portable_file_name, logged_call
from webservice.background_processor_service import BackgroundProcessor

__author__ = 'Andrea Esuli'

MAX_BATCH_SIZE = 1000


class DatasetCollectionService(object):
    def __init__(self, db_connection_string, data_dir, background_processor):
        self._db_connection_string = db_connection_string
        self._db = SQLAlchemyDB(db_connection_string)
        self._download_dir = os.path.join(data_dir, 'downloads')
        self._upload_dir = os.path.join(data_dir, 'uploads')
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
        for name in self._db.dataset_names():
            dataset_info = dict()
            dataset_info['name'] = name
            dataset_info['created'] = str(self._db.get_dataset_creation_time(name))
            dataset_info['updated'] = str(self._db.get_dataset_last_update_time(name))
            dataset_info['size'] = self._db.get_dataset_size(name)
            result.append(dataset_info)
        return result

    @cherrypy.expose
    def create(self, name):
        self._db.create_dataset(name)
        return 'Ok'

    @cherrypy.expose
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

        self._background_processor.put(_create_documents, (self._db_connection_string, dataset_name, fullpath),
                                       description='upload to dataset \'%s\'' % dataset_name)

        return 'Ok'

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
                    for document in self._db.get_dataset_documents(name):
                        writer.writerow([document.external_id, document.text])
            except:
                os.unlink(filename)

        return serve_file(fullpath, "text/csv", "attachment")

    @cherrypy.expose
    def size(self, name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        return str(self._db.get_dataset_size(name))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def document(self, name, position):
        position = int(position)
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        document = self._db.get_dataset_document(name, position)
        if document is not None:
            result = dict()
            result['external_id'] = document.external_id
            result['text'] = document.text
            result['created'] = str(document.creation)
            return result
        else:
            cherrypy.response.status = 404
            return 'Position %i does not exits in \'%s\'' % (position, name)

    @cherrypy.expose
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
        classifiers = numpy.atleast_1d(classifiers).tolist()

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

        job_id = self._background_processor.put(_classify,
                                                (self._db_connection_string, datasetname, classifiers, fullpath),
                                                description='classify dataset \'%s\' with %s' % (
                                                    datasetname,
                                                    ', '.join(['\'%s\'' % classifier for classifier in classifiers])))

        self._db.create_classification_job(datasetname, classifiers, job_id, fullpath)

        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_classification_jobs(self, name):
        result = list()
        to_delete = list()
        for classification_job in self._db.get_classification_jobs(name):
            classification_job_info = dict()
            classification_job_info['id'] = classification_job.id
            if (classification_job.filename is None or not os.path.exists(
                    classification_job.filename)) and classification_job.job.status == Job.status_done:
                to_delete.append(classification_job.id)
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
        return "0.2.5 (db: %s)" % self._db.version()


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
                        classifiers_header['classes'] = db.get_classifier_labels(classifier)
                        header.append(classifiers_header)
                writer.writerow([json.dumps(header)])

                X = list()
                id = list()
                for document in db.get_dataset_documents(datasetname):
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
def _create_documents(db_connection_string, dataset_name, filename):
    detector = UniversalDetector()
    with open(filename, 'rb') as file:
        for line in file:
            detector.feed(line)
            if detector.done:
                break
    encoding = detector.result['encoding']
    cherrypy.log('Encode guessing for uploaded file ' + json.dumps(detector.result), severity=logging.INFO)
    with SQLAlchemyDB(db_connection_string) as db:
        with open(filename, 'r', encoding=encoding, errors='ignore') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 1:
                    document_name = row[0]
                    content = row[1]
                    db.create_document(dataset_name, document_name, content)


if __name__ == "__main__":
    with DatasetCollectionService('sqlite:///%s' % 'test.db', '.', BackgroundProcessor('sqlite:///%s' % 'test.db')) as wcc:
        cherrypy.quickstart(wcc, '/service/wdc')
