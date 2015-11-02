import csv
import json
import os
import shutil
from uuid import uuid4
import cherrypy
from cherrypy.lib.static import serve_download
import numpy
from db.sqlalchemydb import SQLAlchemyDB
from util.util import get_fully_portable_file_name, logged_call
from webservice.background_processor import BackgroundProcessor

__author__ = 'Andrea Esuli'

DOWNLOAD_DIR = os.path.join(os.path.abspath('.'), 'downloads')
UPLOAD_DIR = os.path.join(os.path.abspath('.'), 'uploads')

MAX_BATCH_SIZE = 1000


class WebDatasetCollection(object):
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
        for name in self._db.dataset_names():
            dataset_info = dict()
            dataset_info['name'] = name
            dataset_info['created'] = self._db.get_dataset_creation_time(name)
            dataset_info['updated'] = self._db.get_dataset_last_update_time(name)
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
        fullpath = os.path.join(UPLOAD_DIR, filename)
        with open(fullpath, 'wb') as outfile:
            shutil.copyfileobj(file.file, outfile)

        self._background_processor.put(_create_documents, (self._db_connection_string, dataset_name, fullpath))

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
        filename = 'dataset %s %s.csv' % (name, self._db.get_dataset_last_update_time(name))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(DOWNLOAD_DIR, filename)
        if not os.path.isfile(fullpath):
            try:
                with open(fullpath, 'w') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    for document in self._db.get_dataset_documents(name):
                        writer.writerow([document.external_id, document.text])
            except:
                os.unlink(filename)

        return serve_download(fullpath)

    @cherrypy.expose
    def size(self, name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        return str(self._db.get_dataset_size(name))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def document(self, name, position):
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
        fullpath = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(fullpath):
            cherrypy.response.status = 409
            return 'This classification has been already requested.'

        # TODO do it in background
        with open(fullpath, 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            header = list()
            for classifier in classifiers:
                classifiers_header = dict()
                classifiers_header['name'] = classifier
                classifiers_header['classes'] = self._db.get_classifier_classes(classifier)
                header.append(classifiers_header)
            writer.writerow([json.dumps(header)])

            X = list()
            id = list()
            for document in self._db.get_dataset_documents(datasetname):
                id.append(document.external_id)
                X.append(document.text)
                if len(X) >= MAX_BATCH_SIZE:
                    self._classify_and_write(id, X, classifiers, writer)
                    X = list()
                    id = list()
            if len(X) > 0:
                self._classify_and_write(id, X, classifiers, writer)

        return 'Ok'

    def _classify_and_write(self, id, X, classifiers, writer):
        cols = list()
        cols.append(id)
        cols.append(X)
        for classifier in classifiers:
            cols.append(['%s:%s' % (classifier, y) for y in self._db.classify(classifier, X)])
        for row in zip(*cols):
            writer.writerow(row)

    @cherrypy.expose
    def version(self):
        return "0.2.1 (db: %s)" % self._db.version()


@logged_call
def _create_documents(db_connection_string, dataset_name, filename):
    with SQLAlchemyDB(db_connection_string) as db:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            first_row = next(reader)
            if len(first_row) > 1:
                document_name = first_row[0]
                content = first_row[1]
                db.create_document(dataset_name, document_name, content)
            for row in reader:
                document_name = row[0]
                content = row[1]
                db.create_document(dataset_name, document_name, content)


if __name__ == "__main__":
    with WebDatasetCollection('sqlite:///%s' % 'test.db', BackgroundProcessor('sqlite:///%s' % 'test.db')) as wcc:
        cherrypy.quickstart(wcc, '/service')
