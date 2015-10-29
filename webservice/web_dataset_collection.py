import csv
import os
import shutil
from uuid import uuid4
import cherrypy
from cherrypy.lib.static import serve_download
from db.sqlalchemydb import SQLAlchemyDB
from webservice.util import get_fully_portable_file_name, logged_call

__author__ = 'Andrea Esuli'

DOWNLOAD_DIR = os.path.join(os.path.abspath('.'), 'downloads')
UPLOAD_DIR = os.path.join(os.path.abspath('.'), 'uploads')


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
        except KeyError as e:
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
            result = {}
            result['external_id'] = document.external_id
            result['text'] = document.text
            result['created'] = str(document.creation)
            return result
        else:
            cherrypy.response.status = 404
            return 'Position %i does not exits in \'%s\'' % (position, name)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classify(self, **data):
        # TODO
        raise NotImplementedError()
        # try:
        # name = data['name']
        # except KeyError:
        # cherrypy.response.status = 400
        # return 'Must specify a name'
        # try:
        # X = data['X']
        # except KeyError:
        #     try:
        #         X = data['X[]']
        #     except KeyError:
        #         cherrypy.response.status = 400
        #         return 'Must specify a vector of strings (X)'
        # X = numpy.atleast_1d(X)
        # clf = self._db.get_classifier_model(name)
        # return [[y] for y in clf.predict(X)]

    @cherrypy.expose
    def version(self):
        return "0.1.1 (db: %s)" % self._db.version()


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
    with WebDatasetCollection('sqlite:///%s' % 'test.db') as wcc:
        cherrypy.quickstart(wcc, '/service')
