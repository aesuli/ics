import csv
from io import TextIOWrapper
import cherrypy
from cherrypy.lib import cptools
import numpy
from classifier.online_classifier import OnlineClassifier
from db.sqlalchemydb import SQLAlchemyDB

__author__ = 'Andrea Esuli'


class WebDatasetCollection(object):
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

        if not  self._db.dataset_exists(dataset_name):
            self._db.create_dataset(dataset_name)

        reader = csv.reader(TextIOWrapper(file.file))
        for row in reader:
            document_name = row[0]
            content = row[1]
            self._db.create_document(dataset_name, document_name, content)

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
        # TODO
        raise NotImplementedError()
        # clf = self._db.get_classifier_model(name)
        # return cptools.serveFile(path, "application/x-download",
        #                          "attachment", os.path.basename(path))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classify(self, **data):
        # TODO
        raise NotImplementedError()
        # try:
        #     name = data['name']
        # except KeyError:
        #     cherrypy.response.status = 400
        #     return 'Must specify a name'
        # try:
        #     X = data['X']
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
        return "0.0.2 (db: %s)" % self._db.version()


if __name__ == "__main__":
    with WebDatasetCollection('sqlite:///%s' % 'test.db') as wcc:
        cherrypy.quickstart(wcc, '/service')
