import os

import cherrypy
from mako.lookup import TemplateLookup

from db.sqlalchemydb import SQLAlchemyDB
from webservice.auth_controller import always_auth

__author__ = 'Andrea Esuli'


class WebClient(object):
    def __init__(self, db_connection_string, media_dir, auth_path, classifier_path, dataset_path, processor_path, name):
        self._db = SQLAlchemyDB(db_connection_string)
        self._media_dir = media_dir
        self._template_data = {'auth_path': auth_path,
                               'classifier_path': classifier_path,
                               'dataset_path': dataset_path,
                               'processor_path': processor_path,
                               'name': name}
        self._lookup = TemplateLookup(os.path.join(media_dir, 'template'))

    def get_config(self):
        return {
            '/css':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': os.path.join(self._media_dir, 'css'),
                 },
            '/js':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': os.path.join(self._media_dir, 'js'),
                 },
        }

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def session_data(self):
        return {'username': cherrypy.request.login}

    @cherrypy.expose
    @always_auth()
    def login(self, username=None, password=None, error_message=None):
        if username is None or password is None:
            template = self._lookup.get_template('login.html')
            if username is None:
                username = ""
            if error_message is None:
                error_message = ""
            return template.render(
                **{**self._template_data, **self.session_data(), **{'username': username, 'msg': error_message}})

    @cherrypy.expose
    def index(self):
        template = self._lookup.get_template('datasets.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def typeandlabel(self):
        template = self._lookup.get_template('typeandlabel.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def browseandlabel(self, name=None):
        if name is None:
            raise cherrypy.HTTPRedirect('/datasets')

        if not self._db.dataset_exists(name):
            raise cherrypy.HTTPRedirect('/datasets')

        template = self._lookup.get_template('browseandlabel.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def classify(self, name=None):
        if name is None:
            raise cherrypy.HTTPRedirect('/datasets')

        if not self._db.dataset_exists(name):
            raise cherrypy.HTTPRedirect('/datasets')

        template = self._lookup.get_template('classify.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def classifiers(self):
        template = self._lookup.get_template('classifiers.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def datasets(self):
        template = self._lookup.get_template('datasets.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def users(self):
        template = self._lookup.get_template('users.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def jobs(self):
        template = self._lookup.get_template('jobs.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def about(self):
        template = self._lookup.get_template('about.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def version(self):
        return "0.4.1"


if __name__ == "__main__":
    with WebClient('sqlite:///%s' % 'test.db', '.', '/service/wdc', '/service/wcc', '/service/bp', 'test') as wcc:
        cherrypy.quickstart(wcc, '/', config=wcc.get_config())
