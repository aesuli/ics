import os

import cherrypy
from mako.lookup import TemplateLookup

from ics.db import SQLAlchemyDB

__author__ = 'Andrea Esuli'


class WebApp(object):
    def __init__(self, db_connection_string, media_dir, user_auth_path, admin_path, classifier_path,
                 dataset_path, jobs_path, name):
        self._db = SQLAlchemyDB(db_connection_string)
        self._media_dir = media_dir
        self._template_data = {'user_auth_path': user_auth_path,
                               'admin_path': admin_path,
                               'classifier_path': classifier_path,
                               'dataset_path': dataset_path,
                               'jobs_path': jobs_path,
                               'name': name,
                               'version': self.version()}
        self._lookup = TemplateLookup(os.path.join(media_dir, 'template'), input_encoding='utf-8',
                                      output_encoding='utf-8')

    def get_config(self):
        return {
            '/css':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': os.path.join(self._media_dir, 'css'),
                 'tools.icsauth.on': False,
                 },
            '/js':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': os.path.join(self._media_dir, 'js'),
                 'tools.icsauth.on': False,
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
        return {'username': cherrypy.request.login, 'mount_dir': cherrypy.request.app.script_name}

    @cherrypy.expose
    def login(self, name=None, password=None, error_message=None):
        if name is None or password is None:
            template = self._lookup.get_template('login.html')
            if name is None:
                name = ""
            if error_message is None:
                error_message = ""
            return template.render(
                **{**self._template_data, **self.session_data(), **{'username': name, 'msg': error_message}})

    @cherrypy.expose
    def index(self):
        return self.datasets()

    @cherrypy.expose
    def typeandcode(self):
        template = self._lookup.get_template('typeandcode.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def browseandcode(self, name=None):
        if name is None:
            raise cherrypy.HTTPRedirect(self._template_data()['dataset_path'])

        if not self._db.dataset_exists(name):
            raise cherrypy.HTTPRedirect(self._template_data()['dataset_path'])

        template = self._lookup.get_template('browseandcode.html')
        return template.render(**{**self._template_data, **self.session_data(),
                                  **{'datasetname': name}})

    @cherrypy.expose
    def classify(self, name=None):
        if name is None:
            raise cherrypy.HTTPRedirect(self._template_data()['dataset_path'])

        if not self._db.dataset_exists(name):
            raise cherrypy.HTTPRedirect(self._template_data()['dataset_path'])

        template = self._lookup.get_template('classify.html')
        return template.render(**{**self._template_data, **self.session_data(),
                                  **{'datasetname': name}})

    @cherrypy.expose
    def classification_view(self, datasetname=None, classifiername=None):
        if datasetname is None or not self._db.dataset_exists(
                datasetname) or classifiername is None or not self._db.classifier_exists(classifiername):
            raise cherrypy.HTTPRedirect(self._template_data()['dataset_path'])

        template = self._lookup.get_template('classification_view.html')
        return template.render(**{**self._template_data, **self.session_data(),
                                  **{'datasetname': datasetname, 'classifiername': classifiername}})

    @cherrypy.expose
    def training_data_view(self, classifiername=None):
        if classifiername is None or not self._db.classifier_exists(classifiername):
            raise cherrypy.HTTPRedirect(self._template_data()['classifier_path'])

        template = self._lookup.get_template('training_data_view.html')
        return template.render(**{**self._template_data, **self.session_data(),
                                  **{'classifiername': classifiername}})

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
    def locks(self):
        template = self._lookup.get_template('locks.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def about(self):
        template = self._lookup.get_template('about.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def version(self):
        import ics
        return ics.__version__
