import os

import cherrypy
from mako.lookup import TemplateLookup

from db.sqlalchemydb import SQLAlchemyDB

__author__ = 'Andrea Esuli'


class WebAdmin(object):
    def __init__(self, db_connection_string, media_dir, client_path, user_auth_path, ip_auth_path, key_auth_path,
                 classifier_path,
                 dataset_path, jobs_path, name):
        self._db = SQLAlchemyDB(db_connection_string)
        self._media_dir = media_dir
        self._template_data = {'client_path': client_path,
                               'user_auth_path': user_auth_path,
                               'ip_auth_path': ip_auth_path,
                               'key_auth_path': key_auth_path,
                               'classifier_path': classifier_path,
                               'dataset_path': dataset_path,
                               'jobs_path': jobs_path,
                               'name': name,
                               'version': self.version(),
                               'base_template': 'admin_basewithmenu.html'}
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
        return self.users()

    @cherrypy.expose
    def users(self):
        template = self._lookup.get_template('users.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def ips(self):
        template = self._lookup.get_template('ips.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def keys(self):
        template = self._lookup.get_template('keys.html')
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
        return "0.2.1"
