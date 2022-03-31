import os

import cherrypy
from mako.lookup import TemplateLookup

from ics.db import SQLAlchemyDB
import ics.apps.media as media

__author__ = 'Andrea Esuli'


class WebPublic(object):
    def __init__(self, db_connection_string, ip_auth_path, key_auth_path, classifier_path, name, main_path=None):
        self._db = SQLAlchemyDB(db_connection_string)
        self._media_dir = media.__path__[0]
        self._template_data = {'ip_auth_path': ip_auth_path,
                               'key_auth_path': key_auth_path,
                               'classifier_path': classifier_path,
                               'main_path': main_path,
                               'name': name,
                               'version': self.version(),
                               'base_template': 'public_basewithmenu.html'}
        self._lookup = TemplateLookup(os.path.join(self._media_dir, 'template'), input_encoding='utf-8',
                                      output_encoding='utf-8')

    def get_config(self):
        return {
            '/css':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': os.path.join(self._media_dir, 'css'),
                 'tools.icsauth.always_auth': True,
                 },
            '/js':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': os.path.join(self._media_dir, 'js'),
                 'tools.icsauth.always_auth': True,
                 },
        }

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @property
    def session_data(self):
        return {'ip': cherrypy.request.remote.ip, 'mount_dir': cherrypy.request.app.script_name}

    @cherrypy.expose
    def index(self):
        template = self._lookup.get_template('public_typeandcode.html')
        return template.render(**{**self._template_data, **self.session_data})

    @cherrypy.expose
    def about(self):
        template = self._lookup.get_template('about.html')
        return template.render(**{**self._template_data, **self.session_data})

    @cherrypy.expose
    def api(self):
        template = self._lookup.get_template('public_api.html')
        return template.render(**{**self._template_data, **self.session_data})

    @cherrypy.expose
    def version(self):
        import ics
        return ics.__version__

