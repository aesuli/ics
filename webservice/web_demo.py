import os

import cherrypy
from mako.lookup import TemplateLookup

from db.sqlalchemydb import SQLAlchemyDB

__author__ = 'Andrea Esuli'


class WebDemo(object):
    def __init__(self, db_connection_string, media_dir, ip_auth_path, classifier_path, name):
        self._db = SQLAlchemyDB(db_connection_string)
        self._media_dir = media_dir
        self._template_data = {'ip_auth_path': ip_auth_path,
                               'classifier_path': classifier_path,
                               'name': name,
                               'version': self.version(),
                               'base_template': 'demo_basewithmenu.html'}
        self._lookup = TemplateLookup(os.path.join(media_dir, 'template'), input_encoding='utf-8',
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

    def session_data(self):
        return {'ip': cherrypy.request.remote.ip, 'mount_dir': cherrypy.request.app.script_name,
                'rate': self._db.get_iptracker_hourly_limit(cherrypy.request.remote.ip)}

    @cherrypy.expose
    def index(self):
        template = self._lookup.get_template('demo_typeandlabel.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def about(self):
        template = self._lookup.get_template('about.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def api(self):
        template = self._lookup.get_template('demo_api.html')
        return template.render(**{**self._template_data, **self.session_data()})

    @cherrypy.expose
    def version(self):
        return "0.2.1"
