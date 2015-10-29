import os
import cherrypy
from mako.lookup import TemplateLookup
from db.sqlalchemydb import SQLAlchemyDB

__author__ = 'Andrea Esuli'

MEDIA_DIR = os.path.join(os.path.abspath('.'), 'media')
CSS_DIR = os.path.join(MEDIA_DIR, 'css')
JS_DIR = os.path.join(MEDIA_DIR, 'js')
TEMPLATE_DIR = os.path.join(MEDIA_DIR, 'template')
lookup = TemplateLookup(directories=[TEMPLATE_DIR])

class WebClassifierClient(object):
    def __init__(self, db_connection_string):
        self._db = SQLAlchemyDB(db_connection_string)

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


    @cherrypy.expose
    def index(self):
        template = lookup.get_template('datasets.html')
        return template.render()

    @cherrypy.expose
    def typeandlabel(self):
        template = lookup.get_template('typeandlabel.html')
        return template.render()

    @cherrypy.expose
    def browseandlabel(self,name=None):
        if name is None:
            raise cherrypy.HTTPRedirect('/datasets')

        if not self._db.dataset_exists(name):
            raise cherrypy.HTTPRedirect('/datasets')

        template = lookup.get_template('browseandlabel.html')
        return template.render()

    @cherrypy.expose
    def classifiers(self):
        template = lookup.get_template('classifiers.html')
        return template.render()

    @cherrypy.expose
    def datasets(self):
        template = lookup.get_template('datasets.html')
        return template.render()

    @cherrypy.expose
    def about(self):
        template = lookup.get_template('about.html')
        return template.render()

    @cherrypy.expose
    def version(self):
        return "0.2.0"

config = {
          '/css':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': CSS_DIR,
                },
          '/js':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': JS_DIR,
                },
        }

if __name__ == "__main__":
    with WebClassifierClient('sqlite:///%s' % 'test.db') as wcc:
        cherrypy.quickstart(wcc, '/app', config = config)
