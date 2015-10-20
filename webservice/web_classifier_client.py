import os
import cherrypy
from mako.template import Template
from mako.lookup import TemplateLookup

__author__ = 'Andrea Esuli'

MEDIA_DIR = os.path.join(os.path.abspath('.'), 'media')
lookup = TemplateLookup(directories=[MEDIA_DIR])

class WebClassifierClient(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return True

    @cherrypy.expose
    def index(self):
        template = lookup.get_template('typeandlabel.html')
        return template.render()

    @cherrypy.expose
    def typeandlabel(self):
        template = lookup.get_template('typeandlabel.html')
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
        return "0.1.0"

config = {'/media':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': MEDIA_DIR,
                }
        }

if __name__ == "__main__":
    with WebClassifierClient() as wcc:
        cherrypy.quickstart(wcc, '/app', config = config)
