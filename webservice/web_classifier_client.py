import os
import cherrypy

__author__ = 'Andrea Esuli'

MEDIA_DIR = os.path.join(os.path.abspath('.'), 'media')

class WebClassifierClient(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return True

    @cherrypy.expose
    def index(self):
        return open(os.path.join(MEDIA_DIR, 'index.html'))

    @cherrypy.expose
    def scores(self):
        return open(os.path.join(MEDIA_DIR, 'scores.html'))

    @cherrypy.expose
    def about(self):
        return open(os.path.join(MEDIA_DIR, 'about.html'))

    @cherrypy.expose
    def version(self):
        return "0.0.1"

config = {'/media':
                {'tools.staticdir.on': True,
                 'tools.staticdir.dir': MEDIA_DIR,
                }
        }

if __name__ == "__main__":
    with WebClassifierClient() as wcc:
        cherrypy.quickstart(wcc, '/app', config = config)
