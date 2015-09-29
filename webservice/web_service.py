import cherrypy
import webservice.web_classifier_client as client
from webservice.web_classifier_client import WebClassifierClient
from webservice.web_classifier_collection import WebClassifierCollection

__author__ = 'Andrea Esuli'

if __name__ == "__main__":
    # with WebClassifierClient() as app, WebClassifierCollection(os.path.join(os.path.curdir, 'wcc.db')) as service:
    with WebClassifierClient() as app, WebClassifierCollection('wcc') as service:
        cherrypy.tree.mount(app, '/app', config=client.config)
        cherrypy.quickstart(service, '/service')
        cherrypy.server.start()