import cherrypy
from webservice.background_processor import BackgroundProcessor
import webservice.web_classifier_client as client
from webservice.web_classifier_client import WebClassifierClient
from webservice.web_classifier_collection import WebClassifierCollection
from webservice.web_dataset_collection import WebDatasetCollection

__author__ = 'Andrea Esuli'

if __name__ == "__main__":
    db_connection_string = 'postgresql://wcc:wcc@localhost:5432/wcc'
    with BackgroundProcessor() as background_processor, WebClassifierClient(
            db_connection_string) as app, WebClassifierCollection(
            db_connection_string, background_processor) as classifier_service, WebDatasetCollection(
            db_connection_string, background_processor) as dataset_service:
        background_processor.start()
        cherrypy.tree.mount(app, '/', config=client.config)
        cherrypy.tree.mount(classifier_service, '/service/classifiers')
        cherrypy.tree.mount(dataset_service, '/service/datasets')
        cherrypy.engine.start()
        cherrypy.engine.block()