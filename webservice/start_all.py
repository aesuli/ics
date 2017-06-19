import sys

import cherrypy
from configargparse import ArgParser

import webservice.web_client as client
from webservice.background_processor import BackgroundProcessor
from webservice.web_client import WebClient
from webservice.web_classifier_collection import WebClassifierCollection
from webservice.web_dataset_collection import WebDatasetCollection

__author__ = 'Andrea Esuli'

if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument('-c', '--config', help='configuration file', is_config_file=True)
    parser.add_argument('--db_connection_string', required=True)
    parser.add_argument('--app_path', help='server path of the web client app', type=str, default='/')
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str,
                        default='/service/classifiers')
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str,
                        default='/service/datasets')
    parser.add_argument('--processor_path', help='server path of the background processor web service', type=str,
                        default='/service/jobs')
    args = parser.parse_args(sys.argv[1:])

    with BackgroundProcessor(args.db_connection_string) as background_processor, WebClient(
            args.db_connection_string) as app, WebClassifierCollection(
        args.db_connection_string, background_processor) as classifier_service, WebDatasetCollection(
        args.db_connection_string, background_processor) as dataset_service:
        background_processor.start()
        cherrypy.tree.mount(app, args.app_path, config=client.config)
        cherrypy.tree.mount(classifier_service, args.classifier_path)
        cherrypy.tree.mount(dataset_service, args.dataset_path)
        cherrypy.tree.mount(background_processor, args.processor_path)
        cherrypy.engine.subscribe('stop', background_processor.stop)
        cherrypy.engine.start()
        cherrypy.engine.block()
