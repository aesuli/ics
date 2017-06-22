import os
import sys

import cherrypy
from configargparse import ArgParser

from webservice.background_processor import BackgroundProcessor
from webservice.web_classifier_collection import WebClassifierCollection
from webservice.web_client import WebClient
from webservice.web_dataset_collection import WebDatasetCollection

__author__ = 'Andrea Esuli'

if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument('-c', '--config', help='configuration file', is_config_file=True)
    parser.add_argument('--db_connection_string', type=str, required=True)
    parser.add_argument('--media_dir', help='local directory with static files (html templates, css, js)', type=str,
                        default=os.path.join(os.getcwd(), './webservice/media'))
    parser.add_argument('--data_dir', help='local directory where uploaded/downloaded file are placed', type=str,
                        default=os.path.join(os.getcwd(), './webservice/data'))
    parser.add_argument('--name', help='name to show in the client app', type=str, required=True)
    parser.add_argument('--host', help='host server address', type=str, required=True)
    parser.add_argument('--port', help='host server port', type=int, required=True)
    parser.add_argument('--client_path', help='server path of the web client app', type=str, required=True)
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str, required=True)
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str, required=True)
    parser.add_argument('--processor_path', help='server path of the background processor web service', type=str,
                        required=True)
    args = parser.parse_args(sys.argv[1:])

    with BackgroundProcessor(args.db_connection_string) as background_processor, \
            WebClient(args.db_connection_string, args.media_dir, args.classifier_path, args.dataset_path,
                      args.processor_path, args.name) as client, \
            WebClassifierCollection(args.db_connection_string, args.data_dir,
                                    background_processor) as classifier_service, \
            WebDatasetCollection(args.db_connection_string, args.data_dir, background_processor) as dataset_service:
        background_processor.start()

        cherrypy.server.socket_host = args.host
        cherrypy.server.socket_port = args.port

        cherrypy.tree.mount(client, args.client_path, config=client.get_config())
        cherrypy.tree.mount(classifier_service, args.classifier_path)
        cherrypy.tree.mount(dataset_service, args.dataset_path)
        cherrypy.tree.mount(background_processor, args.processor_path)

        cherrypy.engine.subscribe('stop', background_processor.stop)

        cherrypy.engine.start()
        cherrypy.engine.block()
