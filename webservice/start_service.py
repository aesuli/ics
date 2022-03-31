import os
import sys

import cherrypy
from configargparse import ArgParser

from webservice.auth_controller_service import AuthControllerService, SESSION_KEY
from webservice.background_processor_service import BackgroundProcessor
from webservice.classifier_collection_service import ClassifierCollectionService
from webservice.web_client import WebClient
from webservice.dataset_collection_service import DatasetCollectionService

__author__ = 'Andrea Esuli'

if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument('-c', '--config', help='configuration file', is_config_file=True)
    parser.add_argument('--db_connection_string', type=str, required=True)
    parser.add_argument('--data_dir', help='local directory where uploaded/downloaded file are placed', type=str,
                        default=os.path.join(os.getcwd(), './webservice/data'))
    parser.add_argument('--name', help='name to show in the client app', type=str, required=True)
    parser.add_argument('--host', help='host server address', type=str, required=True)
    parser.add_argument('--port', help='host server port', type=int, required=True)
    parser.add_argument('--client_path')
    parser.add_argument('--auth_path', help='server path of the auth web service', type=str, required=True)
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str, required=True)
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str, required=True)
    parser.add_argument('--processor_path', help='server path of the background processor web service', type=str,
                        required=True)
    parser.add_argument('--min_password_length', help='minimum password length', type=int, required=True)
    args = parser.parse_args(sys.argv[1:])

    with BackgroundProcessor(args.db_connection_string) as background_processor, \
            ClassifierCollectionService(args.db_connection_string, args.data_dir,
                                        background_processor) as classifier_service, \
            DatasetCollectionService(args.db_connection_string, args.data_dir, background_processor) as dataset_service, \
            AuthControllerService(args.db_connection_string, args.min_password_length) as auth_controller:
        background_processor.start()

        cherrypy.server.socket_host = args.host
        cherrypy.server.socket_port = args.port

        def must_be_logged_in():
            return cherrypy.session.get(SESSION_KEY) is not None

        conf_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [must_be_logged_in],
            },
        }

        cherrypy.tree.mount(classifier_service, args.classifier_path, config=conf_service)
        cherrypy.tree.mount(dataset_service, args.dataset_path, config=conf_service)
        cherrypy.tree.mount(auth_controller, args.auth_path, config=conf_service)
        cherrypy.tree.mount(background_processor, args.processor_path, config=conf_service)

        cherrypy.engine.subscribe('stop', background_processor.stop)

        cherrypy.engine.start()
        cherrypy.engine.block()
