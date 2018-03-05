import os
import sys

import cherrypy
from configargparse import ArgParser

from webservice.auth_controller_service import enable_controller_service, fail_with_error_message, redirect, any_of
from webservice.background_processor_service import BackgroundProcessor
from webservice.classifier_collection_service import ClassifierCollectionService
from webservice.dataset_collection_service import DatasetCollectionService
from webservice.ip_controller_service import IPControllerService
from webservice.jobs_service import JobsService
from webservice.key_controller_service import KeyControllerService
from webservice.user_controller_service import UserControllerService, must_be_logged_in
from webservice.web_client import WebClient
from webservice.web_demo import WebDemo

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
    parser.add_argument('--demo_path', help='server path of the web demo app', type=str, required=True)
    parser.add_argument('--ip_hourly_limit', help='ip hourly request limit', type=int, required=True)
    parser.add_argument('--allow_unknown_ips', help='allow unknown IPs with rate limit', type=bool, required=True)
    parser.add_argument('--user_auth_path', help='server path of the user auth web service', type=str, required=True)
    parser.add_argument('--ip_auth_path', help='server path of the ip auth web service', type=str, required=True)
    parser.add_argument('--key_auth_path', help='server path of the key auth web service', type=str, required=True)
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str, required=True)
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str, required=True)
    parser.add_argument('--jobs_path', help='server path of the jobs web service', type=str,
                        required=True)
    parser.add_argument('--min_password_length', help='minimum password length', type=int, required=True)
    args = parser.parse_args(sys.argv[1:])

    with BackgroundProcessor(args.db_connection_string) as background_processor, \
            WebClient(args.db_connection_string, args.media_dir, args.user_auth_path, args.classifier_path,
                      args.dataset_path, args.jobs_path, args.name) as client, \
            WebDemo(args.db_connection_string, args.media_dir, args.ip_auth_path, args.classifier_path,
                    args.name) as demo, \
            ClassifierCollectionService(args.db_connection_string, args.data_dir) as classifier_service, \
            DatasetCollectionService(args.db_connection_string, args.data_dir) as dataset_service, \
            JobsService(args.db_connection_string) as jobs_service, \
            UserControllerService(args.db_connection_string, args.min_password_length) as user_auth_controller, \
            IPControllerService(args.db_connection_string, args.ip_hourly_limit,
                                args.allow_unknown_ips) as ip_auth_controller, \
            KeyControllerService(args.db_connection_string) as key_auth_controller:
        background_processor.start()

        cherrypy.server.socket_host = args.host
        cherrypy.server.socket_port = args.port

        enable_controller_service()

        conf_demo = {
        }

        conf_client = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [any_of(must_be_logged_in(), redirect(args.client_path + '/login'))],
            },
            '/login': {
                'tools.icsauth.require': [],
            },
        }

        conf_generic_service_with_login = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [must_be_logged_in()],
            },
        }

        conf_classifier_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [any_of(must_be_logged_in(), fail_with_error_message(401, 'Not logged in.'))],
            },
            '/info': {
                'tools.icsauth.require': [],
            },
            '/classify': {
                'tools.icsauth.require': [any_of(must_be_logged_in(), key_auth_controller.has_key(),
                                                 ip_auth_controller.ip_rate_limit(),
                                                 fail_with_error_message(401, 'Reached request limit.'))],
            },
        }

        conf_ip_auth_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [must_be_logged_in()],
            },
            '/info': {
                'tools.icsauth.require': [],
            },
        }

        conf_key_auth_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [must_be_logged_in()],
            },
            '/info': {
                'tools.icsauth.require': [],
            },
        }

        conf_user_auth_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [must_be_logged_in()],
            },
            '/login': {
                'tools.icsauth.require': [],
            },
            '/info': {
                'tools.icsauth.require': [],
            },
        }

        cherrypy.config.update({'environment': 'production', 'log.error_file': 'site.log', })

        cherrypy.tree.mount(demo, args.demo_path, config={**demo.get_config(), **conf_demo})
        cherrypy.tree.mount(client, args.client_path, config={**client.get_config(), **conf_client})
        cherrypy.tree.mount(classifier_service, args.classifier_path, config=conf_classifier_service)
        cherrypy.tree.mount(dataset_service, args.dataset_path, config=conf_generic_service_with_login)
        cherrypy.tree.mount(user_auth_controller, args.user_auth_path, config=conf_user_auth_service)
        cherrypy.tree.mount(ip_auth_controller, args.ip_auth_path, config=conf_ip_auth_service)
        cherrypy.tree.mount(key_auth_controller, args.key_auth_path, config=conf_key_auth_service)
        cherrypy.tree.mount(jobs_service, args.jobs_path, config=conf_generic_service_with_login)

        cherrypy.engine.subscribe('stop', background_processor.stop)

        cherrypy.engine.start()
        cherrypy.engine.block()
