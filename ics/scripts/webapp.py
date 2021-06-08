import logging
import multiprocessing
import os
import sys
from logging import handlers

import cherrypy
from configargparse import ArgParser

from ics.apps import WebAdmin
from ics.apps import WebApp
from ics.apps import WebPublic
from ics.db.sqlalchemydb import SQLAlchemyDB
from ics.services import BackgroundProcessor
from ics.services import ClassifierCollectionService
from ics.services import DatasetCollectionService
from ics.services import IPControllerService
from ics.services import JobsService
from ics.services import KeyControllerService
from ics.services import UserControllerService
from ics.services.auth_controller_service import enable_controller_service, any_of, redirect, fail_with_error_message, \
    arg_len_cost_function
from ics.services.user_controller_service import logged_in, name_is
from ics.util.util import str_to_bool

__author__ = 'Andrea Esuli'


def setup_log(access_filename, app_filename):
    path = os.path.dirname(access_filename)
    os.makedirs(path, exist_ok=True)
    path = os.path.dirname(app_filename)
    os.makedirs(path, exist_ok=True)

    cherrypy.config.update({  # 'environment': 'production',
        'log.error_file': '',
        'log.access_file': ''})

    error_handler = handlers.TimedRotatingFileHandler(app_filename + '.log', when='midnight')
    error_handler.setLevel(logging.DEBUG)
    cherrypy.log.error_log.addHandler(error_handler)

    access_handler = handlers.TimedRotatingFileHandler(access_filename + '.log', when='midnight')
    access_handler.setLevel(logging.DEBUG)
    cherrypy.log.access_log.addHandler(access_handler)


def setup_background_processor_log(access_filename, app_filename):
    path = os.path.dirname(access_filename)
    os.makedirs(path, exist_ok=True)
    path = os.path.dirname(app_filename)
    os.makedirs(path, exist_ok=True)

    cherrypy.config.update({  # 'environment': 'production',
        'log.error_file': '',
        'log.access_file': ''})

    process = multiprocessing.current_process()

    error_handler = handlers.TimedRotatingFileHandler(app_filename + '-' + str(process.name) + '.log',
                                                      when='midnight')
    error_handler.setLevel(logging.DEBUG)
    cherrypy.log.error_log.addHandler(error_handler)

    access_handler = handlers.TimedRotatingFileHandler(access_filename + '-' + str(process.name) + '.log',
                                                       when='midnight')
    access_handler.setLevel(logging.DEBUG)
    cherrypy.log.access_log.addHandler(access_handler)


def main():
    parser = ArgParser()
    parser.add_argument('-c', '--config', help='configuration file', is_config_file=True)
    parser.add_argument('--db_connection_string', type=str, default='postgresql://ics:ics@localhost:5432/ics')
    parser.add_argument('--log_dir', help='local directory for log files', type=str, default='log')
    parser.add_argument('--data_dir', help='local directory where uploaded/downloaded file are placed', type=str,
                        default=os.path.join(os.getcwd(), 'data'))
    parser.add_argument('--name', help='name to show in the client app', type=str,
                        default='ICS - Interactive Classification System')
    parser.add_argument('--host', help='host server address', type=str, default='127.0.0.1')
    parser.add_argument('--port', help='host server port', type=int, default=8080)
    parser.add_argument('--main_app_path', help='server path of the web client app', type=str, default='/')
    parser.add_argument('--admin_app_path', help='server path of the web admin app', type=str, default='/admin/')
    parser.add_argument('--public_app_path', help='server path of the web public app', type=str, default='/public/')
    parser.add_argument('--ip_hourly_limit', help='ip hourly request limit', type=int, default=100)
    parser.add_argument('--ip_request_limit', help='ip total request limit', type=int, default=-1)
    parser.add_argument('--allow_unknown_ips', help='allow unknown IPs with rate limit', type=str_to_bool,
                        default=True)
    parser.add_argument('--user_auth_path', help='server path of the user auth web service', type=str,
                        default='/service/userauth/')
    parser.add_argument('--ip_auth_path', help='server path of the ip auth web service', type=str,
                        default='/service/ipauth/')
    parser.add_argument('--key_auth_path', help='server path of the key auth web service', type=str,
                        default='/service/keyauth/')
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str,
                        default='/service/classifiers/')
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str,
                        default='/service/datasets/')
    parser.add_argument('--jobs_path', help='server path of the jobs web service', type=str,
                        default='/service/jobs/')
    parser.add_argument('--min_password_length', help='minimum password length', type=int, default=8)
    args = parser.parse_args(sys.argv[1:])

    setup_log(os.path.join(args.log_dir, 'access'), os.path.join(args.log_dir, 'app'))
    with BackgroundProcessor(args.db_connection_string, os.cpu_count() - 2, initializer=setup_background_processor_log,
                             initargs=(os.path.join(args.log_dir, 'bpaccess'),
                                       os.path.join(args.log_dir, 'bpapp'))) as background_processor, \
            WebApp(args.db_connection_string, args.user_auth_path, args.admin_app_path,
                   args.classifier_path, args.dataset_path, args.jobs_path, args.name, args.public_app_path) as main_app, \
            WebPublic(args.db_connection_string, args.ip_auth_path, args.key_auth_path,
                    args.classifier_path, args.name, args.main_app_path) as public_app, \
            WebAdmin(args.db_connection_string, args.main_app_path, args.user_auth_path,
                     args.ip_auth_path, args.key_auth_path, args.classifier_path, args.dataset_path, args.jobs_path,
                     args.name) as admin_app, \
            ClassifierCollectionService(args.db_connection_string, args.data_dir) as classifier_service, \
            DatasetCollectionService(args.db_connection_string, args.data_dir) as dataset_service, \
            JobsService(args.db_connection_string) as jobs_service, \
            UserControllerService(args.db_connection_string, args.min_password_length) as user_auth_controller, \
            IPControllerService(args.db_connection_string, args.ip_hourly_limit, args.ip_request_limit,
                                args.allow_unknown_ips) as ip_auth_controller, \
            KeyControllerService(args.db_connection_string) as key_auth_controller:
        background_processor.start()

        cherrypy.server.socket_host = args.host
        cherrypy.server.socket_port = args.port

        enable_controller_service()

        conf_public_app = {
        }

        conf_main_app = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [any_of(logged_in(), redirect(args.main_app_path + 'login'))],
            },
            '/login': {
                'tools.icsauth.require': [],
            },
        }

        conf_admin_app = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [any_of(name_is(SQLAlchemyDB.admin_name()),
                                                 redirect(args.admin_app_path + 'login'))],
            },
            '/login': {
                'tools.icsauth.require': [],
            },
        }

        conf_generic_service_with_login = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [logged_in()],
            },
        }

        conf_classifier_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [any_of(logged_in(), fail_with_error_message(401, 'Not logged in.'))],
            },
            '/info': {
                'tools.icsauth.require': [],
            },
            '/classify': {
                'tools.icsauth.require': [
                    any_of(user_auth_controller.logged_in_with_cost(cost_function=arg_len_cost_function('X')),
                           key_auth_controller.has_key(cost_function=arg_len_cost_function('X')),
                           ip_auth_controller.ip_rate_limit(cost_function=arg_len_cost_function('X')),
                           fail_with_error_message(401, 'Reached request limit.'))],
            },
        }

        conf_ip_auth_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [logged_in()],
            },
            '/info': {
                'tools.icsauth.require': [],
            },
        }

        conf_key_auth_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [logged_in()],
            },
            '/info': {
                'tools.icsauth.require': [any_of(logged_in(), key_auth_controller.has_key(default_cost=0))],
            },
        }

        conf_user_auth_service = {
            '/': {
                'tools.sessions.on': True,
                'tools.icsauth.on': True,
                'tools.icsauth.require': [logged_in()],
            },
            '/login': {
                'tools.icsauth.require': [],
            },
        }

        cherrypy.tree.mount(public_app, args.public_app_path, config={**public_app.get_config(), **conf_public_app})
        cherrypy.tree.mount(main_app, args.main_app_path, config={**main_app.get_config(), **conf_main_app})
        cherrypy.tree.mount(admin_app, args.admin_app_path, config={**admin_app.get_config(), **conf_admin_app})
        cherrypy.tree.mount(classifier_service, args.classifier_path, config=conf_classifier_service)
        cherrypy.tree.mount(dataset_service, args.dataset_path, config=conf_generic_service_with_login)
        cherrypy.tree.mount(user_auth_controller, args.user_auth_path, config=conf_user_auth_service)
        cherrypy.tree.mount(ip_auth_controller, args.ip_auth_path, config=conf_ip_auth_service)
        cherrypy.tree.mount(key_auth_controller, args.key_auth_path, config=conf_key_auth_service)
        cherrypy.tree.mount(jobs_service, args.jobs_path, config=conf_generic_service_with_login)

        cherrypy.engine.subscribe('stop', background_processor.stop)

        cherrypy.engine.start()
        cherrypy.engine.block()


if __name__ == "__main__":
    main()
