import cherrypy

from db.sqlalchemydb import SQLAlchemyDB
from webservice.auth_controller_service import require
from webservice.user_controller_service import name_is

__author__ = 'Andrea Esuli'


class KeyControllerService(object):
    def __init__(self, db_connection_string):
        self._db = SQLAlchemyDB(db_connection_string)

    def close(self):
        self._db.close()
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def info(self, page=0, page_count=50):
        result = []
        requester = cherrypy.request.login
        if requester is not None and requester == SQLAlchemyDB.admin_name():
            keys = self._db.keys()
        else:
            key = cherrypy.request.body.params.get('authkey', None)
            if key:
                keys = [key]
            else:
                keys = []
        keys = keys[page * page_count:(page + 1) * page_count]
        for key in keys:
            key_info = dict()
            key_info['key'] = key
            key_info['name'] = str(self._db.get_keytracker_name(key))
            key_info['created'] = str(self._db.get_keytracker_creation_time(key))
            key_info['hourly_limit'] = str(self._db.get_keytracker_hourly_limit(key))
            key_info['request_limit'] = str(self._db.get_keytracker_request_limit(key))
            key_info['total_request_counter'] = str(self._db.get_keytracker_total_request_counter(key))
            key_info['current_request_counter'] = str(self._db.get_keytracker_current_request_counter(key))
            result.append(key_info)
        return result

    def has_key(self, cost=1):

        def check():
            key = cherrypy.request.body.params.get('authkey',None)
            if key:
                return self._db.keytracker_check_and_count_request(key, cost)
            else:
                return False

        return check

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def create(self, name, hourly_limit, request_limit):
        key = self._db.create_keytracker(name, hourly_limit, request_limit)
        return key

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_hourly_limit(self, key, hourly_limit):
        self._db.set_keytracker_hourly_limit(key, hourly_limit)
        return 'Ok'

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_request_limit(self, key, request_limit):
        self._db.set_keytracker_request_limit(key, request_limit)
        return 'Ok'

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_current_request_counter(self, key, count=0):
        self._db.set_keytracker_current_request_counter(key, count)
        return 'Ok'

    @cherrypy.expose
    def version(self):
        return "0.1.3 (db: %s)" % self._db.version()
