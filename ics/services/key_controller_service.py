import cherrypy

from ics.db.sqlalchemydb import SQLAlchemyDB
from ics.services.auth_controller_service import require
from ics.services.user_controller_service import name_is

__author__ = 'Andrea Esuli'


class KeyControllerService(object):
    def __init__(self, db):
        self._db = db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def info(self, page=None, page_size=50):
        result = []
        requester = cherrypy.request.login
        if requester is not None and requester == SQLAlchemyDB.admin_name():
            keys = self._db.keys()
        else:
            key = cherrypy.request.key
            if key:
                keys = [key]
            else:
                keys = []
        if page is not None:
            keys = keys[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
        for key in keys:
            key_info = dict()
            key_info['key'] = key
            key_info['name'] = str(self._db.get_keytracker_name(key))
            key_info['created'] = str(self._db.get_keytracker_creation_time(key))
            key_info['updated'] = str(self._db.get_keytracker_last_update_time(key))
            key_info['hourly_limit'] = str(self._db.get_keytracker_hourly_limit(key))
            key_info['request_limit'] = str(self._db.get_keytracker_request_limit(key))
            key_info['total_request_counter'] = str(self._db.get_keytracker_total_request_counter(key))
            key_info['current_request_counter'] = str(self._db.get_keytracker_current_request_counter(key))
            result.append(key_info)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def count(self):
        requester = cherrypy.request.login
        if requester is not None and requester == SQLAlchemyDB.admin_name():
            keys = self._db.keys()
        else:
            key = cherrypy.request.params.get('authkey', None)
            if key:
                keys = [key]
            else:
                keys = []
        return str(len(list(keys)))

    def has_key(self, default_cost=1, cost_function=None):

        def check():
            key = cherrypy.request.params.get('authkey', None)
            if key:
                try:
                    del cherrypy.request.params['authkey']
                except AttributeError:
                    pass
                cherrypy.request.key = key
                if cost_function:
                    return self._db.keytracker_check_and_count_request(key, cost_function(default_cost))
                else:
                    return self._db.keytracker_check_and_count_request(key, default_cost)
            else:
                try:
                    del cherrypy.request.key
                except AttributeError:
                    pass
                return False

        return check

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def create(self, name, hourly_limit, request_limit):
        name = name.strip()
        if len(name) < 1:
            cherrypy.response.status = 400
            return 'Cannot use an empty name'
        key = self._db.create_keytracker(name, int(hourly_limit), int(request_limit))
        return key

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def delete(self, key):
        self._db.delete_keytracker(key)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_hourly_limit(self, key, hourly_limit):
        self._db.set_keytracker_hourly_limit(key, int(hourly_limit))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_request_limit(self, key, request_limit):
        self._db.set_keytracker_request_limit(key, int(request_limit))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_current_request_counter(self, key, count=0):
        self._db.set_keytracker_current_request_counter(key, int(count))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def version(self):
        import ics
        return ics.__version__
