import cherrypy

from db.sqlalchemydb import SQLAlchemyDB
from webservice.auth_controller_service import require
from webservice.user_controller_service import name_is

__author__ = 'Andrea Esuli'


class IPControllerService(object):
    def __init__(self, db_connection_string, default_hourly_limit=100, create_if_missing=False):
        self._db = SQLAlchemyDB(db_connection_string)
        self.default_hourly_limit = default_hourly_limit
        self.create_if_missing = create_if_missing

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
            ips = self._db.ipaddresses()
        else:
            ips = [cherrypy.request.remote.ip]
        ips = ips[page * page_count:(page + 1) * page_count]
        for ip in ips:
            ip_info = dict()
            ip_info['ip'] = ip
            ip_info['created'] = str(self._db.get_iptracker_creation_time(ip))
            ip_info['hourly_limit'] = str(self._db.get_iptracker_hourly_limit(ip))
            ip_info['total_request_counter'] = str(self._db.get_iptracker_total_request_counter(ip))
            ip_info['current_request_counter'] = str(self._db.get_iptracker_current_request_counter(ip))
            result.append(ip_info)
        return result

    def ip_rate_limit(self, cost=1):

        def check():
            ip = cherrypy.request.remote.ip
            try:
                return self._db.iptracker_check_and_count_request(ip, cost)
            except LookupError:
                if self.create_if_missing:
                    self._db.create_iptracker(ip, self.default_hourly_limit)

            try:
                return self._db.iptracker_check_and_count_request(ip, cost)
            except:
                return False

        return check

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def create(self, ip, hourly_limit):
        self._db.create_iptracker(ip, hourly_limit)
        return 'Ok'

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_hourly_limit(self, ip, hourly_limit):
        self._db.set_iptracker_hourly_limit(ip, hourly_limit)
        return 'Ok'

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_current_request_counter(self, ip, count=0):
        self._db.set_iptracker_current_request_counter(ip, count)
        return 'Ok'

    @cherrypy.expose
    def version(self):
        return "0.1.3 (db: %s)" % self._db.version()
