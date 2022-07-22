import ipaddress

import cherrypy

from ics.db.sqlalchemydb import SQLAlchemyDB
from ics.services.auth_controller_service import require
from ics.services.user_controller_service import name_is

__author__ = 'Andrea Esuli'


class IPControllerService(object):
    def __init__(self, db, default_hourly_limit=100, default_request_limit=10000,
                 create_if_missing=False):
        self._db = db
        self.default_hourly_limit = default_hourly_limit
        self.default_request_limit = default_request_limit
        self.create_if_missing = create_if_missing

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
            ips = self._db.ipaddresses()
        else:
            if cherrypy.request.remote.ip in self._db.ipaddresses():
                ips = [cherrypy.request.remote.ip]
            else:
                ips = []
        if page is not None:
            ips = ips[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
        for ip in ips:
            ip_info = dict()
            ip_info['ip'] = ip
            ip_info['created'] = str(self._db.get_iptracker_creation_time(ip))
            ip_info['updated'] = str(self._db.get_iptracker_last_update_time(ip))
            ip_info['hourly_limit'] = str(self._db.get_iptracker_hourly_limit(ip))
            ip_info['request_limit'] = str(self._db.get_iptracker_request_limit(ip))
            ip_info['total_request_counter'] = str(self._db.get_iptracker_total_request_counter(ip))
            ip_info['current_request_counter'] = str(self._db.get_iptracker_current_request_counter(ip))
            result.append(ip_info)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def count(self):
        requester = cherrypy.request.login
        if requester is not None and requester == SQLAlchemyDB.admin_name():
            ips = self._db.ipaddresses()
        else:
            if cherrypy.request.remote.ip in self._db.ipaddresses():
                ips = [cherrypy.request.remote.ip]
            else:
                ips = []
        return str(len(list(ips)))

    def ip_rate_limit(self, default_cost=1, cost_function=None):

        def check():
            ip = cherrypy.request.remote.ip
            try:
                if cost_function:
                    return self._db.iptracker_check_and_count_request(ip, cost_function(default_cost))
                else:
                    return self._db.iptracker_check_and_count_request(ip, default_cost)
            except LookupError:
                if self.create_if_missing:
                    self._db.create_iptracker(ip, self.default_hourly_limit, self.default_request_limit)

            try:
                if cost_function:
                    return self._db.iptracker_check_and_count_request(ip, cost_function(default_cost))
                else:
                    return self._db.iptracker_check_and_count_request(ip, default_cost)
            except:
                return False

        return check

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def create(self, ip, hourly_limit, request_limit):
        try:
            ipaddress.ip_address(ip)
        except:
            cherrypy.response.status = 400
            return 'Not an IP'
        self._db.create_iptracker(ip, int(hourly_limit), int(request_limit))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def delete(self, ip):
        self._db.delete_iptracker(ip)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_hourly_limit(self, ip, hourly_limit):
        self._db.set_iptracker_hourly_limit(ip, int(hourly_limit))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_request_limit(self, ip, request_limit):
        self._db.set_iptracker_request_limit(ip, int(request_limit))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_current_request_counter(self, ip, count=0):
        self._db.set_iptracker_current_request_counter(ip, int(count))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def version(self):
        import ics
        return ics.__version__
