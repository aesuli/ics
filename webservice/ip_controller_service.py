import cherrypy

from db.sqlalchemydb import SQLAlchemyDB
from webservice.auth_controller_service import SESSION_KEY, require, name_is


def ip_rate_limit():
    def check():
        ip = cherrypy.request.remote.ip
        print(ip, type(ip))
        return False

    return check


class IpControllerService(object):
    def __init__(self, db_connection_string, default_max_request_rate = 100, default_rate_time_interval='day'):
        self._db = SQLAlchemyDB(db_connection_string)
        self._default_max_request_rate = default_max_request_rate
        self._default_rate_time_interval = default_rate_time_interval

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
    def index(self):
        result = []
        requester = cherrypy.session[SESSION_KEY]
        if requester is None:
            return result
        # TODO return rates and limits for current ip
        # or for all ips if admin
        if requester == SQLAlchemyDB.admin_name():
            names = self._db.user_names()
        else:
            names = [requester]
        for name in names:
            user_info = dict()
            user_info['name'] = name
            user_info['created'] = str(self._db.get_user_creation_time(name))
            result.append(user_info)
        return result

    @cherrypy.expose
    def whoami(self):
        return cherrypy.request.remote.ip

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def change_ip_rate(self, ip, rate, rate_unit):
        # TODO set ip rate
        self._db.change_password(username, password)
        return 'Ok'

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_ip_counter(self, ip, count):
        # TODO reset ip counter
        self._db.change_password(username, password)
        return 'Ok'


    @cherrypy.expose
    def version(self):
        return "0.0.1 (db: %s)" % self._db.version()
