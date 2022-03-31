import cherrypy

from db.sqlalchemydb import SQLAlchemyDB
from webservice.auth_controller_service import require, SESSION_KEY

__author__ = 'Andrea Esuli'


def logged_in():
    def check():
        return cherrypy.request.login is not None

    return check


def name_is(required_username):
    def check():
        return required_username == cherrypy.request.login

    return check


class UserControllerService(object):
    def __init__(self, db_connection_string, min_password_length=8):
        self._db = SQLAlchemyDB(db_connection_string)
        self._min_password_length = min_password_length

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
    def info(self, page=0, page_size=50):
        result = []
        requester = cherrypy.request.login
        if requester is None:
            return result
        if requester == SQLAlchemyDB.admin_name():
            names = self._db.user_names()
        else:
            names = [requester]
        names = names[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
        for name in names:
            user_info = dict()
            user_info['name'] = name
            user_info['created'] = str(self._db.get_user_creation_time(name))
            user_info['updated'] = str(self._db.get_user_last_update_time(name))
            user_info['hourly_limit'] = str(self._db.get_user_hourly_limit(name))
            user_info['request_limit'] = str(self._db.get_user_request_limit(name))
            user_info['total_request_counter'] = str(self._db.get_user_total_request_counter(name))
            user_info['current_request_counter'] = str(self._db.get_user_current_request_counter(name))
            result.append(user_info)
        return result

    @cherrypy.expose
    def count(self):
        requester = cherrypy.request.login
        if requester is None:
            return '0'
        if requester == SQLAlchemyDB.admin_name():
            return str(len(list(self._db.user_names())))
        else:
            return '1'

    @cherrypy.expose
    def login(self, username=None, password=None):
        if username is None or password is None:
            cherrypy.response.status = 401
            return 'Wrong credentials'

        if self._db.verify_user(username, password):
            cherrypy.session[SESSION_KEY] = cherrypy.request.login = username
            cherrypy.log('LOGIN(username="' + username + '")')
            return 'Ok'
        else:
            cherrypy.response.status = 401
            cherrypy.log('REJECTED_LOGIN(username="' + username + '")')
            return 'Wrong credentials'

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def create_user(self, username, password):
        if len(password) < self._min_password_length:
            cherrypy.response.status = 400
            return "Password must be at least %i characters long" % self._min_password_length

        if self._db.user_exists(username):
            cherrypy.response.status = 403
            return 'User "%s" already exists' % username

        self._db.create_user(username, password)

    @cherrypy.expose
    def change_password(self, username, password):
        if not self._db.user_exists(username):
            cherrypy.response.status = 403
            return 'User "%s" does not exists' % username

        if len(password) < self._min_password_length:
            cherrypy.response.status = 400
            return "Password must be at least %i characters long" % self._min_password_length

        requester = cherrypy.request.login
        if requester != SQLAlchemyDB.admin_name() and requester != username:
            cherrypy.response.status = 403
            return 'Forbidden'

        self._db.change_password(username, password)
        return 'Ok'

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def delete_user(self, username):
        if username != SQLAlchemyDB.admin_name():
            self._db.delete_user(username)
        else:
            cherrypy.response.status = 403
            return 'Cannot delete "%s"' % username

    @cherrypy.expose
    def logout(self):
        sess = cherrypy.session
        username = sess.get(SESSION_KEY, None)
        sess[SESSION_KEY] = None
        if username:
            cherrypy.request.login = None
            cherrypy.log('LOGOUT(username="' + username + '")')

    def logged_in_with_cost(self, default_cost=1, cost_function=None):
        def check():
            name = cherrypy.request.login
            if name:
                if cost_function:
                    return self._db.user_check_and_count_request(name, cost_function(default_cost))
                else:
                    return self._db.user_check_and_count_request(name, default_cost)
            else:
                return False

        return check

    def logged_in_with_arg_len_cost(self, cost=1):
        def check():
            name = cherrypy.request.login
            if name:
                return self._db.user_check_and_count_request(name, cost)
            else:
                return False

        return check

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_hourly_limit(self, username, hourly_limit):
        if username != SQLAlchemyDB.admin_name():
            self._db.set_user_hourly_limit(username, int(hourly_limit))
            return 'Ok'
        else:
            cherrypy.response.status = 403
            return 'Cannot change "%s"' % username

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_request_limit(self, username, request_limit):
        if username != SQLAlchemyDB.admin_name():
            self._db.set_user_request_limit(username, int(request_limit))
            return 'Ok'
        else:
            cherrypy.response.status = 403
            return 'Cannot change "%s"' % username

    @cherrypy.expose
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_current_request_counter(self, username, count=0):
        self._db.set_user_current_request_counter(username, int(count))
        return 'Ok'

    @cherrypy.expose
    def version(self):
        return "1.2.1 (db: %s)" % self._db.version()
