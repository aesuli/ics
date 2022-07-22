import cherrypy

from ics.db.sqlalchemydb import SQLAlchemyDB
from ics.services.auth_controller_service import USER_SESSION_KEY, require

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
    def __init__(self, db, min_password_length=8):
        self._db = db
        self._min_password_length = min_password_length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def info(self, page=None, page_size=50):
        result = []
        requester = cherrypy.request.login
        if requester is None:
            return result
        if requester == SQLAlchemyDB.admin_name():
            names = self._db.user_names()
        else:
            names = [requester]
        if page is not None:
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
    @cherrypy.tools.json_out()
    def count(self):
        requester = cherrypy.request.login
        if requester is None:
            return '0'
        if requester == SQLAlchemyDB.admin_name():
            return str(len(list(self._db.user_names())))
        else:
            return '1'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def login(self, name=None, password=None):
        if name is None or password is None:
            cherrypy.response.status = 401
            return 'Wrong credentials'

        if self._db.verify_user(name, password):
            cherrypy.session[USER_SESSION_KEY] = cherrypy.request.login = name
            cherrypy.log('LOGIN(username="' + name + '")')
            return 'Ok'
        else:
            cherrypy.response.status = 401
            cherrypy.log('REJECTED_LOGIN(username="' + name + '")')
            return 'Wrong credentials'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def create(self, name, password):
        if len(password) < self._min_password_length:
            cherrypy.response.status = 400
            return f'Password must be at least {self._min_password_length} characters long'

        if self._db.user_exists(name):
            cherrypy.response.status = 403
            return f'User "{name}" already exists'

        try:
            self._db.create_user(name, password)
        except ValueError as ve:
            cherrypy.response.status = 400
            return f'Error: {ve}'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def change_password(self, name, password):
        if not self._db.user_exists(name):
            cherrypy.response.status = 403
            return 'User "%s" does not exist' % name

        if len(password) < self._min_password_length:
            cherrypy.response.status = 400
            return "Password must be at least %i characters long" % self._min_password_length

        requester = cherrypy.request.login
        if requester != SQLAlchemyDB.admin_name() and requester != name:
            cherrypy.response.status = 403
            return 'Forbidden'

        self._db.change_password(name, password)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def delete(self, name):
        if name != SQLAlchemyDB.admin_name():
            self._db.delete_user(name)
        else:
            cherrypy.response.status = 403
            return 'Cannot delete "%s"' % name
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def logout(self):
        sess = cherrypy.session
        username = sess.get(USER_SESSION_KEY, None)
        sess[USER_SESSION_KEY] = None
        if username:
            cherrypy.request.login = None
            cherrypy.log('LOGOUT(username="' + username + '")')
        return 'Ok'

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
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_hourly_limit(self, name, hourly_limit):
        if name != SQLAlchemyDB.admin_name():
            self._db.set_user_hourly_limit(name, int(hourly_limit))
            return 'Ok'
        else:
            cherrypy.response.status = 403
            return 'Cannot change "%s"' % name

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_request_limit(self, name, request_limit):
        if name != SQLAlchemyDB.admin_name():
            self._db.set_user_request_limit(name, int(request_limit))
            return 'Ok'
        else:
            cherrypy.response.status = 403
            return 'Cannot change "%s"' % name

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @require(name_is(SQLAlchemyDB.admin_name()))
    def set_current_request_counter(self, name, count=0):
        self._db.set_user_current_request_counter(name, int(count))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def version(self):
        import ics
        return ics.__version__
