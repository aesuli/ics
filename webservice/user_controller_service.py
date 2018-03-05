
import cherrypy

from db.sqlalchemydb import SQLAlchemyDB
from util.util import logged_call_with_args
from webservice.auth_controller_service import require, SESSION_KEY

__author__ = 'Andrea Esuli'



def must_be_logged_in():
    def check():
        return cherrypy.request.login is not None

    return check


def name_is(required_username):
    def check():
        return lambda: required_username == cherrypy.request.login

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

    @logged_call_with_args
    def on_login(self, username):
        pass

    @logged_call_with_args
    def on_logout(self, username):
        pass

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def info(self):
        result = []
        requester = cherrypy.request.login
        if requester is None:
            return result
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
    def login(self, username=None, password=None):
        if username is None or password is None:
            cherrypy.response.status = 401
            return 'Wrong credentials'

        if self._db.verify_user(username, password):
            cherrypy.session[SESSION_KEY] = cherrypy.request.login = username
            self.on_login(username)
            return 'Ok'
        else:
            cherrypy.response.status = 401
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
            self.on_logout(username)

    @cherrypy.expose
    def version(self):
        return "0.2.5 (db: %s)" % self._db.version()
