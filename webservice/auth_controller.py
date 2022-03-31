#!python
# -*- encoding: UTF-8 -*-
#
# Form based authentication for CherryPy. Requires the Session tool to be loaded.
# original source: https://github.com/cherrypy/tools/blob/master/AuthenticationAndAccessRestrictions

import cherrypy

from db.sqlalchemydb import SQLAlchemyDB
from util.util import logged_call_with_args

SESSION_KEY = '_cp_username'


def check_auth(*args, **kwargs):
    always_auth = cherrypy.request.config.get('tools.icsauth.always_auth', None)
    if always_auth:
        return

    conditions = cherrypy.request.config.get('tools.icsauth.require', None)
    if conditions is not None:
        cherrypy.request.login = cherrypy.session.get(SESSION_KEY)
        for condition in conditions:
            if not condition():
                cherrypy.response.status = 401
                return 'Unathorized'


cherrypy.tools.icsauth = cherrypy.Tool('before_handler', check_auth)


def always_auth():
    def decorate(f):
        if not hasattr(f, '_cp_config'):
            f._cp_config = dict()
        if 'tools.icsauth.always_auth' not in f._cp_config:
            f._cp_config['tools.icsauth.always_auth'] = True
        return f

    return decorate


def require(*conditions):
    def decorate(f):
        if not hasattr(f, '_cp_config'):
            f._cp_config = dict()
        if 'auth.require' not in f._cp_config:
            f._cp_config['tools.icsauth.require'] = []
        f._cp_config['tools.icsauth.require'].extend(conditions)
        return f

    return decorate


# def member_of(groupname):
#     def check():
#         # replace with actual check if <username> is in <groupname>
#         return cherrypy.request.login == 'joe' and groupname == 'admin'
#
#     return check


def name_is(reqd_username):
    return lambda: reqd_username == cherrypy.request.login


def any_of(*conditions):
    def check():
        for c in conditions:
            if c():
                return True
        return False

    return check


def all_of(*conditions):
    def check():
        for c in conditions:
            if not c():
                return False
        return True

    return check


class AuthController(object):
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

    @logged_call_with_args
    def on_login(self, username):
        pass

    @logged_call_with_args
    def on_logout(self, username):
        pass

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        result = []
        for name in self._db.user_names():
            user_info = dict()
            user_info['name'] = name
            user_info['created'] = str(self._db.get_user_creation_time(name))
            result.append(user_info)
        return result

    @cherrypy.expose
    @always_auth()
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
        if self._db.user_exists(username):
            cherrypy.response.status = 403
            return 'User "%s" already exists' % username

        self._db.create_user(username, password)

    @cherrypy.expose
    def change_password(self, username, password):
        if not self._db.user_exists(username):
            cherrypy.response.status = 403
            return 'User "%s" does not exists' % username

        requester = cherrypy.session[SESSION_KEY]
        if requester != SQLAlchemyDB.admin_name() and requester != username:
            cherrypy.response.status = 401
            return 'Unathorized'

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
        return "0.1.2"
