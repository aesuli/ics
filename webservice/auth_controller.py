#!python
# -*- encoding: UTF-8 -*-
#
# Form based authentication for CherryPy. Requires the
# Session tool to be loaded.
# source: https://github.com/cherrypy/tools/blob/master/AuthenticationAndAccessRestrictions
#
import os

import cherrypy
from mako.lookup import TemplateLookup

from util.util import logged_call_with_args

SESSION_KEY = '_cp_username'


def check_auth(*args, **kwargs):
    """A tool that looks in config for 'auth.require'. If found and it
    is not None, a login is required and the entry is evaluated as a list of
    conditions that the user must fulfill"""

    # check if function is decorated with always_auth, which always gives authorization
    always_auth = cherrypy.request.config.get('tools.icsauth.always_auth', None)
    if always_auth:
        return

    conditions = cherrypy.request.config.get('tools.icsauth.require', None)
    if conditions is not None:
        username = cherrypy.session.get(SESSION_KEY)
        if username:
            cherrypy.request.login = username
            for condition in conditions:
                # A condition is just a callable that returns true or false
                if not condition():
                    raise cherrypy.HTTPRedirect("/auth/login/?from_page=" + cherrypy.request.script_name)
        else:
            raise cherrypy.HTTPRedirect("/auth/login/?from_page=" + cherrypy.request.script_name)


cherrypy.tools.icsauth = cherrypy.Tool('before_handler', check_auth)


def always_auth():
    """A decorator that always grants authorization."""

    def decorate(f):
        if not hasattr(f, '_cp_config'):
            f._cp_config = dict()
        if 'tools.icsauth.always_auth' not in f._cp_config:
            f._cp_config['tools.icsauth.always_auth'] = True
        return f

    return decorate


def require(*conditions):
    """A decorator that appends conditions to the auth.require config
    variable."""

    def decorate(f):
        if not hasattr(f, '_cp_config'):
            f._cp_config = dict()
        if 'auth.require' not in f._cp_config:
            f._cp_config['tools.icsauth.require'] = []
        f._cp_config['tools.icsauth.require'].extend(conditions)
        return f

    return decorate


# Conditions are callables that return True
# if the user fulfills the conditions they define, False otherwise
#
# They can access the current username as cherrypy.request.login
#
# Define those at will however suits the application.

def member_of(groupname):
    def check():
        # replace with actual check if <username> is in <groupname>
        return cherrypy.request.login == 'joe' and groupname == 'admin'

    return check


def name_is(reqd_username):
    return lambda: reqd_username == cherrypy.request.login


# These might be handy

def any_of(*conditions):
    """Returns True if any of the conditions match"""

    def check():
        for c in conditions:
            if c():
                return True
        return False

    return check


# By default all conditions are required, but this might still be
# needed if you want to use it inside of an any_of(...) condition
def all_of(*conditions):
    """Returns True if all of the conditions match"""

    def check():
        for c in conditions:
            if not c():
                return False
        return True

    return check


# Controller to provide login and logout actions

class AuthController(object):
    def __init__(self, media_dir, name):
        self._lookup = TemplateLookup(os.path.join(media_dir, 'template'))
        self._name = name

    def close(self):
        # self._db.close()
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @logged_call_with_args
    def on_login(self, username):
        """Called on successful login"""

    @logged_call_with_args
    def on_logout(self, username):
        """Called on logout"""

    def check_credentials(self, username, password):
        """Verifies credentials for username and password.
        Returns None on success or a string describing the error on failure"""
        # Adapt to your needs
        if username in ('joe', 'steve') and password == 'secret':
            return None
        else:
            return "Incorrect username or password."

            # An example implementation which uses an ORM could be:
            # u = User.get(username)
            # if u is None:
            #     return u"Username %s is unknown to me." % username
            # if u.password != md5.new(password).hexdigest():
            #     return u"Incorrect password"

    @cherrypy.expose
    @always_auth()
    def login(self, username=None, password=None, from_page="/"):
        if username is None or password is None:
            template = self._lookup.get_template('login.html')
            if username is None:
                username = ""
            return template.render(
                **{'from_page': from_page, 'msg': 'Please login', 'name': self._name, 'username': username})

        error_msg = self.check_credentials(username, password)
        if error_msg:
            template = self._lookup.get_template('login.html')
            if username is None:
                username = ""
            return template.render(
                **{'from_page': from_page, 'msg': error_msg, 'name': self._name, 'username': username})
        else:
            cherrypy.session[SESSION_KEY] = cherrypy.request.login = username
            self.on_login(username)
            raise cherrypy.HTTPRedirect(from_page or "/")

    @cherrypy.expose
    def logout(self, from_page="/"):
        sess = cherrypy.session
        username = sess.get(SESSION_KEY, None)
        sess[SESSION_KEY] = None
        if username:
            cherrypy.request.login = None
            self.on_logout(username)
        raise cherrypy.HTTPRedirect(from_page or "/")

    @cherrypy.expose
    def version(self):
        return "0.0.2"
