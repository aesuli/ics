# Original source: https://github.com/cherrypy/tools/blob/master/AuthenticationAndAccessRestrictions

import cherrypy

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
                raise cherrypy.HTTPError(401)


def enable_controller_service():
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
        if 'tools.icsauth.require' not in f._cp_config:
            f._cp_config['tools.icsauth.require'] = []
        f._cp_config['tools.icsauth.require'].extend(conditions)
        return f

    return decorate


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
