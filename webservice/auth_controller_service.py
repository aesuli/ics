import cherrypy

__author__ = 'Andrea Esuli, adapted from: https://github.com/cherrypy/tools/blob/master/AuthenticationAndAccessRestrictions'

SESSION_KEY = '_cp_username'


def check_auth(*args, **kwargs):
    conditions = cherrypy.request.config.get('tools.icsauth.require', None)
    if conditions is not None:
        cherrypy.request.login = cherrypy.session.get(SESSION_KEY,None)
        for condition in conditions:
            if not condition():
                raise cherrypy.HTTPError(401)


def enable_controller_service():
    cherrypy.tools.icsauth = cherrypy.Tool('before_handler', check_auth)


def require(*conditions):
    def decorate(f):
        if not hasattr(f, '_cp_config'):
            f._cp_config = dict()
        if 'tools.icsauth.require' not in f._cp_config:
            f._cp_config['tools.icsauth.require'] = []
        f._cp_config['tools.icsauth.require'].extend(conditions)
        return f

    return decorate


def forbid():
    def check():
        return False

    return check


def negation_of(condition):
    def check():
        return not condition()

    return check


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
