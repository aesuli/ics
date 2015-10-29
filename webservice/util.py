import re
import functools
import cherrypy
import logging

__author__ = 'Andrea Esuli'

badchars = re.compile(r'[^A-Za-z0-9_. ]+|^\.|\.$|^ | $|^$')
badnames = re.compile(r'(aux|com[1-9]|con|lpt[1-9]|prn)(\.|$)')


def get_fully_portable_file_name(name):
    name = badchars.sub('_', name)
    if badnames.match(name):
        name = '_' + name
    return name

def logged_call(function):
    @functools.wraps(function)
    def decorated(*args, **kwargs):
        call_str = "%s(args=%s,kwargs=%s" % (function.__name__, str(args), str(kwargs))
        cherrypy.log("Calling %s" % call_str, severity=logging.INFO)
        try:
            ret = function(*args, **kwargs)
        except Exception as e:
            cherrypy.log("Error from %s: %s" % (call_str, e), severity=logging.ERROR, traceback=True)
            raise e
        else:
            cherrypy.log("Returned %s" % call_str, severity=logging.INFO)
        return ret
    return decorated