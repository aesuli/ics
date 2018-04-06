import re

__author__ = 'Andrea Esuli'

badchars = re.compile(r'[^A-Za-z0-9_. ]+|^\.|\.$|^ | $|^$')
badnames = re.compile(r'(aux|com[1-9]|con|lpt[1-9]|prn)(\.|$)')


def get_fully_portable_file_name(name):
    name = badchars.sub('_', name)
    if badnames.match(name):
        name = '_' + name
    return name
