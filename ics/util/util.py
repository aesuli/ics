import argparse
import re

__author__ = 'Andrea Esuli'

badchars = re.compile(r'[^A-Za-z0-9_. ]+|^\.|\.$|^ | $|^$')
badnames = re.compile(r'(aux|com[1-9]|con|lpt[1-9]|prn)(\.|$)')


def get_fully_portable_file_name(name):
    name = badchars.sub('_', name)
    if badnames.match(name):
        name = '_' + name
    return name


def str_to_bool(value):
    if value.strip().lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.strip().lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not a boolean value: ' + value)


def bool_to_string(condition: bool, if_true: str, if_false: str):
    if condition:
        return if_true
    else:
        return if_false
