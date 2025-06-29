import pathlib
from os import walk

from setuptools import setup, find_namespace_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

requirements_list = (here / 'requirements.txt').read_text(encoding='utf-8').split()


def get_version(rel_path):
    init_content = (here / rel_path).read_text(encoding='utf-8')
    for line in init_content.split('\n'):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


scripts = [f"ics-{script[:-3].replace('_', '-')}=ics.scripts.{script[:-3]}:main" for script in
           next(walk(here / 'ics/scripts/'), (None, None, []))[2] if not script.startswith('_')]

setup(
    name='ics_pkg',

    version=get_version("ics/__init__.py"),

    description='Interactive Classification System (ICS): a tool for machine learning-supported labeling of text',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/aesuli/ics',

    author='Andrea Esuli',
    author_email='andrea@esuli.it',

    license='BSD-3-Clause',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    keywords='text, classification, labeling, machine learning, active learning',

    packages=find_namespace_packages(include=['ics', 'ics.*']),

    include_package_data=True,

    python_requires='>=3.8',

    install_requires=requirements_list,

    entry_points={
        'console_scripts': scripts
    },

    project_urls={
        'Bug Reports': 'https://github.com/aesuli/ics/issues',
        'Source': 'https://github.com/aesuli/ics/',
    },
)
