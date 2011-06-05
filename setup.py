#! /usr/bin/env python
# Encoding: UTF-8

from setuptools import setup, find_packages

__copyright__ = "Copyright 2009-2011, Petr Viktorin"
__license__ = "MIT"
__version__ = '0.1'
__author__ = 'Petr "En-Cu-Kou" Viktorin'
__email__ = 'encukou@gmail.com'

setup(
    name='touchgames',
    version=__version__,
    description=u'Games for a touch table',
    author=__author__,
    author_email=__email__,
    install_requires=[
            "kivy>=1.0",
            "numpy>=1.3",
            "cython>=0.14",
        ],
    packages=find_packages(),
)
