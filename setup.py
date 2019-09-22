#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup file.

Created on Tue Sep 17 22:20:05 2019
@author: Juan Beleño
"""
import os
from setuptools import setup

about = {}
directory = os.path.dirname(__file__)
version_filepath = os.path.join(directory, 'meli_category_classifier',
                                '__version__.py')
with open(version_filepath) as version:
    exec(version.read(), about)

setup(
    name='meli_category_classifier',
    version=about['__version__'],
    packages=['meli_category_classifier'],
    install_requires=[
        'fire==0.2.1',
        'numpy==1.17.2',
        'pandas==0.25.1',
        'pyyaml==5.1.2',
        'tensorflow-datasets==1.2.0',
        'bpemb==0.3.0'
    ],
    extras_require={
        'cpu': ['tensorflow==2.0.0-rc1'],
        'gpu': ['tensorflow-gpu==2.0.0-rc1'],
    },
    entry_points={
        'console_scripts': [
            'meliclassifier = meli_category_classifier.__main__:main',
        ]
    },
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
    ]
)
