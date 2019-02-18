from __future__ import print_function, unicode_literals, division

# Using setuptools allows the --develop option.
from setuptools import setup

setup(
    name='audio_sheet_retrieval',
    version='0.1',
    packages=['audio_sheet_retrieval'],
    include_package_data=True,
    url='',
    license='(c) All rights reserved.',
    author='Matthias Dorfer',
    author_email='matthias.dorfer@jku.at',
    description='Tools for training and applying audio - sheet-music retrieval models.'
)
