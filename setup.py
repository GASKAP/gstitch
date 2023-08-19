# -*- coding: utf-8 -*-                                                                                                                                                                                                                             
from setuptools import setup, find_packages

with open('README.md') as f:
        readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='gstitch',
    version='0.1.0',
    description='gstitch for GASKAP',
    long_description=readme,
    classifiers=[
        'Development status :: 1 - Alpha',
        'License :: CC-By-SA2.0',
        'Programming Language :: Python',
        'Topic :: Data Analysis'
    ],
    author='Antoine Marchal',
    author_email='antoine.marchal@anu.edu.au',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
            'numpy',
	    'astropy',
    ],
    include_package_data=True
)
