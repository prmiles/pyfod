from setuptools import setup, find_packages
import codecs
import re
import os

def read(fname):
    with codecs.open(fname, 'r', 'latin') as f:
        return f.read()
    
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with codecs.open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
    
def get_version():
    VERSIONFILE = os.path.join('pyfod', '__init__.py')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = '^__version__ = \"\d+\.\d+\.\d.*\"'
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split('"')[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

setup(
    name='pyfod',
    version=get_version(),
    description=('A library to evaluating fractional-order derivative operators'),
    long_description=long_description,
    url='https://github.com/prmiles/pyfod',
    download_url='https://github.com/prmiles/pyfod',
    author='Paul Miles',
    author_email='prmiles@ncsu.edu',
    license='MIT',
    package_dir={'pyfod': 'pyfod'},
    packages=find_packages(),
    zip_safe=False,
    install_requires=['numpy>=1.14', 'scipy>=1.0', 'sympy>=1.3'],
    extras_require = {'docs':['sphinx']},
    classifiers=['License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Framework :: IPython',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   ]
)
