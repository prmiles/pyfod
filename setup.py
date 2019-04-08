from setuptools import setup, find_packages
import codecs

def read(fname):
    with codecs.open(fname, 'r', 'latin') as f:
        return f.read()
    
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with codecs.open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
    
# read in version number
version_dummy = {}
exec(read('pyfod/__version__.py'), version_dummy)
__version__ = version_dummy['__version__']
del version_dummy

setup(
    name='pyfod',
    version=__version__,
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
