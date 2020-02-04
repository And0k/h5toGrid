# pipenv-gdal-select/setup.py
from setuptools import setup

setup(
    name='pipenv-gdal-select',
    version='1.0',
    installs_requires=[
        'path = "/home/korzh/Downloads/Distr/gdal-2.1.4/swig/python"; sys.platform == "linux"',
        'file = "d:/Distr/_coding/Python/GDAL-3.0.0-cp36-cp36m-win32.whl"; sys.platform == "win32"',
        ],
    classifiers=[
        'Programming Language :: Python :: 3'
        ]
    )
