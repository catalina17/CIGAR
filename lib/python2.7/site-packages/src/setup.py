#!/usr/bin/python
#
# setup.py
#

from setuptools import setup, find_packages

setup(
  name = "atlas",
  version = "0.25.6",
  packages = find_packages(),
  scripts = ['atlas'],

  install_requires = ['threadpool>=1.2.4'],

  author = "Mike Parent",
  description = "Atlas Build Tool - simple, extensible build system.",
  license = "Apache 2.0",
  keywords = "build test",
  url = "http://code.google.com/p/atlas-build-tool",
  zip_safe=True,
)
