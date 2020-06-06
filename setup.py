#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nearest Class Mean Metric Learning (NCMML) v0.4
Modified version of T. Mensink's Matlab code
@ date : Created on Wed, Nov, 19, 2018
@ author : Sangjun Han, LG CNS, South Korea
"""

import numpy as np
from os import path
from codecs import open

from Cython.Build import cythonize
from setuptools import setup, find_packages
from distutils.extension import Extension

here = path.abspath(path.dirname(__file__))
extensions = [Extension("*", ["NCMML/*.pyx"])]
    
setup(
      name = "NCMML", 
      version = "0.4,
      description = "Nearest Class Mean Metric Learning",
      url = None,
      author = "Sangjun Han",
      author_email = "sangjunhan@lgcns.com",
      keywords = "Distance Metric Learning Classification Nearest Class Mean Metric Learning",
      packages = find_packages(exclude = ["misc", "data", "test", "utils"]),
      install_requires = ["numpy", "scikit-learn", "Cython"],
      language = "c++",
      ext_modules = cythonize(extensions),
      include_dirs = [np.get_include()] 
)
