#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:55:28 2017

@author: mittelberger2
"""

from distutils.core import setup
from distutils.extension import Extension

setup(
      name = 'AnalyzeMaxima',
      py_modules = ['AnalyzeMaxima.local_maxima'],
      ext_modules = [Extension('AnalyzeMaxima.analyze_maxima', ['AnalyzeMaxima/analyze_maxima.c'])],
)
