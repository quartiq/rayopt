#!/usr/bin/python
# -*- coding: utf8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2012 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

from setuptools import setup, find_packages
from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np

setup(
    name="rayopt",
    version="0.0+dev",
    description="raytracing and lens design framework",
    long_description=open("README.rst").read(),
    author="Robert Jordens",
    author_email="jordens@gmail.com",
    url="https://github.com/jordens/rayopt",
    license="GPLv3+",
    keywords=("optics lens design raytracing optimization "
              "point spread opd aberration"),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        "sqlalchemy",
        "cython",
    ],
    extras_require={},
    packages=find_packages(),
    test_suite="rayopt.test",
    ext_modules=cythonize([
        Extension("rayopt._transformations", ["rayopt/_transformations.c"]),
        Extension("rayopt.simplex_accel", ["rayopt/simplex_accel.pyx"]),
    ]),
    include_dirs=[np.get_include()],
    entry_points={},
    package_data={"rayopt": ["library.sqlite"]},
    classifiers=[f.strip() for f in """
        Development Status :: 4 - Beta
        Intended Audience :: Science/Research
        License :: OSI Approved :: GNU General Public License (GPL)
        Operating System :: OS Independent
        Programming Language :: Python :: 2
        Programming Language :: Python :: 3
        Topic :: Scientific/Engineering :: Physics
    """.splitlines() if f.strip()],
)
