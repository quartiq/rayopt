#!/usr/bin/python
# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
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

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

setup(
        name = "pyrayopt",
        version = "0.0+dev",
        description = "raytracing for optical imaging systems",
        long_description = """build on enthought/{traits, chaco}.""",
        author = "Robert Jordens",
        author_email = "jordens@phys.ethz.ch",
        url = "http://launchpad.net/pyrayopt",
        license = "GPLv3+",
        keywords = "optics lens raytracing optimization point spread",
        install_requires = [
            "numpy", "scipy", "matplotlib", "nose", "numba"],
        extras_require = {
            "gui": ["chaco", "traitsui"],
            },
        #dependency_links = [],
        packages = find_packages(),
        #namespace_packages = [],
        #test_suite = "bullseye.tests.test_all",
        ext_modules=[
                Extension("rayopt._transformations",
                    sources=["rayopt/_transformations.c"],),
            ],
        cmdclass = {"build_ext": build_ext},

        entry_points = {
            #"gui_scripts": ["bullseye = bullseye.app:main"],
            },
        include_package_data = True,
        classifiers = [f.strip() for f in """
            Development Status :: 4 - Beta
            Environment :: X11 Applications :: GTK
            Environment :: X11 Applications :: Qt
            Intended Audience :: Science/Research
            License :: OSI Approved :: GNU General Public License (GPL)
            Operating System :: OS Independent
            Programming Language :: Python :: 2
            Topic :: Multimedia :: Graphics :: Capture :: Digital Camera
            Topic :: Scientific/Engineering :: Physics
        """.splitlines() if f.strip()],
        )
