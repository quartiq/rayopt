# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2013 Robert Jordens <robert@joerdens.org>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

import unittest

from rayopt import system_from_yaml, Analysis
from .test_raytrace import cooke


class DemotripCase(unittest.TestCase):
    def setUp(self):
        self.s = system_from_yaml(cooke)
        self.s.update()

    def test_run(self):
        a = Analysis(self.s)
        return
        for _ in a.text:
            print(_)
        for i, _ in enumerate(a.figures):
            _.savefig("analysis_%i.pdf" % i)
