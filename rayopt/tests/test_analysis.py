# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2013 Robert Jordens <jordens@phys.ethz.ch>
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

import os
import unittest

from scipy import constants as ct
import numpy as np
from numpy import testing as nptest


from rayopt import system_from_yaml, Analysis


class DemotripCase(unittest.TestCase):
    def setUp(self):
        self.s = system_from_yaml("""
description: 'oslo cooke triplet example 50mm f/4 20deg'
object: {angle: .364}
stop: 5
elements:
- {material: air}
- {roc: 21.25, distance: 5.0, material: SK16, radius: 6.5}
- {roc: -158.65, distance: 2.0, material: air, radius: 6.5}
- {roc: -20.25, distance: 6.0, material: F4, radius: 5.0}
- {roc: 19.3, distance: 1.0, material: air, radius: 5.0}
- {material: basic/air, radius: 4.75}
- {roc: 141.25, distance: 6.0, material: SK16, radius: 6.5}
- {roc: -17.285, distance: 2.0, material: air, radius: 6.5}
- {distance: 42.95, radius: 0.364}
""")

    def test_run(self):
        a = Analysis(self.s)
