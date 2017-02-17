# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2014 Robert Jordens <jordens@phys.ethz.ch>
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

from rayopt import (system_from_yaml, system_to_yaml, system_to_json,
                    system_from_json)


class TripletYamlCase(unittest.TestCase):
    def setUp(self):
        text = """
description: oslo cooke triplet example 50mm f/4 20deg
wavelengths: [546.1e-9, 486.e-9, 656.e-9]
object: {angle_deg: 20, pupil: {radius: 6.25}}
stop: 4
elements:
- {material: air}
- {distance: 5, radius: 6.5, roc: 21.25, material: schott-sk|n-sk16}
- {distance: 2, radius: 6.5, roc: -158.65, material: air}
- {distance: 6, radius: 5, roc: -20.25, material: schott-f|n-f2}
- {distance: 1, radius: 5, roc: 19.3, material: air}
- {distance: 6, radius: 6.5, roc: 141.25, material: schott-sk|n-sk16}
- {distance: 2, radius: 6.5, roc: -17.285, material: air}
- {distance: 42.95, radius: .364, material: air}
"""
        self.s = system_from_yaml(text)

    def test_load(self):
        assert self.s is not None

    def test_dump(self):
        d = system_to_yaml(self.s)
        s = system_from_yaml(d)
        s

    def test_json(self):
        d = system_to_json(self.s)
        s = system_from_json(d)
        s
