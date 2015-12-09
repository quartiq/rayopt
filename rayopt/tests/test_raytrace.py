# -*- coding: utf8 -*-
#
#   rayopt - raytracing for optical imaging systems
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

import unittest

import numpy as np
from numpy import testing as nptest


from rayopt import (system_from_yaml, ParaxialTrace, GeometricTrace,
                    system_to_yaml)
from rayopt.utils import tanarcsin


cooke = """
description: 'oslo cooke triplet example 50mm f/4 20deg'
wavelengths: [587.56e-9, 656.27e-9, 486.13e-9]
object: {angle_deg: 20, pupil: {radius: 6.25, aim: True}}
image: {type: finite, pupil: {radius: 0, update_radius: True}}
elements:
- {material: air}
- {roc: 21.25, distance: 5.0, material: SCHOTT-SK|N-SK16, radius: 6.5}
- {roc: -158.65, distance: 2.0, material: air, radius: 6.5}
- {roc: -20.25, distance: 6.0, material: SCHOTT-F|N-F2, radius: 5.0}
- {roc: 19.6, distance: 1.0, material: air, radius: 5.0}
- {material: air, radius: 4.75}
- {roc: 141.25, distance: 6.0, material: SCHOTT-SK|N-SK16, radius: 6.5}
- {roc: -17.285, distance: 2.0, material: air, radius: 6.5}
- {distance: 42.95, radius: 0.364}
stop: 5
pickups:
- {get: [1, radius], set: [2, radius]}
- {get: [3, radius], set: [4, radius]}
- {get: [6, radius], set: [7, radius]}
validators:
- {get: [edge_y, 2], minimum: .5}
- {get: [2, distance], minimum: .5}
- {get: [edge_y, 4], minimum: .5}
- {get: [4, distance], minimum: .5}
- {get: [edge_y, 7], minimum: .5}
- {get: [7, distance], minimum: .5}
"""


class DemotripCase(unittest.TestCase):
    def setUp(self):
        self.s = system_from_yaml(cooke)
        self.s.update()
        self.s.paraxial.refocus()

    def test_from_text(self):
        self.assertFalse(self.s.object.finite)
        for i, el in enumerate(self.s):
            if i not in (0,):
                self.assertGreater(el.radius, 0)
            if i not in (0, self.s.stop):
                self.assertGreater(el.distance, 0)
            if i not in (0, self.s.stop, len(self.s)-1):
                self.assertGreater(abs(el.curvature), 0)
            if i not in (len(self.s)-1, ):
                self.assertIsNot(el.material, None)

    def test_system(self):
        s = self.s
        self.assertGreater(len(str(s).splitlines()), 10)
        self.assertIs(s.aperture, s[s.stop])
        # self.assertEqual(len(self.s), 9)

    def test_reverse(self):
        s = self.s
        s.reverse()
        s.reverse()
        self.test_from_text()
        self.test_system()

    def test_rescale(self):
        l = [el.distance for el in self.s]
        self.s.rescale(123)
        nptest.assert_allclose([el.distance/123 for el in self.s], l)
        self.s.rescale()
        nptest.assert_allclose([el.distance for el in self.s], l)

    def test_funcs(self):
        self.s.resize_convex()
        list(self.s.surfaces_cut(axis=1, points=11))
        self.s.paraxial_matrices(self.s.wavelengths[0], start=1, stop=None)
        self.s.paraxial_matrix(self.s.wavelengths[0], start=1, stop=None)
        self.s.track
        self.s.origins
        self.s.mirrored
        self.s.align(np.ones_like(self.s.track))

    def test_paraxial(self):
        p = ParaxialTrace(self.s)
        # print(str(p))
        nptest.assert_allclose(p.u[0, 0], 0)
        nptest.assert_allclose(p.u[0, 1], tanarcsin(self.s.object.angle))
        nptest.assert_allclose(p.y[self.s.stop, 0], self.s[self.s.stop].radius,
                               rtol=1e-2)
        nptest.assert_allclose(p.y[self.s.stop, 1], 0, atol=1e-9)
        nptest.assert_allclose(p.working_f_number[1], -self.s.image.pupil.fno,
                               rtol=1e-2)
        nptest.assert_allclose(p.working_f_number[1], 4, rtol=1e-2)
        nptest.assert_allclose(p.focal_length[1], 50, rtol=5e-3)
        nptest.assert_allclose(p.magnification[0], 0, rtol=1e-3)
        nptest.assert_allclose(p.numerical_aperture[1], .124, rtol=5e-3)
        p.update_conjugates()
        self.s.image.na = .125
        p.update_stop("image")
        p = ParaxialTrace(self.s)
        p.update_conjugates()
        print(system_to_yaml(self.s))
        print(str(p))

    def test_reverse_size(self):
        p = ParaxialTrace(self.s)
        p.update_conjugates()
        self.s.reverse()
        p = ParaxialTrace(self.s)
        # print(system_to_yaml(self.s))
        # print(str(p))

    def traces(self):
        p = ParaxialTrace(self.s)
        p.update_conjugates()
        g = GeometricTrace(self.s)
        return p, g

    def test_aim(self):
        p, g = self.traces()
        # print(z, a)
        z, p = self.s.pupil((0, 1.))
        # print(z, a)

    def test_aim_point(self):
        p, g = self.traces()
        g.rays_point((0, 1.))
        g.rays_clipping((0, 1.))
        g.rays_line((0, 1.))

    def test_aim_point_more(self):
        p, g = self.traces()
        i = self.s.stop
        r = np.array([el.radius for el in self.s[1:-1]])

        g.rays_clipping((0, 1.))
        if not self.s.object.finite:
            nptest.assert_allclose(g.u[0, :, :],
                                   g.u[0, (0,)*g.u.shape[1], :])
        nptest.assert_allclose(g.y[i, 0, 1], 0, atol=5e-3)
        nptest.assert_allclose(min(g.y[1:-1, 1, 1] + r), 0, atol=1e-3)
        nptest.assert_allclose(max(g.y[1:-1, 2, 1] - r), 0, atol=1e-3)

        g.rays_point((0, 1.), distribution="cross", nrays=5,
                     filter=False)
        if not self.s.object.finite:
            nptest.assert_allclose(g.u[0, :, :],
                                   g.u[0, (0,)*g.u.shape[1], :])
        nptest.assert_allclose(g.y[i, :3, 1]/self.s[i].radius,
                               [-1, 0, 1], atol=1e-3, rtol=3e-2)
        nptest.assert_allclose(g.y[i, :, 0]/self.s[i].radius,
                               [0, 0, 0, -1, 0, 1], atol=1e-1)
        # print(g.y[i, :, :2]/self.s[i].radius)
        g.rays_line((0, 1.))

    def test_pupil(self):
        p, g = self.traces()
        p.update_conjugates()
        for y in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                  (.1, .1), (-.2, .5)]:
            self.s.pupil(y)

    def test_quadrature(self):
        p, g = self.traces()
        p.update_conjugates()
        g.rays_point((0, 1.), nrays=13, distribution="radau",
                     filter=False)
        a = g.rms()
        nptest.assert_allclose(a, .052, rtol=1e-2)
        g.rays_point((0, 1.), nrays=500, distribution="square",
                     clip=False, filter=True)
        b = g.rms()
        nptest.assert_allclose(a, b, rtol=5e-2)
