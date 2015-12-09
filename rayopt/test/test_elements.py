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

import os
import unittest

from scipy import constants as ct
import numpy as np
from numpy import testing as nptest


from rayopt import Spheroid, ModelMaterial, mirror
from rayopt.utils import sinarctan, tanarcsin


class TransformCase(unittest.TestCase):
    def setUp(self):
        self.s = Spheroid(distance=2., direction=(1, 3, 4.),
                angles=(.3, .2, .1))

    def test_offset(self):
        nptest.assert_allclose(self.s.offset,
                self.s.distance*self.s.direction)
    
    def test_from_to_axis(self, n=10):
        x = np.random.randn(n, 3)
        x1 = self.s.to_axis(x)
        x2 = self.s.from_axis(x1)
        nptest.assert_allclose(x, x2)

    def test_from_to_normal(self, n=10):
        x = np.random.randn(n, 3)
        x1 = self.s.to_normal(x)
        x2 = self.s.from_normal(x1)
        nptest.assert_allclose(x, x2)

    def test_rot(self):
        self.s.angles = 0, 0, 0
        x = np.array([0., 0, 3])
        x1 = self.s.from_normal(x)
        nptest.assert_allclose(x1, self.s.direction*3)
        self.s.direction = 0, 0, 1.
        self.s.angles = .1, 0, 0
        x1 = self.s.from_normal(x)
        nptest.assert_allclose(x1, (0, 3*np.sin(.1), 3*np.cos(.1)))


class ParaxialCase(unittest.TestCase):
    def setUp(self):
        self.mat = mat = ModelMaterial(n=1.5)
        self.s0 = Spheroid(curvature=0., distance=0., material=mat)
        self.s = Spheroid(curvature=.1, distance=0, material=mat)
        self.sm0 = Spheroid(curvature=0, distance=0, material=mirror)
        self.sm = Spheroid(curvature=.1, distance=0, material=mirror)

    def test_offset(self):
        nptest.assert_allclose(self.s.direction, (0, 0, 1))
        nptest.assert_allclose(self.s.distance, 0)
        nptest.assert_allclose(self.s.offset, 0)
    
    def test_rotation(self):
        nptest.assert_allclose(self.s.angles, (0, 0, 0.))
        nptest.assert_allclose(self.s.from_axis([0, 0, 1]), (0, 0, 1))
        nptest.assert_allclose(self.s.from_normal([0, 0, 1]), (0, 0, 1))

    def test_snell_paraxial(self):
        y0, u0 = (1, 2), (.2, .1)
        yu, n = self.s0.propagate_paraxial(np.hstack((y0, u0)), 1., 1.)
        y, u = np.hsplit(yu, 2)
        mu = 1/self.s0.material.n
        nptest.assert_allclose(y, y0)
        nptest.assert_allclose(u/mu, u0)

    def test_snell_paraxial_mirror(self):
        y0, u0 = (1, 2), (.2, .1)
        yu, n = self.sm0.propagate_paraxial(np.hstack((y0, u0)), 1., 1.)
        y, u = np.hsplit(yu, 2)
        nptest.assert_allclose(-y, y0)
        nptest.assert_allclose(-u, u0)

    def test_align(self):
        d = 0, -.1, 1
        d /= np.linalg.norm(d)
        mu = 1/self.s0.material.n
        self.s0.align(d, mu)
        e = self.s0.from_normal(self.s0.excidence(mu))
        nptest.assert_allclose(e, d)
        y0, u0 = (1, 2), (.2, .0)
        yu, n = self.s0.propagate_paraxial(np.hstack((y0, u0)), 1., 1.)
        y, u = np.hsplit(yu, 2)
        nptest.assert_allclose(y[0], y0[0])
        nptest.assert_allclose(u[0]/mu, u0[0])
        nptest.assert_allclose(u[1]/mu, d[0])


class ParaxToRealCase(unittest.TestCase):
    def setUp(self):
        self.mat = mat = ModelMaterial(n=1.5)
        d = np.random.randn(3)*1e-1 + (0, 0, 1.)
        a = np.random.randn(3)*1e-8
        a[1:] = 0
        self.s = Spheroid(curvature=.1, distance=.2, material=mat,
                direction=d, angles=a)
        de = self.s.excidence(1/self.s.material.n)
        self.sa = Spheroid(direction=de)

    def test_real_similar_to_parax(self, n=100, e=1e-3):
        y0p = np.random.randn(n, 2.)*e
        u0p = np.random.randn(n, 2.)*e
        y0r = np.hstack((y0p, np.ones((n, 1))*-self.s.distance))
        u0r = np.hstack((sinarctan(u0p), np.zeros((n, 1))))
        u0r[:, 2] = np.sqrt(1 - np.square(u0p).sum(1))
        yup, np_ = self.s.propagate_paraxial(np.hstack((y0p, u0p)), 1., 1.)
        yp, up = np.hsplit(yup, 2)
        #y0r, u0r = self.s.to_normal(y0r, u0r)
        yr, ur, nr, tr  = self.s.propagate(y0r, u0r, 1., 1.)
        #yr, ur = self.s.from_normal(yr, ur)
        yr, ur = self.sa.to_axis(yr, ur)
        nptest.assert_allclose(nr, np_, rtol=1e-4, atol=1e-9)
        nptest.assert_allclose(yr[:, :2], yp, rtol=2e-4, atol=1e-9)
        nptest.assert_allclose(tanarcsin(ur), up, rtol=2e-4, atol=1e-9)
