# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2015 Robert Jordens <jordens@phys.ethz.ch>
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

import itertools
from collections import namedtuple

import numpy as np

from .utils import public
from .raytrace import Trace
from .simplex import make_simplex, simplex_transform


PolyState = public(namedtuple("PolyState", "f n r p k s t v w o"))


@public
class PolyTrace(Trace):
    """Polynomial ray trace (after T. B. Andersen, Automatic computation of optical
    aberration coefficients, Applied Optics 19, 3800-3816 (1980),
    http://dx.doi.org/10.1364/AO.19.003800).

    With generalizations to finite and telecentric objects (after F. Bociort,
    T. B. Andersen, and L.H.J.F. Beckmann, High-order optical aberration
    coefficients: extension to finite objects and to telecentricity in object
    space, Applied Optics 47, 5691-5700 (2008),
    http://dx.doi.org/10.1364/AO.47.005691).

    And with generalizations to arbitrary orders (see simplex.py and
    simplex_accel.pyx).
    """
    def __init__(self, system, kmax=3, wavelength=0):
        super(PolyTrace, self).__init__(system)
        self.kmax = kmax
        self.l = self.system.wavelengths[wavelength]
        self.allocate()
        self.rays()
        self.propagate()
        if self.system.object.finite:
            self.bst = self.transform()

    def allocate(self):
        super(PolyTrace, self).allocate()
        self.Simplex = make_simplex(3, self.kmax)
        n = self.length
        self.n = np.empty(n)
        self.stvwof = np.empty((n, 6, self.Simplex.q))

    def telecentric(self):
        if not self.system.object.finite:
            return False
        if self.system.object.pupil.telecentric:
            return True
        return (abs(self.system.object.pupil.slope) >
                abs(self.system.object.slope))

    def rays(self):
        self.n[0] = self.system.refractive_index(self.l, 0)
        if self.telecentric():
            pos = 0
        else:
            pos = self.system.object.pupil.distance
        S = self.Simplex
        state = PolyState(f=S().shift(pos),
                          n=self.n[0], r=S(), p=S(), k=S(),
                          s=S().shift(1), t=S(), v=S(), w=S().shift(1), o=S())
        state.r[1], state.p[2], state.k[3] = 1, 1, 1
        self._state = state

    def propagate(self, start=1, stop=None):
        super(PolyTrace, self).propagate()
        state = self._state
        self.stvwof[start - 1] = (state.s, state.t, state.v, state.w,
                                  state.o, state.f)
        for j, state in enumerate(self.system.propagate_poly(
                state, self.l, start, stop)):
            j += start
            self.stvwof[j] = (state.s, state.t, state.v, state.w,
                              state.o, state.f)
            self.n[j] = state.n

    def transform(self, i=-1):
        assert self.system.object.finite
        r = self.system.object.pupil.radius
        a = self.system.object.pupil.slope
        c = self.system.object.slope
        telecentric = abs(a) > abs(c)
        if telecentric:
            r = -self.system.object.radius
            a, c = c, a
        m = np.array([[r**2, 0, 0], [a**2, c**2, 2*a*c], [r*a, 0, r*c]])
        st = np.dot([[r, a], [0, c]], self.stvwof[i, :2])
        bst = simplex_transform(self.Simplex.i.ravel(), self.Simplex.j, st, m)
        if telecentric:
            i, j, k = self.Simplex.j.T
            ii = self.Simplex.i[j, i, k]
            bst = bst[::-1, ii].copy()
        return bst[0].view(self.Simplex), bst[1].view(self.Simplex)

    def st(self, i=-1):
        if self.system.object.finite:
            if i == -1:
                return self.bst
            else:
                return self.transform(i)
        else:
            s, t = self.stvwof[i, :2, :]
            return s.view(self.Simplex), t.view(self.Simplex)

    def evaluate(self, xy, ab, i=-1):
        """one-normalized xy and ab (field and pupil coordinates)"""
        xy, ab = np.atleast_2d(xy, ab)
        xy, ab = np.broadcast_arrays(xy, ab)
        if not self.system.object.finite:
            xy = xy*self.system.object.pupil.radius
            ab = ab*self.system.object.angle
        # assert xy.shape[1] == 2
        r = (xy**2).sum(1)
        p = (ab**2).sum(1)
        k = (xy*ab).sum(1)

        s, t = self.st(i)
        return s(r, p, k)[..., None]*xy + t(r, p, k)[..., None]*ab

    def buchdahl(self, s, t):
        n = "Ap Cp Bp S1p S3p S2p S6p S5p S4p".split()
        n.extend("_" + _ for _ in n)
        v = list(s[1:10]) + list(-t[1:10])
        flip = 0, 1, 3, 4, 6, 8
        for i in flip:
            v[i] *= -1
            v[i + 9] *= -1
        return list(zip(n, v))

    def seidel(self, s, t):
        n = "s1 s2 s3 s4 s5 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12".split()
        v = [s[1], -t[1], t[3]/2, s[2] - t[3]/2, -t[2],
             s[4], -t[4] - s[6]/2, -s[6]/2, t[6] + s[5], s[5], s[9],
             -t[5] - t[9]/2 - s[8]/2, -t[9]/2 - s[8]/2, s[8]/2, s[7] + t[8],
             s[7], -t[7]]
        return list(zip(n, v))

    def print_seidel(self):
        for n, v in self.seidel(*self.st()):
            yield "{:3s}: {:12.5e}".format(n, v)

    names = [
        # s/bs, t/bt [1:10]
        ("spherical aberration", "sagittal coma"),
        ("field curvature", "distortion"),
        ("meridional coma", "field curvature"),
        ("spherical aberration", "circular coma"),
        ("sagittal oblique spherical aberration",
         "meridional elliptical coma"),
        ("circular coma", "oblique spherical aberration"),
        ("field curvature", "distortion"),
        ("sagittal elliptical coma", "meridional field curvature"),
        ("sagittal oblique spherical aberration",
         "meridional elliptical coma"),
    ]

    def print_names(self):
        s, t = self.st()
        for (ns, nt), s, t, (i, j, k) in zip(self.names, s[1:], t[1:],
                                             self.Simplex.j[1:]):
            yield "s{:1d}{:1d}{:1d}{:1d}: {:37s}: {:12.5e}".format(
                self.Simplex.i[i, j, k], i, j, k, ns, s)
            yield "t{:1d}{:1d}{:1d}{:1d}: {:37s}: {:12.5e}".format(
                self.Simplex.i[i, j, k], i, j, k, nt, t)

    def print_params(self):
        yield "maximum order: {:d}".format(self.Simplex.n)
        yield "wavelength: {:g}".format(self.l/1e-9)

    def print_trace(self, components="stvwof", elements=None, cutoff=None,
                    width=12):
        for n in components:
            a = self.stvwof[:, "stvwof".index(n), :].T
            if elements is None:
                elements = range(1, a.shape[1])
            if cutoff is None:
                idx = slice(None)
            else:
                idx = self.Simplex.j.sum(1) < cutoff
            yield "{:s}".format(n.upper())
            yield "  n  i  j  k " + " ".join(
                "{:12d}".format(i) for i in elements)
            for (i, j, k), ai in zip(self.Simplex.j[idx], a[idx][:, elements]):
                i = "{:3d}{:3d}{:3d}{:3d}".format(self.Simplex.i[i, j, k],
                                                  i, j, k)
                ai = " ".join("{:12.5e}".format(j) for j in ai)
                yield "{:s} {:s}".format(i, ai)
            yield ""

    def __str__(self):
        return "\n".join(itertools.chain(
            self.print_params(), ("",),
            # self.print_trace(), ("",),
            self.print_seidel(), ("",),
            self.print_names(), ("",),
        ))
