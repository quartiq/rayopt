# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
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

from __future__ import print_function, absolute_import, division

import itertools
from collections import namedtuple

import numpy as np

from .utils import sinarctan, tanarcsin, public
from .raytrace import Trace
from .simplex import make_simplex, simplex_transform


PolyState = public(namedtuple("PolyState", "f n r p k s t v w o"))


@public
class PolyTrace(Trace):
    def __init__(self, system, kmax=7, wavelength=0):
        super(PolyTrace, self).__init__(system)
        self.kmax = kmax
        self.l = self.system.wavelengths[wavelength]
        self.allocate()
        self.rays()
        self.propagate()

    def allocate(self):
        super(PolyTrace, self).allocate()
        self.Simplex = make_simplex(3, self.kmax)
        n = self.length
        self.n = np.empty(n)
        self.stvwo = np.empty((n, 5, self.Simplex.q))

    def rays(self):
        self.n[0] = self.system[0].refractive_index(self.l)
        S = self.Simplex
        state = PolyState(f=S().shift(self.system.object.pupil_distance),
                          n=self.n[0], r=S(), p=S(), k=S(),
                          s=S().shift(1), t=S(), v=S(), w=S().shift(1), o=S())
        state.r[1], state.p[2], state.k[3] = 1, 1, 1
        self._state = state

    def propagate(self, start=1, stop=None):
        super(PolyTrace, self).propagate()
        init = start - 1
        state = self._state
        self.stvwo[init] = state.s, state.t, state.v, state.w, state.o
        for j, state in enumerate(self.system.propagate_poly(
                state, self.l, start, stop)):
            j += start
            self.stvwo[j] = state.s, state.t, state.v, state.w, state.o
            self.n[j] = state.n

    def transform(self, i=-1):
        assert self.system.object.finite
        r = self.system.object.pupil_radius
        a = self.system.object.slope
        c = self.system.object.chief_slope
        telecentric = abs(a) > abs(c)
        if telecentric:
            a, c = c, a
        m = np.array([[r**2, 0, 0], [a**2, c**2, 2*a*c], [r*a, 0, r*c]])
        st = np.dot([[r, a], [0, c]], self.stvwo[i, :2])
        bst = simplex_transform(self.Simplex.i.ravel(), self.Simplex.j, st, m)
        if telecentric:
            i, j, k = self.Simplex.j.T
            ii = self.Simplex.i[j, i, k]
            bst = bst[::-1, ii].copy()
        return bst[0].view(self.Simplex), bst[1].view(self.Simplex)

    def evaluate(self, xy, ab, i=-1):
        if self.system.object.finite:
            s, t = self.transform(i)
        else:
            s, t = self.stvwo[i, (0, 1), :]
        s, t = self.Simplex(s), self.Simplex(t)
        xy, ab = np.atleast_2d(xy, ab)
        xy, ab = np.broadcast_arrays(xy, ab)
        assert xy.shape[1] == 2
        r = (xy**2).sum(1)
        p = (ab**2).sum(1)
        k = (xy*ab).sum(1)
        return s(r, p, k)[..., None]*xy + t(r, p, k)[..., None]*ab

    def print_params(self):
        yield "maximum order: {:d}".format(self.Simplex.n)
        yield "wavelength: {:g}".format(self.l/1e-9)

    def print_trace(self, components="stvwo", elements=None, cutoff=None,
                    width=12):
        for n in components:
            a = self.stvwo[:, "stvwo".index(n), :].T
            if elements is None:
                elements = range(1, a.shape[1])
            if cutoff is None:
                idx = slice(None)
            else:
                idx = self.Simplex.j.sum(1) < cutoff
            yield "{:s}".format(n.upper())
            yield "  n  i  j  k " + " ".join("{:12d}".format(i) for i in elements)
            for (i, j, k), ai in zip(self.Simplex.j[idx], a[idx][:, elements]):
                i = "{:3d}{:3d}{:3d}{:3d}".format(self.Simplex.i[i, j, k],
                                                  i, j, k)
                ai = " ".join("{:12.5e}".format(j) for j in ai)
                yield "{:s} {:s}".format(i, ai)
            yield ""
        #return self.print_coeffs(
        #    c, "track/axial y/axial u/chief y/chief u".split("/"),
        #    sum=False)

    def __str__(self):
        return "\n".join(itertools.chain(
            self.print_params(), ("",),
            self.print_trace(), ("",),
        ))
