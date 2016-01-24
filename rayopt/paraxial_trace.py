# -*- coding: utf-8 -*-
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

import itertools

import numpy as np

from .utils import sinarctan, tanarcsin, public
from .raytrace import Trace


@public
class ParaxialTrace(Trace):
    # y[i] is ray height after the ith element perpendicular to the
    # excidence direction (assumes excidence and offset of the
    # next element coincide: use el.align())
    # u[i] is tan(u) angle after the ith element: always a slope.
    # a.k.a. "the paraxial u is tan u"
    #
    # calculations assume aplanatic (and not paraxial system
    # oslo also assumes aplanatic). aplanatic is equivalent to the abbe
    # sine condition, magnification is equal to optical input ray sine
    # over optical output sine for all rays:
    # m = n0 sin u0/ (nk sin uk)
    def __init__(self, system, axis=1, update=True):
        super(ParaxialTrace, self).__init__(system)
        self.axis = axis
        if update:
            self.update()

    def update(self):
        self.allocate()
        self.rays()
        self.propagate()
        self.aberrations()

    @property
    def wavelength(self):
        return self.system.wavelengths[0]

    def allocate(self):
        super(ParaxialTrace, self).allocate()
        n = self.length
        if hasattr(self, "n") and self.n.shape[0] == n:
            return
        self.n = np.empty(n)
        self.y = np.empty((n, 2))  # (idx, marginal, chief)
        self.u = np.empty((n, 2))  # n*u
        self.c = np.empty((n, 7))

    def rays(self):
        self.n[0] = n0 = self.system.refractive_index(self.wavelength, 0)
        y, u = self.y, self.u
        o = self.system.object
        if o.finite:
            y[0] = 0, -o.radius
            u[0] = n0*o.pupil.slope, n0*o.slope
        else:
            if self.system.object.wideangle:
                c = 1.
            else:
                c = tanarcsin(self.system.object.angle)
            y[0] = o.pupil.radius, -o.slope*o.pupil.distance
            u[0] = 0, n0*c

    def propagate(self, start=1, stop=None):
        super(ParaxialTrace, self).propagate()
        init = start - 1
        # FIXME not really round for gen astig...
        yu = np.vstack((self.y[init], self.y[init],
                        self.u[init], self.u[init]))
        n = self.n[init]
        for j, (yu, n) in enumerate(self.system.propagate_paraxial(
                yu, n, self.wavelength, start, stop)):
            j += start
            self.y[j], self.u[j] = np.vsplit(yu[self.axis::2], 2)
            self.n[j] = n

    def aberrations(self, start=1, stop=None):
        self.c[start - 1] = 0
        v = 0
        l1, l2 = min(self.system.wavelengths), max(self.system.wavelengths)
        for i, el in enumerate(self.system[start:stop]):
            i += start
            v0, v = v, el.dispersion(l1, l2)
            self.c[i] = el.aberration(self.y[i], self.u[i - 1], self.u[i],
                                      self.n[i - 1], self.n[i], v0, v)

    @property
    def transverse3(self):
        # transverse image seidel (like oslo)
        return self.c*self.height[1]

    @property
    def track_length(self):
        """distance from first to last surface"""
        return self.track[-2] - self.track[1]

    @property
    def height(self):
        """object and image ray height"""
        return np.fabs(self.y[(0, -1), 1])
        # self.lagrange/(self.n[-2]*self.u[-2,0])

    @property
    def pupil_distance(self):
        """pupil location relative to first/last surface"""
        return -self.y[(1, -2), 1]/self.u[(0, -2), 1]*self.n[(0, -2), ]

    @property
    def pupil_height(self):
        p = self.pupil_distance
        return np.fabs(self.y[(1, -2), 0] +
                       p*self.u[(0, -2), 0]/self.n[(0, -2), ])

    @property
    def lagrange(self):
        u, y = self.u[0], self.y[0]
        return u[0]*y[1] - u[1]*y[0]

    @property
    def focal_length(self):
        """signed distance from principal planes to foci (infinite
        conjugates), Malacara1989 p27 2.41, 2.42: F-P"""
        f = self.lagrange/(
                self.u[0, 1]*self.u[-2, 0] -
                self.u[0, 0]*self.u[-2, 1])
        return f*self.n[(-2, 0), ]*(-1, 1)

    @property
    def focal_distance(self):
        """front/back focal distance relative to first/last surface
        Malacara1989 p27 2.43 2.44, F-V"""
        c = self.focal_length/self.lagrange/self.n[(-2, 0), ]
        fd = (self.y[(1, -2), 1]*self.u[(-2, 0), 0] -
              self.y[(1, -2), 0]*self.u[(-2, 0), 1])*c
        return fd

    @property
    def principal_distance(self):
        """distance from first/last surface to principal planes
        Malacara1989: P-V"""
        return self.focal_distance - self.focal_length

    @property
    def nodal_distance(self):
        """nodal points relative to first/last surfaces
        Malacara1989, N-V"""
        return self.focal_length[::-1] + self.focal_distance

    @property
    def numerical_aperture(self):
        n = self.n[(0, -2), ]
        na = n*sinarctan(self.u[(0, -2), 0]/n)
        if self.system.object.finite and self.system.image.finite:
            # use abbe sine condition assuming we are tracing from long
            # to short conjugate
            na[1] = na[0]/self.magnification[0]
        return np.fabs(na)

    @property
    def f_number(self):
        return np.fabs(self.focal_length/(2*self.pupil_height))

    @property
    def working_f_number(self):
        na = self.numerical_aperture
        return self.n[(0, -2), ]/(2*na)

    @property
    def airy_radius(self):
        na = self.numerical_aperture
        return 1.22*self.wavelength/(2*na)/self.system.scale

    @property
    def rayleigh_range(self):
        r = self.airy_radius
        return np.pi*r**2/self.wavelength*self.system.scale

    @property
    def magnification(self):
        mt = self.u[0, 0]/self.u[-2, 0]
        ma = self.u[-2, 1]*self.n[0]/(self.u[0, 1]*self.n[-2])
        return np.array([mt, ma])

    @property
    def number_of_points(self):
        """number of resolvable independent diffraction points
        (assuming no aberrations)"""
        return 4*self.lagrange**2/self.wavelength**2

    @property
    def eigenrays(self):
        n, m = self.system.paraxial_matrix(self.wavelength)
        e, v = np.linalg.eig(m)
        return e, v

    def print_transverse3(self):
        return self.print_coeffs(self.transverse3,
                                 "SA3 CMA3 AST3 PTZ3 DIS3 TACHC TCHC".split())

    def print_params(self):
        yield "lagrange: %.5g" % self.lagrange
        yield "track length: %.5g" % self.track_length
        yield "object, image height: %s" % self.height
        yield "front, back focal length (from PP): %s" % self.focal_length
        yield "entry, exit pupil height: %s" % self.pupil_height
        yield "entry, exit pupil distance: %s" % self.pupil_distance
        yield "front, back focal distance: %s" % self.focal_distance
        yield "front, back principal distance: %s" % self.principal_distance
        yield "front, back nodal distance: %s" % self.nodal_distance
        yield "front, back numerical aperture: %s" % self.numerical_aperture
        yield "front, back f number: %s" % self.f_number
        yield "front, back working f number: %s" % self.working_f_number
        yield "front, back airy radius: %s" % self.airy_radius
        yield "transverse, angular magnification: %s" % self.magnification

    def print_trace(self):
        c = np.c_[self.path, self.n, self.y[:, 0], self.u[:, 0],
                  self.y[:, 1], self.u[:, 1]]
        return self.print_coeffs(
            c, "path/n/axial y/axial nu/chief y/chief nu".split("/"),
            sum=False)

    def __str__(self):
        return "\n".join(self.text())

    def text(self):
        return itertools.chain(
            self.print_params(), ("",),
            self.print_trace(), ("",),
            self.print_transverse3(), ("",),
        )

    def plot(self, ax, principals=False, pupils=False, focals=False,
             nodals=False, **kwargs):
        kwargs.setdefault("color", "black")
        # this assumes that the outgoing oa of an element
        # coincides with the incoming of the next, use align()
        y = self.y[:, :, None] * np.ones(3)
        y[:, :, 2] = self.path[:, None]
        y = self.from_axis(y, range(self.length))
        ax.plot(y[:, :, 2], y[:, :, self.axis], **kwargs)
        h = self.system.aperture.radius*1.5
        for p, flag in [
                (self.principal_distance, principals),
                (self.focal_distance, focals),
                (self.nodal_distance, nodals),
                ]:
            if flag:
                for i, pi, zi in zip((1, -1), p,
                                     (0, self.system[-1].distance)):
                    y = self.origins[i] + self.system[i].from_axis(
                            np.array([(h, h, pi-zi), (-h, -h, pi-zi)]))
                    ax.plot(y[:, 2], y[:, self.axis], **kwargs)
        if pupils:
            p = self.pupil_distance
            h = self.pupil_height
            for i, hi, pi, zi in zip((1, -1), h, p,
                                     (0, self.system[-1].distance)):
                y = np.empty((4, 3))
                y[:, 0] = y[:, 1] = -1.5, 1.5, -1, 1
                y *= hi
                y[:, 2] = pi - zi
                y = self.origins[i] + self.system[i].from_axis(y)
                y = y.reshape(2, 2, 3)
                ax.plot(y[:, :, 2], y[:, :, self.axis], **kwargs)

    def plot_yybar(self, ax, **kwargs):
        kwargs.setdefault("color", "black")
        ax.plot(self.y[:, 0], self.y[:, 1], **kwargs)

    # TODO
    # * introduce aperture at argmax(abs(y_axial)/radius)
    #   or at argmin(abs(u_axial))
    # * setting any of (obj na, entrance radius, ax slope, image na,
    # working fno), (field angle, obj height, image height), (obj dist,
    # img dist, obj pp, img pp, mag) works

    def resize(self):
        for e, y in zip(self.system[1:], self.y[1:]):
            e.radius = np.fabs(y).sum()  # marginal+chief

    def focal_length_solve(self, f, i=-2):
        assert i == -2, "only works for the last surface"  # TODO
        y0, y = self.y[(i - 1, i), 0]
        u0, u = self.u[i - 1, 0], -self.y[0, 0]/f*self.n[0]
        n0, n = self.n[(i - 1, i), ]
        c = (u - u0)/(y*(n0 - n))
        self.system[i].curvature = c

    def _focal_length_solve(self, f, i=None):  # TODO: not exact
        if i is None:
            i = len(self.system) - 2
        seq = (1, i), (i, i + 1), (i + 1, None)
        m0, m1, m2 = (self.system.paraxial_matrix(
            self.wavelength, start=a, stop=b)[1]
            [self.axis::2, self.axis::2] for a, b in seq)
        n0, n = self.n[(i - 1, i), ]
        c = -(1/(n0*f) +
              m0[1, 0]*m1[0, 0]*m2[0, 0] +
              m0[1, 0]*m1[0, 1]*m2[1, 0] +
              m0[1, 1]*m1[1, 1]*m2[1, 0]
              )/(m0[1, 1]*m2[0, 0])
        self.system[i].curvature = c/(n0 - n)*n

    def refocus(self, idx=-1):
        self.system[idx].distance = \
            -self.n[idx - 1]*self.y[idx - 1, 0]/self.u[idx - 1, 0]

    def update_conjugates(self):
        ai = self.system.stop
        r = self.system[ai].radius
        na, ma = self.system.paraxial_matrix(self.wavelength, stop=ai + 1)
        ma = ma[self.axis::2, self.axis::2]
        a, b = ma[0]
        b *= self.system[0].refractive_index(self.wavelength)
        self.system.object.update(self.system[0].radius, b/a, r/a)
        nb, mb = self.system.paraxial_matrix(self.wavelength, start=ai + 1)
        mb = mb[self.axis::2, self.axis::2]
        a, b = np.linalg.inv(mb)[0]
        b *= nb
        # m = np.dot(mb, ma)
        self.system.image.update(self.system[-1].radius, b/a, r/a)

    def update_stop(self, end="image"):
        # TODO: verify
        ai = self.system.stop
        if end == "image":
            n, m = self.system.paraxial_matrix(self.wavelength, start=ai + 1)
            m = m[self.axis::2, self.axis::2]
            m = np.linalg.inv(m)
            y, u = self.system.image.aim((0, 0), (0, -1))
        elif end == "object":
            n, m = self.system.paraxial_matrix(self.wavelength, stop=ai + 1)
            m = m[self.axis::2, self.axis::2]
            y, u = self.system.object.aim((0, 0), (0, 1))
        u = tanarcsin(u)
        y, u = np.dot(m, (y[0, 1], u[0, 1]))
        self.system[ai].radius = y
