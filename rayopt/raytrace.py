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

"""
Raytracing like Spencer and Murty 1962, J Opt Soc Am 52, 6
with some improvements
"""

import itertools

import numpy as np
from scipy.optimize import (newton, fsolve)

from traits.api import (HasTraits, Float, Array, Property,
    cached_property, Instance, Int)

from .material import lambda_d
from .system import System
from .elements import Aperture


def dir_to_angles(x,y,z):
    r = np.array([x,y,z], dtype=np.float64)
    return r/np.linalg.norm(r)


class Trace(HasTraits):
    length = Int()
    nrays = Int()
    l = Array(dtype=np.float, shape=(None)) # wavelength
    l1 = Array(dtype=np.float, shape=(None)) # min l
    l2 = Array(dtype=np.float, shape=(None)) # max l
    n = Array(dtype=np.float, shape=(None, None)) # refractive index
    v = Array(dtype=np.float, shape=(None, None)) # dispersion
    p = Array(dtype=np.float, shape=(None, None)) # lengths
    y = Array(dtype=np.float, shape=(3, None, None)) # height
    u = Array(dtype=np.float, shape=(3, None, None)) # angle
    i = Array(dtype=np.float, shape=(3, None, None)) # incidence

    def allocate(self):
        self.l = np.zeros((self.nrays,), dtype=np.float)
        self.l1 = np.zeros((self.nrays,), dtype=np.float)
        self.l2 = np.zeros((self.nrays,), dtype=np.float)
        self.n = np.zeros((self.length, self.nrays), dtype=np.float)
        self.v = np.zeros((self.length, self.nrays), dtype=np.float)
        self.p = np.zeros((self.length, self.nrays), dtype=np.float)
        self.y = np.zeros((3, self.length, self.nrays), dtype=np.float)
        self.u = np.zeros((3, self.length, self.nrays), dtype=np.float)
        self.i = np.zeros((3, self.length, self.nrays), dtype=np.float)

    def __init__(self, **kw):
        super(Trace, self).__init__(**kw)
        self.length = len(self.system.all)


class ParaxialTrace(Trace):
    system = Instance(System)
    
    # marginal/axial, principal/chief
    nrays = 2

    def allocate(self):
        super(ParaxialTrace, self).allocate()
        self.c3 = np.zeros((7, self.length), dtype=np.float)
        self.c5 = np.zeros((7, self.length), dtype=np.float)

    lagrange = Property
    image_height = Property
    focal_length = Property
    focal_distance = Property
    pupil_height = Property
    pupil_position = Property
    f_number = Property
    numerical_aperture = Property
    airy_radius = Property
    magnification = Property

    def __init__(self, **k):
        super(ParaxialTrace, self).__init__(**k)
        self.allocate()
        self.l[:] = self.system.object.wavelengths[0]
        self.l1[:] = min(self.system.object.wavelengths)
        self.l2[:] = max(self.system.object.wavelengths)
        self.n[0] = self.system.object.material.refractive_index(self.l)

    def __str__(self):
        t = itertools.chain(
                self.print_params(),
                self.print_trace(),
                self.print_c3(),
                )
        return "\n".join(t)

    def find_rays(self):
        c = self.system.object.radius
        a, b = self.u, self.y
        if self.system.object.infinity:
            a, b = b, a
        eps = 1e-2
        a[0, 0], b[0, 0] = (eps, 0), (0, eps)
        r, h, k = self.to_aperture()
        a[0, 0], b[0, 0] = (r*eps/h[0], -c*h[1]/h[0]), (0, c)

    # TODO introduce aperture as max(height/radius)

    def size_elements(self):
        for i, e in enumerate(self.system.elements):
            e.radius = np.fabs(self.y[0, i+1]).sum()

    def propagate(self):
        self.find_rays()
        for i, e in enumerate(self.system.elements):
            e.propagate_paraxial(self, i+1)
            e.aberration3(self, i+1)
        self.system.image.propagate_paraxial(self, i+2)
    
    def to_aperture(self):
        for i, e in enumerate(self.system.elements):
            e.propagate_paraxial(self, i+1)
            if isinstance(e, Aperture):
                return e.radius, self.y[0, i+1], self.u[0, i+1]

    def focal_length_solve(self, f, i=None):
        # TODO only works for last surface
        if i is None:
            i = len(self.system.elements)-1
        y0, y = self.y[0, (i, i+1), 0]
        u0, u = self.u[0, i, 0], -self.y[0, 0, 0]/f
        n0, n = self.n[(i, i+1), 0]
        c = (n0*u0-n*u)/(y*(n-n0))
        self.system.elements[i].curvature = c

    def focal_plane_solve(self):
        self.system.image.origin[2] -= self.y[0, -1, 0]/self.u[0, -1, 0]

    def print_c3(self):
        sys, p = self.system, self
        # p.c3 *= -2*p.image_height*p.u[0,-1,0] # seidel
        # p.c3 *= -p.image_height/p.u[0,-1,0] # longit
        # p.c3 *= p.image_height # transverse
        yield "%2s %1s% 10s% 10s% 10s% 10s% 10s% 10s% 10s" % (
                "#", "T", "TSC", "CC", "TAC", "TPC", "DC", "TAchC", "TchC")
        for i, ab in enumerate(p.c3.swapaxes(0, 1)[1:-1]):
            yield "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g" % (
                    i+1, sys.elements[i].typestr,
                    ab[0], ab[1], ab[2], ab[3], ab[4], ab[5], ab[6])
        ab = p.c3.sum(0)
        yield "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g" % (
              " ∑", "", ab[0], ab[1], ab[2], ab[3], ab[4], ab[5], ab[6])

    def print_params(self):
        yield "lagrange: %.5g" % self.lagrange
        yield "focal length: %.5g" % self.focal_length
        yield "image height: %.5g" % self.image_height
        yield "focal distance: %.5g, %.5g" % self.focal_distance
        yield "pupil position: %.5g, %.5g" % self.pupil_position
        yield "pupil height: %.5g, %.5g" % self.pupil_height
        yield "numerical aperture: %.5g, %.5g" % self.numerical_aperture
        yield "f number: %.5g, %.5g" % self.f_number
        yield "airy radius: %.5g, %.5g" % self.airy_radius
        yield "magnification: %.5g, %.5g" % self.magnification

    def print_trace(self):
        yield "%2s %1s% 10s% 10s% 10s% 10s% 10s% 10s" % (
                "#", "T", "marg h", "marg a", "marg i", "chief h",
                "chief a", "chief i")
        for i, ((hm, hc), (am, ac), (im, ic)) in enumerate(zip(
                self.y[0], self.u[0], self.i[0])):
            yield "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g" % (
                    i, self.system.all[i].typestr, hm, am, im, hc, ac, ic)

    def _get_lagrange(self):
        return self.n[0,0]*(
                self.u[0,0,0]*self.y[0,0,1]-
                self.u[0,0,1]*self.y[0,0,0])

    def _get_focal_length(self):
        return -self.lagrange/self.n[0,0]/(
                self.u[0,0,0]*self.u[0,-2,1]-
                self.u[0,0,1]*self.u[0,-2,0])

    def _get_image_height(self):
        return self.lagrange/(self.n[-2,0]*self.u[0,-2,0])
 
    def _get_focal_distance(self):
        return (-self.y[0,1,0]/self.u[0,0,0],
                -self.y[0,-2,0]/self.u[0,-2,0])
       
    def _get_numerical_aperture(self):
        return (abs(self.n[0,0]*self.u[0,0,0]),
                abs(self.n[-2,0]*self.u[0,-2,0]))

    def _get_pupil_position(self):
        return (-self.y[0,1,1]/self.u[0,1,1],
                -self.y[0,-2,1]/self.u[0,-2,1])

    def _get_pupil_height(self):
        return (self.y[0,1,0]+
                self.pupil_position[0]*self.u[0,0,0],
                self.y[0,-2,0]+
                self.pupil_position[0]*self.u[0,-2,0])

    def _get_f_number(self):
        return (1/(2*self.numerical_aperture[0]),
                1/(2*self.numerical_aperture[1]))

    def _get_airy_radius(self):
        return (1.22*self.l[0]/(2*self.numerical_aperture[0]),
                1.22*self.l[0]/(2*self.numerical_aperture[1]))

    def _get_magnification(self):
        return ((self.n[0,0]*self.u[0,0,0])/(
                self.n[-2,0]*self.u[0,-2,0]),
                (self.n[-2,0]*self.u[0,-2,1])/(
                self.n[0,0]*self.u[0,0,1]))


class FullTrace(Trace):
    def propagate(self):
        for i, e in enumerate(self.system.elements):
            e.propagate(self, i+1)
        self.system.image.propagate(self, i+2)
 
    def rays_like_paraxial(self, p):
        self.nrays = 2
        self.allocate()
        self.l = p.l
        self.l1 = p.l1
        self.l2 = p.l2
        self.n[0] = p.n[0]
        self.y[0, 0] = p.y[0, 0]
        self.y[1, 0] = 0.
        self.y[2, 0] = 0.
        self.u[0, 0] = p.u[0, 0]/np.sqrt(1+p.u[0, 0]**2)
        self.u[1, 0] = 0.
        self.u[2, 0] = np.sqrt(1-self.u[0, 0]**2)

    def __str__(self):
        t = itertools.chain(
                #self.print_params(),
                self.print_trace(),
                #self.print_c3(),
                )
        return "\n".join(t)

    def print_trace(self):
        yield "%2s %1s% 10s% 10s% 10s% 10s% 10s" % (
                "#", "T", "height x", "height y",
                "angle x", "angle y", "length")
        for i in range(self.nrays):
            yield "ray %i" % i
            for j in range(self.length):
                yield "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g" % (
                        j, self.system.all[j].typestr, 
                        self.y[0, j, i], self.y[1, j, i],
                        self.u[0, j, i], self.u[1, j, i],
                        self.p[j, i])
