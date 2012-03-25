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
    p = Array(dtype=np.float, shape=(None, None)) # optical path length
    y = Array(dtype=np.float, shape=(3, None, None)) # height
    u = Array(dtype=np.float, shape=(3, None, None)) # angle
    i = Array(dtype=np.float, shape=(3, None, None)) # incidence

    def _constant_default(self):
        return np.empty((self.nrays,), dtype=np.float)
 
    def _scalar_default(self):
        return np.empty((self.length, self.nrays), dtype=np.float)
    
    def _vector_default(self):
        return np.empty((3, self.length, self.nrays), dtype=np.float)

    _l_default = _constant_default
    _l1_default = _constant_default
    _l2_default = _constant_default
    _n_default = _scalar_default
    _v_default = _scalar_default
    _p_default = _scalar_default
    _y_default = _vector_default
    _u_default = _vector_default
    _i_default = _vector_default

    def __init__(self, **kw):
        super(Trace, self).__init__(**kw)
        self.length = len(self.system.all)


class ParaxialTrace(Trace):
    system = Instance(System)
    
    # marginal/axial, principal/chief
    nrays = 2

    c3 = Array(dtype=np.float, shape=(7, None)) # third order aberration
    c5 = Array(dtype=np.float, shape=(7, None)) # fifth order aberration

    def _aberration_default(self):
        return np.empty((7, self.length), dtype=np.float)

    _c3_default = _aberration_default
    _c5_default = _aberration_default

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
        self.l[:] = self.system.wavelengths[0]
        self.l1[:] = min(self.system.wavelengths)
        self.l2[:] = max(self.system.wavelengths)
        self.n[0] = self.system.object.material.refractive_index(self.l)

    def __str__(self):
        t = itertools.chain(
                self.print_params(),
                self.print_trace(),
                self.print_c3(),
                )
        return "\n".join(t)

    def find_rays(self):
        if self.system.object.radius == np.inf:
            a, b, c = self.y, self.u, self.system.object.field_angle
        else:
            a, b, c = self.u, self.y, self.system.object.radius
        a[0, 0], b[0, 0] = (1, 0), (0, 1)
        r, h, k = self.to_aperture()
        a[0, 0], b[0, 0] = (r/h[0], -c*h[1]/h[0]), (0, c)

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


class Rays(HasTraits):
    # wavelength for all rays
    wavelength = Float(lambda_d)
    # refractive index we are in
    refractive_index = Float(1.)
    # start positions
    positions = Array(dtype=np.float64, shape=(None, 3))
    # angles
    angles = Array(dtype=np.float64, shape=(None, 3))
    # geometric length of the rays
    lengths = Array(dtype=np.float64, shape=(None,))
    # end positions
    end_positions = Property
    # total optical path lengths to start (including previous paths)
    optical_path_lengths = Array(dtype=np.float64, shape=(None,))

    def transform(self, t):
        n = len(self.positions)
        p = self.positions.T.copy()
        a = self.angles.T.copy()
        p.resize((4, n))
        a.resize((4, n))
        p[3,:] = 1
        p = np.dot(t, p)
        a = np.dot(t, a)
        p.resize((3, n))
        a.resize((3, n))
        return Rays(positions=p.T, angles=a.T)

    def _get_end_positions(self):
        return self.positions + (self.lengths*self.angles.T).T

    def propagate(self, rays):
        for a, b in zip([self.object] + self.elements,
                        self.elements + [self.image]):
            a_rays, rays = b.propagate(rays)
            yield a, a_rays
        yield b, rays

    def propagate_through(self, rays):
        for element, rays in self.propagate(rays):
            pass
        return rays

    def height_at_aperture(self, rays):
        for element, in_rays in self.propagate(rays):
            if isinstance(element, Aperture):
                return in_rays.end_positions[...,(0,1)]/element.radius

    def chief_and_marginal(self, height, rays,
            paraxial_chief=True,
            paraxial_marginal=True):
        assert sum(1 for e in self.elements
                if isinstance(e, Aperture)) == 1
       
        def stop_for_pos(x,y):
            # returns relative aperture height given object angles and
            # relative object height
            rays.positions, rays.angles = self.object.rays_to_height(
                    (x,y), height)
            return self.height_at_aperture(rays)[0]

        d = 1e-3 # arbitrary to get newton started, TODO: better scale

        if paraxial_chief:
            d0 = stop_for_pos(0,0)
            chief = -d*d0/(stop_for_pos(d,d)-d0)
        else:
            chief = fsolve(lambda p: stop_for_pos(*p),
                    (0,0), xtol=1e-2, epsfcn=d)

        if paraxial_marginal:
            dmarg = d/(stop_for_pos(*(chief+d))-stop_for_pos(*chief))
            marg_px, marg_py = chief+dmarg
            marg_nx, marg_ny = chief-dmarg
        else:
            marg_px = newton(lambda x: stop_for_pos(x, chief[1])[0]-1,
                    chief[0]+d)
            marg_nx = newton(lambda x: stop_for_pos(x, chief[1])[0]+1,
                    chief[0]-d)
            marg_py = newton(lambda y: stop_for_pos(chief[0], y)[1]-1,
                    chief[1]+d)
            marg_ny = newton(lambda y: stop_for_pos(chief[0], y)[1]+1,
                    chief[1]-d)

        return chief, (marg_px, marg_nx, marg_py, marg_ny)

    def get_ray_bundle(self, wavelength, height, number, **kw):
        rays = Rays(wavelength=wavelength, height=height)
        c, m = self.chief_and_marginal(height, rays, **kw)
        print c, m
        p, a = self.object.rays_for_point(height, c, m, number)
        rays.positions = p
        rays.angles = a
        return rays


