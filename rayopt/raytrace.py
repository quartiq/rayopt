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

import numpy as np

from traits.api import (HasTraits, Float, Array, Property,
    cached_property, Any, Instance)

from .material import lambda_d
from .system import System

def dir_to_angles(x,y,z):
    r = np.array([x,y,z], dtype=np.float64)
    return r/np.linalg.norm(r)


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



class ParaxialTrace(HasTraits):
    system = Instance(System)

    refractive_indices = Array(dtype=np.float64, shape=(None,))
    dispersions = Array(dtype=np.float64, shape=(None,))

    # marginal/axial,
    # principal/chief
    heights = Array(dtype=np.float64, shape=(None,2))
    angles = Array(dtype=np.float64, shape=(None,2))
    incidence = Array(dtype=np.float64, shape=(None,2))

    aberration3 = Array(dtype=np.float64, shape=(None,7))
    aberration5 = Array(dtype=np.float64, shape=(None,7))

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
        system = self.system
        if system is None:
            return
        length = len(system.elements)+2
        self.refractive_indices = np.zeros((length,), dtype=np.float64)
        self.heights = np.zeros((length,2), dtype=np.float64)
        self.angles = np.zeros((length,2), dtype=np.float64)
        self.incidence = np.zeros((length,2), dtype=np.float64)
        self.dispersions = np.zeros((length,), dtype=np.float64)
        self.aberration3 = np.zeros((length,7), dtype=np.float64)
        self.aberration5 = np.zeros((length,7), dtype=np.float64)
        self.trace()

    def propagate_paraxial(self):
        for i,e in enumerate(self.system.elements):
            e.propagate_paraxial(i+1, self)
            e.aberration3(i+1, self)
        self.system.image.propagate_paraxial(i+2, self)
    
    def height_at_aperture_paraxial(self):
        for i,e in enumerate(self.system.elements):
            e.propagate_paraxial(i+1, self)
            if isinstance(e, Aperture):
                return rays.heights[i+1]

    def trace(self):
        sys, p = self.system, self
        p.wavelength = sys.wavelengths[0]
        p.wavelength_long = max(sys.wavelengths)
        p.wavelength_short = min(sys.wavelengths)
        p.refractive_indices[0] = sys.object.material.refractive_index(
                p.wavelength)
        #p.heights[0], p.angles[0] = (18.5, -6.3), (0, .25) # photo
        #p.heights[0], p.angles[0] = (6.25, -7.102), (0, .6248) # dbl gauss
        #p.heights[0], p.angles[0] = (0, -.15), (0.25, -.0004) # k_z_i
        #p.heights[0], p.angles[0] = (5, 0), (0, .01) # k_z_o
        #p.heights[0], p.angles[0] = (sys.object.radius, 0), (0, .5) # schwarzschild
        if sys.object.radius == np.inf:
            p.heights[0] = (6.25, -7.1)
            p.angles[0] = (0, sys.object.field_angle)
        else:
            p.heights[0] = (sys.object.radius, 0)
            p.angles[0] = (0, .3)
        #print "h at aperture:", sys.height_at_aperture_paraxial(p)
        self.propagate_paraxial()
        #print "heights:", p.heights
        #print "angles:", p.angles
        #print "incidences:", p.incidence
        # p.aberration3 *= -2*p.image_height*p.angles[-1,0] # seidel
        # p.aberration3 *= -p.image_height/p.angles[-1,0] # longit
        # p.aberration3 *= p.image_height # transverse
        s = ""
        s += "%2s %1s% 10s% 10s% 10s% 10s% 10s% 10s% 10s\n" % (
                "#", "T", "TSC", "CC", "TAC", "TPC", "DC", "TAchC", "TchC")
        for i in range(1,len(p.aberration3)-1):
            ab = p.aberration3[i]
            s += "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g\n" % (
                    i-1, sys.elements[i-1].typestr,
                    ab[0], ab[1], ab[2], ab[3], ab[4], ab[5], ab[6])
        ab = p.aberration3.sum(0)
        s += "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g\n" % (
              " ∑", "", ab[0], ab[1], ab[2], ab[3], ab[4], ab[5], ab[6])
        print s
        print "lagrange:", p.lagrange
        print "focal length:", p.focal_length
        print "image height:", p.image_height
        print "numerical aperture:", p.numerical_aperture
        print "focal distance:", p.focal_distance
        print "pupil position:", p.pupil_position
        print "pupil height:", p.pupil_height
        print "f number:", p.f_number
        print "airy radius:", p.airy_radius
        print "magnification:", p.magnification

    def print_trace(self):
        print "%2s %1s% 10s% 10s% 10s% 10s% 10s% 10s" % (
                "#", "T", "marg h", "marg a", "marg i", "chief h",
                "chief a", "chief i")
        for i, ((hm, hc), (am, ac), (im, ic)) in enumerate(zip(
                self.heights, self.angles, self.incidence)):
            print "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g" % (
                    i, self.system.all[i].typestr, hm, am, im, hc, ac, ic)

    def _get_lagrange(self):
        return self.refractive_indices[0]*(
                self.angles[0,0]*self.heights[0,1]-
                self.angles[0,1]*self.heights[0,0])

    def _get_focal_length(self):
        return -self.lagrange/self.refractive_indices[0]/(
                self.angles[0,0]*self.angles[-2,1]-
                self.angles[0,1]*self.angles[-2,0])

    def _get_image_height(self):
        return self.lagrange/(self.refractive_indices[-2]*
                self.angles[-2,0])
 
    def _get_focal_distance(self):
        return (-self.heights[1,0]/self.angles[0,0],
                -self.heights[-2,0]/self.angles[-2,0])
       
    def _get_numerical_aperture(self):
        return (abs(self.refractive_indices[0]*self.angles[0,0]),
                abs(self.refractive_indices[-2]*self.angles[-2,0]))

    def _get_pupil_position(self):
        return (-self.heights[1,1]/self.angles[1,1],
                -self.heights[-2,1]/self.angles[-2,1])

    def _get_pupil_height(self):
        return (self.heights[1,0]+
                self.pupil_position[0]*self.angles[0,0],
                self.heights[-2,0]+
                self.pupil_position[0]*self.angles[-2,0])

    def _get_f_number(self):
        #return self.focal_length/(2*self.entrance_pupil_height)
        return (self.refractive_indices[-2]/(
                2*self.numerical_aperture[0]),
        #return self.focal_length/(2*self.exit_pupil_height)
               self.refractive_indices[0]/(
                2*self.numerical_aperture[1]))

    def _get_airy_radius(self):
        return (1.22*self.wavelength/(2*self.numerical_aperture[0]),
                1.22*self.wavelength/(2*self.numerical_aperture[1]))

    def _get_magnification(self):
        return ((self.refractive_indices[0]*self.angles[0,0])/(
                self.refractive_indices[-2]*self.angles[-2,0]),
                (self.refractive_indices[-2]*self.angles[-2,1])/(
                self.refractive_indices[0]*self.angles[0,1]))
