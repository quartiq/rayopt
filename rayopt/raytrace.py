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


class ParaxialTrace(HasTraits):
    system = Any # Instance(System)

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
    focal_length = Property
    back_focal_length = Property
    front_focal_length = Property
    image_height = Property
    entrance_pupil_height = Property
    entrance_pupil_position = Property
    exit_pupil_height = Property
    exit_pupil_position = Property
    front_f_number = Property
    back_f_number = Property
    back_numerical_aperture = Property
    front_numerical_aperture = Property
    front_airy_radius = Property
    back_airy_radius = Property
    magnification = Property
    angular_magnification = Property

    def __init__(self, system=None, **k):
        super(ParaxialTrace, self).__init__(**k)
        if system is None:
	    return
	self.system = system
        length = len(system.elements)+2
        self.refractive_indices = np.zeros((length,), dtype=np.float64)
        self.heights = np.zeros((length,2), dtype=np.float64)
        self.angles = np.zeros((length,2), dtype=np.float64)
        self.incidence = np.zeros((length,2), dtype=np.float64)
        self.dispersions = np.zeros((length,), dtype=np.float64)
        self.aberration3 = np.zeros((length,7), dtype=np.float64)
        self.aberration5 = np.zeros((length,7), dtype=np.float64)
        self.trace()

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
            p.heights[0] = (6, -6)
            p.angles[0] = (0, sys.object.field_angle)
        else:
            p.heights[0] = (sys.object.radius, 0)
            p.angles[0] = (0, .3)
        #print "h at aperture:", sys.height_at_aperture_paraxial(p)
        sys.propagate_paraxial(p)
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
        print "front numerical aperture:", p.front_numerical_aperture
        print "back numerical aperture:", p.back_numerical_aperture
        print "front focal length:", p.front_focal_length
        print "back focal length:", p.back_focal_length
        print "image height:", p.image_height
        print "entrance pupil position:", p.entrance_pupil_position
        print "exit pupil position:", p.exit_pupil_position
        print "entrance pupil height:", p.entrance_pupil_height
        print "exit pupil height:", p.exit_pupil_height
        print "front f number:", p.front_f_number
        print "back f number:", p.back_f_number
        print "front airy radius:", p.front_airy_radius
        print "back airy radius:", p.back_airy_radius
        print "magnification:", p.magnification
        print "angular magnification:", p.angular_magnification

    def _get_lagrange(self):
        return self.refractive_indices[0]*(
                self.angles[0,0]*self.heights[0,1]-
                self.angles[0,1]*self.heights[0,0])

    def _get_focal_length(self):
        return -self.lagrange/self.refractive_indices[0]/(
                self.angles[0,0]*self.angles[-2,1]-
                self.angles[0,1]*self.angles[-2,0])

    def _get_front_focal_length(self):
        return -self.heights[1,0]/self.angles[0,0]

    def _get_back_focal_length(self):
        return -self.heights[-2,0]/self.angles[-2,0]

    def _get_image_height(self):
        return self.lagrange/(self.refractive_indices[-2]*
                self.angles[-2,0])
        
    def _get_back_numerical_aperture(self):
        return abs(self.refractive_indices[-2]*self.angles[-2,0])

    def _get_front_numerical_aperture(self):
        return abs(self.refractive_indices[0]*self.angles[0,0])

    def _get_entrance_pupil_position(self):
        return -self.heights[1,1]/self.angles[1,1]

    def _get_exit_pupil_position(self):
        return -self.heights[-2,1]/self.angles[-2,1]

    def _get_entrance_pupil_height(self):
        return self.heights[1,0]+\
                self.entrance_pupil_position*self.angles[0,0]

    def _get_exit_pupil_height(self):
        return self.heights[-2,0]+\
                self.entrance_pupil_position*self.angles[-2,0]

    def _get_front_f_number(self):
        #return self.focal_length/(2*self.entrance_pupil_height)
        return self.refractive_indices[-2]/(
                2*self.front_numerical_aperture)

    def _get_back_f_number(self):
        #return self.focal_length/(2*self.exit_pupil_height)
        return self.refractive_indices[0]/(
                2*self.back_numerical_aperture)

    def _get_back_airy_radius(self):
        return 1.22*self.wavelength/(2*self.back_numerical_aperture)

    def _get_front_airy_radius(self):
        return 1.22*self.wavelength/(2*self.front_numerical_aperture)

    def _get_magnification(self):
        return (self.refractive_indices[0]*self.angles[0,0])/(
                self.refractive_indices[-2]*self.angles[-2,0])

    def _get_angular_magnification(self):
        return (self.refractive_indices[-2]*self.angles[-2,1])/(
                self.refractive_indices[0]*self.angles[0,1])
