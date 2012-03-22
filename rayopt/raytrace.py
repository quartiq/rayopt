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
	cached_property)

from .material import lambda_d

def dir_to_angles(x,y,z):
    r = array([x,y,z], dtype=np.float64)
    return r/linalg.norm(r)

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
        p = dot(t, p)
        a = dot(t, a)
        p.resize((3, n))
        a.resize((3, n))
        return Rays(positions=p.T, angles=a.T)

    def _get_end_positions(self):
        return self.positions + (self.lengths*self.angles.T).T


class ParaxialTrace(HasTraits):
    wavelength = Float
    wavelength_short = Float
    wavelength_long = Float
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

    def __init__(self, length=None, **k):
        super(ParaxialTrace, self).__init__(**k)
        if length is not None:
            self.refractive_indices = zeros((length,), dtype=np.float64)
            self.heights = zeros((length,2), dtype=np.float64)
            self.angles = zeros((length,2), dtype=np.float64)
            self.incidence = zeros((length,2), dtype=np.float64)
            self.dispersions = zeros((length,), dtype=np.float64)
            self.aberration3 = zeros((length,7), dtype=np.float64)
            self.aberration5 = zeros((length,7), dtype=np.float64)

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
