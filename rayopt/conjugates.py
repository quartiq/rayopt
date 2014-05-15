# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2014 Robert Jordens <jordens@phys.ethz.ch>
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

import numpy as np

from .utils import sinarctan, tanarcsin, public
from .name_mixin import NameMixin

# finite/infinite focal/afocal object/image
# regular/telecentric pupils
# hyperhemispheric objects
# hyperhemispheric pupils

# aperture: radius, object na, slope, image na, working fno
# field: object angle, object radius, image radius
# conjugate: object distance, image distance, magnification

# entrance_radius, entrance_distance, pupil_distance
# pupil_radius/na/fno, pupil_distance


@public
class Conjugate(NameMixin):
    _default_type = "infinite"
    finite = None

    def __init__(self, refractive_index=1.,
            entrance_distance=0., entrance_radius=0.,
            pupil_distance=None):
        self.refractive_index = refractive_index
        self._entrance_distance = entrance_distance
        self.entrance_radius = entrance_radius
        self.pupil_distance = pupil_distance or entrance_distance

    @property
    def entrance_distance(self):
        return self._entrance_distance

    @entrance_distance.setter
    def entrance_distance(self, d):
        self.pupil_distance += d - self._entrance_distance
        self._entrance_distance = d

    def rescale(self, scale):
        self.pupil_distance *= scale
        self.entrance_distance *= scale
        self.entrance_radius *= scale

    def text(self):
        return [] # TODO

    def aim(self, xy, pq, z=None, a=None):
        """
        h 2d fractional object coordinate (object knows meaning)
        yp 2d fractional angular pupil coordinate (since object points
        emit into solid angles)
        
        aiming should be aplanatic (the grid is equal solid angle
        in object space) and not paraxaial (equal area in entrance
        beam plane)
        
        z pupil distance from "surface 0 apex" (also infinite object)
        a pupil aperture (also for infinite object, then from z=0)
        """
        raise NotImplementedError
        if z is None:
            z = self.pupil_distance
        if a is None:
            a = self.pupil_radius
        yo, yp = np.broadcast_arrays(*np.atleast_2d(yo, yp))
        n = yo.shape[0]
        uz = np.array((0., 0., z))
        if self.finite:
            # do not take yo as angular fractional as self.radius is
            # not angular eigher. This does become problematic if object
            # is finite and hyperhemispherical. TODO: improve Spheroid
            y = np.zeros((n, 3))
            y[:, :2] = -yo*self.radius
            y[:, 2] = self.surface_sag(y)
            u = uz - y
        else:
            # lambert azimuthal equal area
            # planar coords
            yo = yo*2*np.sin(self.angular_radius/2)
            yo2 = np.square(yo).sum(1)
            u = np.empty((n, 3))
            u[:, :2] = yo*np.sqrt(1 - yo2[:, None]/4)
            u[:, 2] = 1 - yo2/2
            y = uz - z*u # have rays start on sphere around pupil center
        if z < 0:
            u *= -1
        usag = np.cross(u, uz)
        usagn = np.sqrt(np.square(usag).sum(1))[:, None]
        usagnz = usagn == 0.
        usag = np.where(usagnz, (1., 0, 0), usag)
        usagn = np.where(usagnz, 1., usagn)
        usag /= usagn
        umer = np.cross(u, usag)
        umer /= np.sqrt(np.square(umer).sum(1))[:, None]
        # umer /= np.sqrt(np.square(umer).sum(1)) by construction
        # lambert azimuthal equal area
        # yp is relative angular X and Y pupil coords (aplanatic)
        #yp = yp*np.tan(a)*z
        yp = yp*2*np.sin(a/2)
        yp2 = np.square(yp).sum(1)[:, None]
        # unit vector to pupil point from (0, 0, 0)
        #up = np.empty((n, 3))
        #up[:, :2] = np.sqrt(1 - yp2/4)*yp
        #up[:, 2] = 1 - yp2/2
        yp *= np.sqrt(1 - yp2/4)/(1 - yp2/2)*z
        yp = usag*yp[:, 0, None] + umer*yp[:, 1, None]
        if self.finite:
            u += yp
            u /= np.sqrt(np.square(u).sum(1))[:, None]
        else:
            y += yp
            # u is normalized
        return y, u


@public
@Conjugate.register
class FiniteConjugate(Conjugate):
    _type = "finite"
    finite = True

    def __init__(self, radius=0., na=None, fno=None, slope=None,
            pupil_radius=None, **kwargs):
        super(FiniteConjugate, self).__init__(**kwargs)
        self.radius = radius
        if na is not None:
            self.na = na
        if fno is not None:
            self.fno = fno
        if slope is not None:
            self.slope = slope
        if pupil_radius is not None:
            self.pupil_radius = pupil_radius

    def dict(self):
        dat = super(FiniteConjugate, self).dict()
        if self.radius:
            dat["radius"] = float(self.radius)
        return dat

    def rescale(self, scale):
        super(FiniteConjugate, self).rescale(scale)
        self.radius *= scale

    @property
    def pupil_radius(self):
        return self.entrance_radius*(self.pupil_distance/
                self.entrance_distance)

    @pupil_radius.setter
    def pupil_radius(self, p):
        self.entrance_radius = p*(self.entrance_distance/
                self.pupil_distance)

    @property
    def slope(self):
        return self.entrance_radius/self.entrance_distance

    @property
    def na(self):
        return self.refractive_index*sinarctan(self.slope)

    @property
    def fno(self):
        return 1/(2*self.na)

    @slope.setter
    def slope(self, slope):
        self.entrance_radius = slope*self.entrance_distance

    @na.setter
    def na(self, na):
        self.slope = tanarcsin(na/self.refractive_index)

    @fno.setter
    def fno(self, fno):
        self.na = 1/(2*fno)

    def aim(self, yo, yp, z=None, a=None, surface=None):
        if z is None:
            z = self.pupil_distance
        if a is None:
            a = self.pupil_radius
        yo, yp = np.broadcast_arrays(*np.atleast_2d(yo, yp))
        yp = z*np.tan(yp*np.arctan2(a, z))
        y = np.zeros((yo.shape[0], 3))
        y[..., :2] = -yo*self.radius
        if surface:
            y[..., 2] = -surface.surface_sag(y)
        u = (0, 0, z) - y
        u[..., :2] += yp[..., :2]
        u /= np.sqrt(np.square(u).sum(1))[..., None]
        if z < 0:
            u *= -1
        return y, u


@public
@Conjugate.register
class InfiniteConjugate(Conjugate):
    _type = "infinite"
    finite = False

    def __init__(self, angle=0., pupil_radius=None, **kwargs):
        super(InfiniteConjugate, self).__init__(**kwargs)
        self.angle = angle
        if pupil_radius is not None:
            self.pupil_radius = pupil_radius

    def dict(self):
        dat = super(InfiniteConjugate, self).dict()
        if self.angle:
            dat["angle"] = float(self.angle)
        return dat

    @property
    def pupil_radius(self):
        return self.entrance_radius

    @pupil_radius.setter
    def pupil_radius(self, p):
        self.entrance_radius = p

    def aim(self, yo, yp, z=None, a=None, surface=None):
        if z is None:
            z = self.pupil_distance
        if a is None:
            a = self.pupil_radius
        yo, yp = np.broadcast_arrays(*np.atleast_2d(yo, yp))
        u = np.zeros((yo.shape[0], 3))
        u[..., :2] = np.sin(yo*self.angle)
        u2 = np.square(u[..., :2]).sum(-1)
        u[..., 2] = np.sqrt(1 - u2)
        y = -z*u
        y[..., :2] += a*yp
        y[..., 2] += z
        if surface:
            y += surface.intercept(y, u)*u
        return y, u
