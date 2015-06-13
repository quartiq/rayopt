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

from .utils import (sinarctan, tanarcsin, public, sagittal_meridional,
        normalize, normalize_z)
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
        self._pupil_distance = pupil_distance

    def dict(self):
        dat = super(Conjugate, self).dict()
        if self._pupil_distance is not None:
            dat["pupil_distance"] = float(self._pupil_distance)
        return dat

    @property
    def entrance_distance(self):
        return self._entrance_distance

    @entrance_distance.setter
    def entrance_distance(self, d):
        if self._pupil_distance is not None:
            self._pupil_distance += d - self._entrance_distance
        self._entrance_distance = d

    @property
    def pupil_distance(self):
        if self._pupil_distance is not None:
            return self._pupil_distance
        else:
            return self._entrance_distance

    @pupil_distance.setter
    def pupil_distance(self, p):
        self._pupil_distance = p

    def rescale(self, scale):
        if self._pupil_distance is not None:
            self._pupil_distance *= scale
        self._entrance_distance *= scale
        self.entrance_radius *= scale

    def text(self):
        yield "Index: %.3g" % self.refractive_index
        yield "Entrance: %.3g dia at %.3g" % (2*self.entrance_radius,
                self.entrance_distance)
        if self._pupil_distance:
            yield "Pupil: %.3g dia at %.3g" % (2*self.pupil_radius,
                    self.pupil_distance)

    def map_pupil(self, y, a, filter=True):
        # a = [[-sag, -mer], [+sag, +mer]]
        am = np.fabs(a).max()
        y = np.atleast_2d(y)*am
        if filter:
            c = np.sum(a, axis=0)/2
            d = np.diff(a, axis=0)/2
            r = ((y - c)**2/d**2).sum(1)
            y = y[r <= 1]
        return y

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


@public
@Conjugate.register
class FiniteConjugate(Conjugate):
    _type = "finite"
    finite = True

    def __init__(self, radius=0., na=None, fno=None, slope=None,
            pupil_radius=None, telecentric=False, **kwargs):
        super(FiniteConjugate, self).__init__(**kwargs)
        self.radius = radius
        self._na = na
        self.telecentric = telecentric
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
        if self._na is not None:
            dat["na"] = float(self._na)
        if self.telecentric:
            dat["telecentric"] = self.telecentric
        return dat

    def text(self):
        for _ in super(FiniteConjugate, self).text():
            yield _
        yield "Radius: %.3g" % self.radius
        if self._na is not None:
            yield "NA: %.3g" % self.na
        if self.telecentric:
            yield "Telecentric: True"

    def rescale(self, scale):
        super(FiniteConjugate, self).rescale(scale)
        self.radius *= scale

    @property
    def na(self):
        if self._na is not None:
            return self._na
        else:
            return self.refractive_index*sinarctan(
                    self.entrance_radius/self.entrance_distance)

    @na.setter
    def na(self, na):
        self._na = na

    @property
    def pupil_radius(self):
        return self.slope*self.pupil_distance

    @pupil_radius.setter
    def pupil_radius(self, p):
        self.slope = p/self.pupil_distance

    @property
    def slope(self):
        return tanarcsin(self.na/self.refractive_index)

    @slope.setter
    def slope(self, slope):
        self.na = self.refractive_index*sinarctan(slope)

    @property
    def fno(self):
        return 1/(2*self.na)

    @fno.setter
    def fno(self, fno):
        self.na = 1/(2*fno)

    @property
    def height(self):
        return self.radius

    @height.setter
    def height(self, h):
        self.radius = h

    @property
    def chief_slope(self):
        return self.radius/self.pupil_distance

    @chief_slope.setter
    def chief_slope(self, c):
        self.radius = abs(self.pupil_distance*c)

    def aim(self, yo, yp=None, z=None, a=None, surface=None, filter=True):
        if z is None:
            z = self.pupil_distance
        yo = np.atleast_2d(yo)
        if yp is not None:
            if a is None:
                a = self.pupil_radius
                a = np.array(((-a, -a), (a, a)))
            a = np.arctan2(a, z)
            yp = np.atleast_2d(yp)
            yp = self.map_pupil(yp, a, filter)
            yp = z*np.tan(yp)
            yo, yp = np.broadcast_arrays(yo, yp)

        y = np.zeros((yo.shape[0], 3))
        y[..., :2] = -yo*self.radius
        if surface:
            y[..., 2] = -surface.surface_sag(y)
        uz = (0, 0, z)
        if self.telecentric:
            u = uz
        else:
            u = uz - y
        if yp is not None:
            s, m = sagittal_meridional(u, uz)
            u += yp[..., 0, None]*s + yp[..., 1, None]*m
        normalize(u)
        if z < 0:
            u *= -1
        return y, u


@public
@Conjugate.register
class InfiniteConjugate(Conjugate):
    _type = "infinite"
    finite = False

    def __init__(self, angle=0., angle_deg=None, pupil_radius=None,
            projection="rectilinear", wideangle=None, **kwargs):
        super(InfiniteConjugate, self).__init__(**kwargs)
        if angle_deg is not None:
            angle = np.deg2rad(angle_deg)
        self.angle = angle
        if wideangle is None:
            wideangle = self.angle > np.deg2rad(45)
        self.wideangle = wideangle
        self.projection = projection
        self._pupil_radius = pupil_radius

    def dict(self):
        dat = super(InfiniteConjugate, self).dict()
        if self.angle:
            dat["angle"] = float(self.angle)
        if self._pupil_radius is not None:
            dat["pupil_radius"] = float(self._pupil_radius)
        if self.projection != "rectilinear":
            dat["projection"] = self.projection
        if self.wideangle:
            dat["wideangle"] = self.wideangle
        return dat

    def text(self):
        for _ in super(InfiniteConjugate, self).text():
            yield _
        yield "Semi-Angle: %.3g" % np.rad2deg(self.angle)
        if self.projection is not "rectilinear":
            yield "Projection: %s" % self.projection
        if self.wideangle:
            yield "Wideangle: True"

    @property
    def pupil_radius(self):
        if self._pupil_radius is not None:
            return self._pupil_radius
        else:
            return self.entrance_radius

    @pupil_radius.setter
    def pupil_radius(self, p):
        self._pupil_radius = p

    @property
    def height(self):
        return tanarcsin(self.radius)*self.pupil_distance

    @height.setter
    def height(self, h):
        self.radius = sinarctan(h/self.pupil_distance)

    def map_object(self, yo, a):
        p = self.projection
        n = yo.shape[0]
        if p == "rectilinear":
            y = yo*np.tan(a)
            u = np.hstack((y, np.ones((n, 1))))
            u /= np.sqrt(np.square(u).sum(-1))[:, None]
        elif p == "stereographic":
            y = yo*(2*np.tan(a/2))
            r = np.square(y).sum(-1)[:, None]/4
            u = np.hstack((y, 1 - r))/(r + 1)
        elif p == "equisolid":
            y = yo*(2*np.sin(a/2))
            r = np.square(y).sum(-1)[:, None]
            u = np.hstack((y*np.sqrt(1 - r/4), 1 - r/2))
        elif p == "orthographic":
            y = yo*np.sin(a)
            r = np.square(y).sum(-1)[:, None]
            u = np.hstack((y, np.sqrt(1 - r)[:, None]))
        elif p == "equidistant":
            y = yo*a
            b = np.square(y).sum(-1) > (np.pi/2)**2
            y = np.sin(y)
            z = np.sqrt(np.square(y).sum(-1))
            z = np.where(b, -z, z)[:, None]
            u = np.hstack((y, z))
        return u

    def aim(self, yo, yp=None, z=None, a=None, surface=None, filter=True):
        if z is None:
            z = self.pupil_distance
        yo = np.atleast_2d(yo)
        if yp is not None:
            if a is None:
                a = self.pupil_radius
                a = np.array(((-a, -a), (a, a)))
            yp = np.atleast_2d(yp)
            yp = self.map_pupil(yp, a, filter)
            yo, yp = np.broadcast_arrays(yo, yp)
        u = self.map_object(yo, self.angle)
        yz = (0, 0, z)
        y = yz - z*u
        if yp is not None:
            s, m = sagittal_meridional(u, yz)
            y += yp[..., 0, None]*s + yp[..., 1, None]*m
        if surface:
            y += surface.intercept(y, u)[..., None]*u
        return y, u
