# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2014 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

import numpy as np

from .utils import (sinarctan, tanarcsin, public, sagittal_meridional,
                    normalize)
from .name_mixin import NameMixin
from .pupils import Pupil, RadiusPupil

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

    def __init__(self, pupil=None, projection="rectilinear",
                 update_radius=False):
        if pupil is None:
            self.pupil = RadiusPupil(radius=0.)
        else:
            self.pupil = Pupil.make(pupil)
        self.projection = projection
        self.update_radius = update_radius

    def text(self):
        if self.projection != "rectilinear":
            yield "Projection: %s" % self.projection
        if self.update_radius:
            yield "Update Radius: %s" % self.update_radius
        yield "Pupil:"
        for _ in self.pupil.text():
            yield "  %s" % _

    def dict(self):
        dat = super(Conjugate, self).dict()
        dat["pupil"] = self.pupil.dict()
        if self.projection != "rectilinear":
            dat["projection"] = self.projection
        return dat

    @property
    def wideangle(self):
        # FIXME: elaborate this
        return self.projection != "rectilinear"

    def rescale(self, scale):
        self.pupil.rescale(scale)

    def aim(self, xy, pq, z=None, a=None):
        """
        xy 2d fractional xy object coordinate (object knows meaning)
        pq 2d fractional sagittal/meridional pupil coordinate

        aiming should be aplanatic (the grid is by solid angle
        in object space) and not paraxaxial (equal area in entrance
        beam plane)

        z pupil distance from "surface 0 apex" (also for infinite object)
        a pupil aperture (also for infinite object or telecentric pupils,
        then from z=0)

        if z, a are not provided they are takes from the (paraxial data) stored
        in object/pupil
        """
        raise NotImplementedError


@public
@Conjugate.register
class FiniteConjugate(Conjugate):
    _type = "finite"
    finite = True

    def __init__(self, radius=0., **kwargs):
        super(FiniteConjugate, self).__init__(**kwargs)
        self.radius = radius

    @property
    def point(self):
        return not self.radius

    def dict(self):
        dat = super(FiniteConjugate, self).dict()
        if self.radius:
            dat["radius"] = float(self.radius)
        return dat

    def text(self):
        yield "Radius: %.3g" % self.radius
        for _ in super(FiniteConjugate, self).text():
            yield _

    def update(self, radius, pupil_distance, pupil_radius):
        self.pupil.update(pupil_distance, pupil_radius)
        if self.update_radius:
            self.radius = radius

    def rescale(self, scale):
        super(FiniteConjugate, self).rescale(scale)
        self.radius *= scale

    @property
    def slope(self):
        return self.radius/self.pupil.distance

    @slope.setter
    def slope(self, c):
        self.radius = self.pupil.distance*c

    def aim(self, yo, yp=None, z=None, a=None, surface=None, filter=True):
        if z is None:
            z = self.pupil.distance
        yo = np.atleast_2d(yo)
        if yp is not None:
            if a is None:
                a = self.pupil.radius
                a = np.array(((-a, -a), (a, a)))
            a = np.arctan2(a, z)
            yp = np.atleast_2d(yp)
            yp = self.pupil.map(yp, a, filter)
            yp = z*np.tan(yp)
            yo, yp = np.broadcast_arrays(yo, yp)

        y = np.zeros((yo.shape[0], 3))
        y[..., :2] = -yo*self.radius
        if surface is not None:
            y[..., 2] = -surface.surface_sag(y)
        uz = (0, 0, z)
        if self.pupil.telecentric:
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

    def __init__(self, angle=0., angle_deg=None, **kwargs):
        super(InfiniteConjugate, self).__init__(**kwargs)
        if angle_deg is not None:
            angle = np.deg2rad(angle_deg)
        self.angle = angle

    @property
    def point(self):
        return not self.angle

    def dict(self):
        dat = super(InfiniteConjugate, self).dict()
        if self.angle:
            dat["angle"] = float(self.angle)
        return dat

    def update(self, radius, pupil_distance, pupil_radius):
        self.pupil.update(pupil_distance, pupil_radius)
        if self.update_radius:
            self.angle = np.arctan2(radius, pupil_distance)

    def text(self):
        yield "Semi-Angle: %.3g deg" % np.rad2deg(self.angle)
        for _ in super(InfiniteConjugate, self).text():
            yield _

    @property
    def slope(self):
        return tanarcsin(self.angle)

    @slope.setter
    def slope(self, c):
        self.angle = sinarctan(c)

    def map(self, yo, a):
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
            z = self.pupil.distance
        yo = np.atleast_2d(yo)
        if yp is not None:
            if a is None:
                a = self.pupil.radius
                a = np.array(((-a, -a), (a, a)))
            yp = np.atleast_2d(yp)
            yp = self.pupil.map(yp, a, filter)
            yo, yp = np.broadcast_arrays(yo, yp)
        u = self.map(yo, self.angle)
        yz = (0, 0, z)
        y = yz - z*u
        if yp is not None:
            s, m = sagittal_meridional(u, yz)
            y += yp[..., 0, None]*s + yp[..., 1, None]*m
        if surface is not None:
            y += surface.intercept(y, u)[..., None]*u
        return y, u
