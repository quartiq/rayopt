#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2014 Robert Jordens <robert@joerdens.org>
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


import numpy as np

from .utils import sinarctan, tanarcsin, public
from .name_mixin import NameMixin


@public
class Pupil(NameMixin):
    _default_type = "radius"

    def __init__(self, distance=1., update_distance=True,
                 update_radius=False, aim=False, telecentric=False,
                 refractive_index=1., projection="rectilinear"):
        self.distance = distance
        self.update_distance = update_distance
        self.update_radius = update_radius
        self.refractive_index = refractive_index
        self.aim = aim
        self.telecentric = telecentric
        self.projection = projection

    def rescale(self, scale):
        self.distance *= scale

    def update(self, distance, radius):
        if self.update_distance:
            self.distance = distance
        if self.update_radius:
            self.radius = radius

    def dict(self):
        dat = super().dict()
        dat["distance"] = float(self.distance)
        if not self.update_distance:
            dat["update_distance"] = self.update_distance
        if self.update_radius:
            dat["update_radius"] = self.update_radius
        if self.aim:
            dat["aim"] = self.aim
        if self.projection != "rectilinear":
            dat["projection"] = self.projection
        if self.telecentric:
            dat["telecentric"] = self.telecentric
        if self.refractive_index != 1.:
            dat["refractive_index"] = float(self.refractive_index)
        return dat

    def text(self):
        yield "Pupil Distance: %g" % self.distance
        if self.telecentric:
            yield "Telecentric: %s" % self.telecentric
        if self.refractive_index != 1.:
            yield "Refractive Index: %g" % self.refractive_index
        if self.projection != "rectilinear":
            yield "Projection: %s" % self.projection
        if not self.update_distance:
            yield "Track Distance: %s" % self.update_distance
        if self.update_radius:
            yield "Update Radius: %s" % self.update_radius
        if self.aim:
            yield "Aim: %s" % self.aim

    @property
    def radius(self):
        return self.slope*self.distance

    @property
    def slope(self):
        return self.radius/self.distance

    @property
    def na(self):
        return sinarctan(self.slope)*self.refractive_index

    @property
    def fno(self):
        return 1/(2.*self.na)

    def map(self, y, a, filter=True):
        # FIXME: projection
        # a = [[-sag, -mer], [+sag, +mer]]
        am = np.fabs(a).max()
        y = np.atleast_2d(y)*am
        if filter:
            c = np.sum(a, axis=0)/2
            d = np.diff(a, axis=0)/2
            r = ((y - c)**2/d**2).sum(1)
            y = y[r <= 1]
        return y


@public
@Pupil.register
class NaPupil(Pupil):
    _type = "na"
    na = None

    def __init__(self, na, **kwargs):
        super().__init__(**kwargs)
        self.na = na

    def dict(self):
        dat = super().dict()
        dat["na"] = float(self.na)
        return dat

    def text(self):
        yield from super().text()
        yield "NA: %g" % self.na

    @property
    def slope(self):
        return tanarcsin(self.na/self.refractive_index)

    @property
    def radius(self):
        return self.slope*self.distance

    @radius.setter
    def radius(self, r):
        self.na = self.refractive_index*sinarctan(r/self.distance)


@public
@Pupil.register
class SlopePupil(Pupil):
    _type = "slope"
    slope = None

    def __init__(self, slope, **kwargs):
        super().__init__(**kwargs)
        self.slope = slope

    def dict(self):
        dat = super().dict()
        dat["slope"] = float(self.slope)
        return dat

    def text(self):
        yield from super().text()
        yield "Slope: %g" % self.slope

    @property
    def radius(self):
        return self.slope*self.distance

    @radius.setter
    def radius(self, r):
        self.slope = r/self.distance


@public
@Pupil.register
class RadiusPupil(Pupil):
    _type = "radius"
    radius = None

    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def dict(self):
        dat = super().dict()
        dat["radius"] = float(self.radius)
        return dat

    def text(self):
        yield from super().text()
        yield "Radius: %g" % self.radius

    def rescale(self, scale):
        super().rescale(scale)
        self.radius *= scale


@public
@Pupil.register
class FnoPupil(Pupil):
    _type = "fno"
    fno = None

    def __init__(self, fno, **kwargs):
        super().__init__(**kwargs)
        self.fno = fno

    def dict(self):
        dat = super().dict()
        dat["fno"] = float(self.fno)
        return dat

    def text(self):
        yield from super().text()
        yield "F-Number: %g" % self.fno

    @property
    def slope(self):
        return tanarcsin(self.na/self.refractive_index)

    @property
    def na(self):
        return 1/(2.*self.fno)

    @property
    def radius(self):
        return self.slope*self.distance

    @radius.setter
    def radius(self, r):
        self.fno = 1/(2*self.refractive_index*sinarctan(r/self.distance))
