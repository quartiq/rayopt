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

import numpy as np

from traits.api import (HasTraits, Str, Callable, Float, Tuple, Bool)


class Demerit(HasTraits):
    name = Str
    func = Callable
    weight = Float(1)

    def __call__(self, system, ptrace, rays):
        return self.func(system, ptrace, rays)

demerit_rms_position = Demerit(name="rms size",
    func=lambda system, ptrace, rays:
    [np.ma.masked_invalid(r.positions[...,(0,1)]).std(axis=0) for r in rays])

demerit_rms_angle = Demerit(name="rms angle",
    func=lambda system, ptrace, rays:
    [np.ma.masked_invalid(r.angles[...,(0,1)]).std(axis=0) for r in rays])

demerit_mean_angle = Demerit(name="mean angle",
    func=lambda system, ptrace, rays:
    [np.ma.masked_invalid(r.angles[...,(0,1)]).mean(axis=0) for r in rays])

demerit_aberration3 = Demerit(name="primary aberrations",
    func=lambda system, ptrace, rays:
    ptrace.aberration3.sum(0)*ptrace.image_height)


class Parameter(HasTraits):
    name = Str
    bounds = Tuple((-np.inf, np.inf))
    scale = Float

    def __init__(self, name, bounds=None, scale=1, **k):
        super(Parameter, self).__init__(name=name,
                bounds=bounds, scale=scale, **k)

    def set_value(self, system, value):
        #exec "system.%s=%s" % (self.name, value)
        setattr(system, self.name, value)

    def get_value(self, system):
        #return eval("system.%s" % self.name)
        return getattr(system, self.name)


class Constraint(HasTraits):
    equality = Bool(True)

    def __call__(self, system):
        pass


class MaterialThickness(Constraint):
    minimum = Float(1e-3)
    maximum = Float(10e-2)
    equality = False
    
    def __call__(self, system):
        r = []
        for i,e in enumerate(system.elements[:-1]):
            en = system.elements[i+1]
            if isinstance(e, Aperture):
                continue
            if isinstance(en, Aperture):
                en = system.elements[i+2]
            if e.material not in (air, vacuum):
                center = en.origin[2]
                edge = (center+
                          e.shape_func(array([(0, e.radius, 0)]))-
                          en.shape_func(array([(0, en.radius, 0)])))
                r.append(self.minimum-min(center, edge))
                r.append(max(center, edge)-self.maximum)
                print(i, center, edge)
        return array(r)
