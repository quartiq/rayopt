# -*- coding: utf8 -*-
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

from __future__ import print_function, absolute_import, division

import json

import numpy as np
import yaml

from .system import System
from .elements import Spheroid
from .material import air, Material


def try_get(line, columns, field, default=None):
    v = default
    if field in columns:
        v = line[columns.index(field)]
        try:
            v = float(v)
        except ValueError:
            pass
    return v


def system_from_array(data,
        columns="type roc distance diameter material".split(), shifts={},
        material_map={}, **kwargs):
    data = np.array(data)
    assert data.ndim == 2
    for k, v in shifts.items():
        i = columns.index(k)
        data[:, i] = np.roll(data[:, i], v)
    s = System(**kwargs)
    for line in data:
        typ = try_get(line, columns, "type", "S")
        extra = line[len(columns):]
        el = Spheroid()
        s.append(el)
        if typ == "A":
            s.aperture = el
        curv = try_get(line, columns, "curvature")
        if curv is None:
            roc = try_get(line, columns, "roc", 0.)
            if roc == 0:
                curv = 0.
            else:
                curv = 1./roc
        if hasattr(el, "curvature"):
            el.curvature = curv
        el.distance = try_get(line, columns, "distance", 0.)
        el.radius = (try_get(line, columns, "radius", 0.) or
                try_get(line, columns, "diameter", 0.)/2.)
        if typ == "O":
            el.angular_radius = el.radius # default to infinite
        if hasattr(el, "material"):
            mat = try_get(line, columns, "material")
            mat = material_map.get(mat, mat)
            m = Material.make(mat)
            el.material = m
    return s


def system_from_text(text, *args, **kwargs):
    array = [line.split() for line in text.splitlines()]
    n = max(len(l) for l in array)
    array = [l for l in array if len(l) == n]
    return system_from_array(array, *args, **kwargs)


def system_from_yaml(text):
    dat = yaml.load(text)
    assert dat.pop("type", "system") == "system"
    return System(**dat)


def system_to_yaml(system):
    dat = system.dict()
    return yaml.dump(dat) #, default_flow_style=False)


def system_from_json(text):
    dat = json.loads(text)
    assert dat.pop("type", "system") == "system"
    return System(**dat)


def system_to_json(system):
    dat = system.dict()
    return json.dumps(dat)
