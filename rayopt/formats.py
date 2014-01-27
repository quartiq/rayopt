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

from __future__ import print_function, absolute_import, division

import json

import numpy as np
import yaml

from .system import System
from .elements import Spheroid, Aperture
from .material import air, ModelMaterial, get_material


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
    element_map = {"O": Spheroid, "S": Spheroid, "A": Aperture, "I":
            Spheroid}
    s = System(**kwargs)
    for line in data:
        typ = try_get(line, columns, "type", "S")
        extra = line[len(columns):]
        el = element_map[typ](*extra)
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
            m = get_material(mat)
            el.material = m
        s.append(el)
    return s


def system_from_text(text, *args, **kwargs):
    array = [line.split() for line in text.splitlines()]
    n = max(len(l) for l in array)
    array = [l for l in array if len(l) == n]
    return system_from_array(array, *args, **kwargs)


def system_from_oslo(fil):
    s = System()
    e = Spheroid()
    th = 0.
    for line in fil.readlines():
        p = line.split()
        if not p:
            continue
        cmd, args = p[0], p[1:]
        if cmd == "LEN":
            s.description = " ".join(args[1:-2]).strip("\"")
        elif cmd == "UNI":
            s.scale = float(args[0])*1e-3
        elif cmd == "AIR":
            e.material = air
        elif cmd == "TH":
            th = float(args[0])
            if th > 1e2:
                th = np.inf
        elif cmd == "AP":
            e.radius = float(args[0])
        elif cmd == "GLA":
            e.material = get_material(args[0])
        elif cmd == "AST":
            s.append(Aperture(radius=e.radius))
        elif cmd == "RD":
            e.curvature = 1/float(args[0])
        elif cmd in ("NXT", "END"):
            s.append(e)
            e = Spheroid()
            e.distance = th
        elif cmd in ("//", "DES", "EBR", "GIH", "DLRS", "WW", "WV"):
            pass
        else:
            print(cmd, "not handled", args)
            continue
    return s


def system_from_zemax(fil):
    s = System()
    next_pos = 0.
    for line in fil.readlines():
        if not line.strip():
            continue
        line = line.strip().split(" ", 1)
        cmd = line[0]
        args = len(line) == 2 and line[1] or ""
        if cmd == "UNIT":
            s.scale = {
                    "MM": 1e-3,
                    "INCH": 25.4e-3,
                    "IN": 25.4e-3,
                    }[args.split()[0]]
        elif cmd == "NAME":
            s.description = args.strip("\"")
        elif cmd == "SURF":
            e = Spheroid(distance=next_pos, material=air)
            s.insert(-1, e)
        elif cmd == "CURV":
            e.curvature = float(args.split()[0])
        elif cmd == "DISZ":
            next_pos = float(args)
        elif cmd == "GLAS":
            args = args.split()
            name = args[0]
            try:
                e.material = get_material(name)
            except KeyError:
                try:
                    e.material = get_material(float(args[3]), float(args[4]))
                except Exception as e:
                    print("material not found", name, e)
        elif cmd == "DIAM":
            e.radius = float(args.split()[0])/2
        elif cmd == "STOP":
            s.insert(-1, Aperture())
        elif cmd == "WAVL":
            s.wavelengths = [float(i)*1e-6 for i in args.split() if i]
        elif cmd in ("GCAT", # glass catalog names
                     "OPDX", # opd
                     "RAIM", # ray aiming
                     "CONF", # configurations
                     "ENPD", "PUPD", # pupil
                     "EFFL", # focal lengths
                     "VERS", # version
                     "MODE", # mode
                     "NOTE", # note
                     "TYPE", # surface type
                     "HIDE", # surface hide
                     "MIRR", # surface is mirror
                     "TOL", "MNUM", "MOFF", "FTYP", "SDMA", "GFAC",
                     "PUSH", "PICB", "ROPD", "PWAV", "POLS", "GLRS",
                     "BLNK", "COFN", "NSCD", "GSTD", "DMFS", "ISNA",
                     "VDSZ", "ENVD", "ZVDX", "ZVDY", "ZVCX", "ZVCY",
                     "ZVAN", "XFLN", "YFLN", "VDXN", "VDYN", "VCXN",
                     "VCYN", "VANN", "FWGT", "FWGN", "WWGT", "WWGN",
                     "WAVN", "WAVM", "XFLD", "YFLD", "MNCA", "MNEA",
                     "MNCG", "MNEG", "MXCA", "MXCG", "RGLA", "TRAC",
                     "FLAP", "TCMM", "FLOA", "PMAG", "TOTR", "SLAB",
                     "POPS", "COMM", "PZUP",
                     ):
            pass
        else:
            print(cmd, "not handled", args)
            continue
    s.aperture.radius = s[s.aperture_index - 1].radius
    return s


def system_from_yaml(text):
    dat = yaml.load(text)
    assert dat.pop("type") == "system"
    return System(**dat)


def system_to_yaml(system):
    dat = system.dict()
    return yaml.dump(dat, default_flow_style=False)


def system_from_json(text):
    dat = json.loads(text)
    assert dat.pop("type") == "system"
    return System(**dat)


def system_to_json(system):
    dat = system.dict()
    return json.dumps(dat)
