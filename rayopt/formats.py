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


import numpy as np

from .system import System
from .elements import Spheroid, Aperture, Image, Object
from .material import air, all_materials, Material


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
        columns="type roc thickness diameter material".split(),
        material_map={}, **kwargs):
    element_map = {"O": Object, "S": Spheroid, "A": Aperture, "I": Image}
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
        el.thickness = try_get(line, columns, "thickness", 0.)
        el.radius = (try_get(line, columns, "radius", 0.) or
                try_get(line, columns, "diameter", 0.)/2.)
        if hasattr(el, "material"):
            mat = try_get(line, columns, "material")
            mat = material_map.get(mat, mat)
            if type(mat) is type(1.):
                m = Material(name="%.5g" % mat, nd=mat)
            else:
                try:
                    m = all_materials[mat]
                except KeyError:
                    m = air
            el.material = m
        s.append(el)
    return s


def system_from_text(text, *args, **kwargs):
    return system_from_array(line.split()
            for line in text.splitlines() if line.strip(), *args, **kwargs)


def system_from_table(data, **kwargs):
    s = System(**kwargs)
    pos = 0.
    for line in data.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "Stop":
            s.elements.append(Aperture(
                origin=(0,0,0),
                radius=rad))
            continue
        roc = float(p[1])
        if roc == 0:
            curv = 0
        else:
            curv = 1/roc
        rad = float(p[-1])/2
        if p[-2].upper() in all_materials:
            mat = all_materials[p[-2].upper()]
        else:
            mat = air
        e = Spheroid(
            curvature=curv,
            origin=(0,0,pos),
            radius=rad,
            material=mat)
        s.elements.append(e)
        pos = float(p[2])
    return s


def system_from_oslo(fil):
    s = System()
    th = 0.
    for line in fil.readlines():
        p = line.split()
        if not p:
            continue
        cmd, args = p[0], p[1:]
        if cmd == "LEN":
            s.description = " ".join(args[1:-2]).strip("\"")
        elif cmd == "UNI":
            #s.scale = float(args[0])*1e-3
            e = Spheroid()
        elif cmd == "AIR":
            e.material = air
        elif cmd == "TH":
            th = float(args[0])
            if th > 1e2:
                th = np.inf
        elif cmd == "AP":
            e.radius = float(args[0])
        elif cmd == "GLA":
            e.material = all_materials[args[0]]
        elif cmd == "AST":
            s.append(Aperture(radius=e.radius))
        elif cmd == "RD":
            e.curvature = 1/(float(args[0]))
        elif cmd in ("NXT", "END"):
            s.append(e)
            e = Spheroid()
            e.thickness = th
        elif cmd in ("//", "DES", "EBR", "GIH", "DLRS", "WW", "WV"):
            pass
        else:
            print cmd, "not handled", args
            continue
        #assert len(s) - 1 == int(args[0])
    return s


def system_from_zemax(fil):
    s = System([Object(), Image()])
    next_pos = 0.
    a = None
    for line in fil.readlines():
        if not line.strip(): continue
        line = line.strip().split(" ", 1)
        cmd = line[0]
        args = len(line) == 2 and line[1] or ""
        if cmd == "UNIT":
            #s.scale = {"MM": 1e-3}[args.split()[0]]
            pass
        elif cmd == "NAME":
            s.description = args.strip("\"")
        elif cmd == "SURF":
            e = Spheroid(thickness=next_pos)
            s.insert(-1, e)
        elif cmd == "CURV":
            e.curvature = float(args.split()[0])
        elif cmd == "DISZ":
            next_pos = float(args)
        elif cmd == "GLAS":
            args = args.split()
            name = args[0]
            if name in all_materials:
                e.material = all_materials[name]
            else:
                print "material not found: %s" % name
                try:
                    nd = float(args[3])
                    vd = float(args[4])
                    e.material = Material(nd=nd, vd=vd)
                except:
                    e.material = air
        elif cmd == "DIAM":
            e.radius = float(args.split()[0])/2
        elif cmd == "STOP":
            s.insert(-1, Aperture())
        elif cmd == "WAVL":
            s.object.wavelengths = [float(i)*1e-6 for i in args.split() if i]
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
                     "POPS", "COMM",
                     ):
            pass
        else:
            print cmd, "not handled", args
            continue
        #assert len(s) - 1 == int(args[0])
    # the first element is the object, the last is the image, convert them
    s.object.radius = s[1].radius
    del s.elements[1]
    s.image.radius = s[-2].radius
    s.image.thickness = s[-2].thickness
    del s[-2]
    s.aperture.radius = s[s.aperture_index-1].radius
    return s
