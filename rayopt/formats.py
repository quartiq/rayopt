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

from .system import System
from .elements import Spheroid, Aperture, Image, Object
from .material import air, misc, all_materials, FictionalMaterial

def system_from_array(data, material_map={}, **kwargs):
    # data is a list of (typ, radius of curvature,
    # offset from previous, clear radius, material after)
    element_map = {"O": Object, "S": Spheroid, "A": Aperture, "I": Image}
    s = System(**kwargs)
    for line in data:
        typ, mat = line[0], line[4]
        roc, off, rad = map(float, line[1:4])
        extra = line[5:]
        e = element_map[typ](*extra)
        e.radius = rad
        e.origin = (0, 0, off)
        e.curvature = roc and 1/roc or 0.
        mat = material_map.get(mat, mat)
        if mat in all_materials.db:
            m = all_materials.db[mat]
        else:
            try:
                m = FictionalMaterial(nd=float(mat))
            except ValueError:
                m = air
        e.material = m
        s.elements.append(e)
    return s

def system_from_text(data, **kwargs):
    return system_from_array(line.split()
            for line in data.splitlines() if line.strip(), **kwargs)

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
        if p[-2].upper() in all_materials.db:
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
            s.name = " ".join(args[1:-2]).strip("\"")
        elif cmd == "UNI":
            s.scale = float(args[0])*1e-3
            e = Spheroid()
            e.origin = (0,0,0)
        elif cmd == "AIR":
            e.material = air
        elif cmd == "TH":
            th = float(args[0])
            if th > 1e2:
                th = 0
        elif cmd == "AP":
            e.radius = float(args[0])
        elif cmd == "GLA":
            e.material = {"SILICA": misc["SILICA"],
                          "SFL56": schott["SFL56"],
                          "SF6": schott["SF6"],
                          "CAF2": misc["CAF2"],
                          "O_S-BSM81": ohara["S-BSM81"],}[args[0]]
        elif cmd == "AST":
            s.elements.append(Aperture(radius=e.radius, origin=(0,0,0)))
        elif cmd == "RD":
            e.curvature = 1/(float(args[0]))
        elif cmd in ("NXT", "END"):
            s.elements.append(e)
            e = Spheroid()
            e.origin = (0,0,th)
        elif cmd in ("//", "DES", "EBR", "GIH", "DLRS", "WW", "WV"):
            pass
        else:
            print cmd, "not handled", args
            continue
        #assert len(s.elements) - 1 == int(args[0])
    return s


def system_from_zemax(fil):
    s = System(elements=[Object(), Image()])
    next_pos = 0.
    a = None
    for line in fil.readlines():
        if not line.strip(): continue
        line = line.strip().split(" ", 1)
        cmd = line[0]
        args = len(line) == 2 and line[1] or ""
        if cmd == "UNIT":
            s.scale = {"MM": 1e-3}[args.split()[0]]
        elif cmd == "NAME":
            s.name = args.strip("\"")
        elif cmd == "SURF":
            e = Spheroid(origin=(0, 0, next_pos))
            s.elements.insert(-1, e)
        elif cmd == "CURV":
            e.curvature = float(args.split()[0])
        elif cmd == "DISZ":
            next_pos = float(args)
        elif cmd == "GLAS":
            args = args.split()
            name = args[0]
            if name not in all_materials.db:
                print "material not found: %s" % name
            e.material = all_materials.db.get(name, air)
        elif cmd == "DIAM":
            e.radius = float(args.split()[0])/2
        elif cmd == "STOP":
            s.elements.insert(-1, Aperture())
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
        #assert len(s.elements) - 1 == int(args[0])
    # the first element is the object, the last is the image, convert them
    s.object.radius = s.elements[1].radius
    del s.elements[1]
    s.image.radius = s.elements[-2].radius
    s.image.origin = s.elements[-2].origin
    del s.elements[-2]
    s.aperture.radius = s.elements[s.aperture_index-1].radius
    return s
