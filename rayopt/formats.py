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
from .material import air, misc


def system_from_table(data, scale):
    s = System(scale=scale)
    pos = 0.
    for line in data.splitlines():
        p = line.split()
        if not p:
            continue
        roc = float(p[1])
        if roc == 0:
            curv = 0
        else:
            curv = 1/roc
        if p[-1]=="Stop":
            del p[-1]
            s.elements.append(Aperture(
                origin=(0,0,0),
                radius=rad))
        rad = float(p[-1])/2
        if "Silica" in p:
            mat = misc["SILICA"]
        else:
            mat = air
        e = Spheroid(
            curvature=curv,
            origin=(0,0,pos),
            radius=rad,
            material=mat)
        s.elements.append(e)
        if s.object is None:
            s.object = Object(radius=rad, origin=(0,0,0))
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
    s = System(object=Object(), image=Image())
    next_pos = 0.
    a = None
    for line in fil.readlines():
        line = line.strip().split(" ", 1)
        cmd, args = line[0], line[1:]
        if args:
            args = args[0]
        if not cmd:
            continue

        if cmd in ("VERS", "MODE", "NOTE"):
            pass
        elif cmd == "UNIT":
            if args.split()[0] == "MM":
                s.scale = 1e-3
        elif cmd == "NAME":
            s.name = args.strip("\"")
        elif cmd == "SURF":
            e = Spheroid(origin=(0,0,next_pos))
            s.elements.append(e)
        elif cmd in ("TYPE", "HIDE", "MIRR", "SLAB", "POPS"):
            pass
        elif cmd == "CURV":
            e.curvature = float(args.split()[0])
        elif cmd == "DISZ":
            next_pos = float(args)
        elif cmd == "GLAS":
            args = args.split()
            name = args[0]
            for db in schott, ohara, infrared, misc:
                if name in db.db:
                    e.material = db[name]
            if not e.material:
                e.material = {
                        "SILICA": misc["SILICA"],
                        "CAF2": misc["CAF2"],
                        "O_S-BSM81": ohara["S-BSM81"],}[name]
        elif cmd == "COMM":
            pass
        elif cmd == "DIAM":
            args = args.split()
            e.radius = float(args[0])/2
            if a is not None:
                a.radius = e.radius
            a = None
        elif cmd == "STOP":
            a = Aperture(radius=e.radius, origin=(0,0,0))
            s.elements.append(a)
        elif cmd == "WAVN":
            s.wavelengths = [float(i)*1e-6 for i in args.split() if
                    float(i) > 0 and float(i) != .55]
        elif cmd == "XFLN":
            ohx = [float(i) for j,i in enumerate(args.split()) if
                    float(i) > 0 or j == 0]
        elif cmd == "YFLN":
            ohy = [float(i) for j,i in enumerate(args.split()) if
                    float(i) > 0 or j == 0]
        elif cmd == "ENPD":
            s.object.radius = float(args)/2
            s.object.origin = (0,0,0)
        elif cmd in ("GCAT", "OPDX", "TOL", "MNUM", "MOFF", "FTYP",
                     "SDMA", "RAIM", "GFAC", "PUSH", "PICB", "ROPD",
                     "PWAV", "POLS", "GLRS", "BLNK", "COFN", "NSCD",
                     "GSTD", "CONF", "DMFS", "ISNA", "VDSZ", "PUPD", "ENVD",
                     "ZVDX", "ZVDY", "ZVCX", "ZVCY", "ZVAN",
                     "VDXN", "VDYN", "VCXN", "VCYN", "VANN",
                     "FWGT", "FWGN", "WWGT", "WWGN",
                     "WAVL", "WAVM", "XFLD", "YFLD",
                     "MNCA", "MNEA", "MNCG", "MNEG", "MXCA", "MXCG",
                     "EFFL", "RGLA"):
            pass
        else:
            print cmd, "not handled", args
            continue
        #assert len(s.elements) - 1 == int(args[0])
    s.heights = [(x,y) for x in ohx for y in ohy]
    return s
