# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2012 Robert Jordens <robert@joerdens.org>
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

from struct import Struct
import os
import codecs
import io

import numpy as np

from .utils import sfloat, sint
from .material import Material, air, CoefficientsMaterial, Thermal
from .elements import Spheroid
from .system import System
from .library_items import Material as LibMaterial, Lens, Catalog


def register_parsers():
    Catalog.parsers[".zmf"] = zmf_read
    Catalog.parsers[".agf"] = agf_read
    Lens.parsers["zmx"] = zmx_to_system
    LibMaterial.parsers["agf"] = agf_to_material


def zmf_read(file, session):
    cat = Catalog()
    cat.load(file)
    cat.name = os.path.splitext(os.path.basename(file))[0]
    cat.type, cat.source, cat.format = "lens", "zemax", "zmx"
    f = io.open(file, "rb")
    head = Struct("<I")
    lens = Struct("<100sIIIIIIIdd")
    shapes = "?EBPM"
    cat.version, = head.unpack(f.read(head.size))
    assert cat.version in (1001, )
    session.add(cat)
    while True:
        l = Lens()
        li = f.read(lens.size)
        if len(li) != lens.size:
            if len(li) > 0:
                print(f, "additional data", repr(li))
            break
        li = list(lens.unpack(li))
        l.name = li[0].decode("latin1").strip("\0")
        l.shape = shapes[li[3]]
        l.elements = li[2]
        l.aspheric = li[4]
        l.version = li[1]
        l.grin = li[5]
        l.toroidal = li[6]
        l.efl = li[8]
        l.enp = li[9]
        description = f.read(li[7])
        assert len(description) == li[7]
        description = zmf_obfuscate(description, l.efl, l.enp)
        description = description.decode("latin1")
        assert description.startswith("VERS {:06d}\n".format(l.version))
        l.data = description
        cat.lenses.append(l)
    return cat


def zmf_obfuscate(data, a, b):
    iv = np.cos(6*a + 3*b)
    iv = np.cos(655*(np.pi/180)*iv) + iv
    p = np.arange(len(data))
    k = 13.2*(iv + np.sin(17*(p + 3)))*(p + 1)
    k = (int(("{:.8e}".format(_))[4:7]) for _ in k)
    data = np.fromstring(data, np.uint8)
    data ^= np.fromiter(k, np.uint8, len(data))
    return data.tostring()


def zmx_to_system(data, item=None):
    s = System()
    next_pos = 0.
    s.append(Spheroid(material=air))
    for line in data.splitlines():
        e = s[-1]
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
            s.append(Spheroid(distance=next_pos, material=air))
        elif cmd == "CURV":
            e.curvature = float(args.split()[0])
        elif cmd == "DISZ":
            next_pos = float(args)
        elif cmd == "GLAS":
            args = args.split()
            name = args[0]
            try:
                e.material = Material.make(name)
            except KeyError:
                try:
                    e.material = Material.make((float(args[3]),
                                                float(args[4])))
                except Exception as e:
                    print("material not found", name, e)
        elif cmd == "DIAM":
            e.radius = float(args.split()[0])
        elif cmd == "STOP":
            e.stop = True
        elif cmd == "WAVL":
            s.wavelengths = [float(i)*1e-6 for i in args.split() if i]
        elif cmd == "COAT":
            e.coating = args.split()[0]
        elif cmd == "CONI":
            e.conic = float(args.split()[0])
        elif cmd == "PARM":
            i, j = args.split()
            i = int(i) - 1
            j = float(j)
            if i < 0:
                if j:
                    print("aspheric 0 degree not supported", cmd, args)
                continue
            if e.aspherics is None:
                e.aspherics = []
            while len(e.aspherics) <= i:
                e.aspherics.append(0.)
            e.aspherics[i] = j
        elif cmd in ("GCAT",  # glass catalog names
                     "OPDX",  # opd
                     "RAIM",  # ray aiming
                     "CONF",  # configurations
                     "ENPD", "PUPD",  # pupil
                     "EFFL",  # focal lengths
                     "VERS",  # version
                     "MODE",  # mode
                     "NOTE",  # note
                     "TYPE",  # surface type
                     "HIDE",  # surface hide
                     "MIRR",  # surface is mirror
                     "PARM",  # aspheric parameters
                     "SQAP",  # square aperture?
                     "XDAT", "YDAT",  # xy toroidal data
                     "OBNA",  # object na
                     "PKUP",  # pickup
                     "MAZH", "CLAP", "PPAR", "VPAR", "EDGE", "VCON",
                     "UDAD", "USAP", "TOLE", "PFIL", "TCED", "FNUM",
                     "TOL", "MNUM", "MOFF", "FTYP", "SDMA", "GFAC",
                     "PUSH", "PICB", "ROPD", "PWAV", "POLS", "GLRS",
                     "BLNK", "COFN", "NSCD", "GSTD", "DMFS", "ISNA",
                     "VDSZ", "ENVD", "ZVDX", "ZVDY", "ZVCX", "ZVCY",
                     "ZVAN", "XFLN", "YFLN", "VDXN", "VDYN", "VCXN",
                     "VCYN", "VANN", "FWGT", "FWGN", "WWGT", "WWGN",
                     "WAVN", "WAVM", "XFLD", "YFLD", "MNCA", "MNEA",
                     "MNCG", "MNEG", "MXCA", "MXCG", "RGLA", "TRAC",
                     "FLAP", "TCMM", "FLOA", "PMAG", "TOTR", "SLAB",
                     "POPS", "COMM", "PZUP", "LANG", "FIMP",
                     ):
            pass
        else:
            print(cmd, "not handled", args)
            continue
    return s


def agf_read(fil, session):
    cat = Catalog()
    cat.load(fil)
    cat.name = os.path.splitext(os.path.basename(fil))[0]
    cat.type, cat.source, cat.format = "material", "zemax", "agf"
    cat.version = 0
    session.add(cat)
    raw = open(fil, "rb").read(32)
    if raw.startswith(codecs.BOM_UTF16):
        dat = io.open(fil, encoding="utf-16")
    else:
        dat = io.open(fil, encoding="latin1")
    for line in dat:
        if not line.strip():
            continue

        # Skip internal comments
        if line.startswith('!'):
            continue

        cmd, args = line.split(" ", 1)
        if cmd == "CC":
            continue
        if cmd == "NM":
            mat = LibMaterial()
            cat.materials.append(mat)
            args = args.split()
            mat.name = args[0]
            mat.nd = sfloat(args[3])
            mat.vd = sfloat(args[4])
            mat.code = args[2]
            if len(args) >= 7:
                mat.status = sint(args[6])
            mat.data = ""
        elif cmd == "GC":
            mat.comment = args
        elif cmd == "ED":
            args = args.split()
            mat.tce = sfloat(args[0])
            mat.density = sfloat(args[2])
        mat.data += line
    return cat


def agf_to_material(dat, item=None):
    typs = ("schott sellmeier_squared herzberger sellmeier2 conrady "
            "sellmeier_squared handbook_of_optics1 handbook_of_optics2 "
            "sellmeier_squared_offset extended1 sellmeier5 extended2 hikari"
            ).split()
    g = CoefficientsMaterial(coefficients=[])
    for line in dat.splitlines():
        if not line:
            continue
        cmd, args = line[:2], line[3:]
        if cmd == "NM":
            args = args.split()
            typ = typs[int(float(args[1])) - 1]
            g.glasscode = sfloat(args[2])
            g.name = args[0]
            g.typ = typ
        elif cmd == "GC":
            g.comment = args.strip()
        elif cmd == "ED":
            args = list(map(sfloat, args.split()))
            g.alpham3070, g.alpha20300, g.density = args[0:3]
        elif cmd == "CD":
            g.coefficients = np.array([sfloat(_) for _ in args.split()])
        elif cmd == "TD":
            s = [sfloat(_) for _ in args.split()]
            g.thermal = Thermal(s[:3], s[3:5], *s[5:])
        elif cmd == "OD":
            g.chemical = list(map(sfloat, args[1:]))
            g.price = sfloat(args[0])
        elif cmd == "LD":
            g.lambda_min = sfloat(args[0])
            g.lambda_max = sfloat(args[1])
        elif cmd == "IT":
            s = list(map(sfloat, args.split()))
            if not hasattr(g, "transmission"):
                g.transmission = {}
            g.transmission[(s[0], tuple(s[2:]))] = s[1]
        else:
            print(cmd, args, "not handled")
    return g


if __name__ == "__main__":
    import glob
    import sys
    fs = sys.argv[1:]
    if not fs:
        p = "glass/Stockcat/"
        fs = glob.glob(p + "*.zmf") + glob.glob(p + "*.ZMF")
    from rayopt.library import Library
    l = Library()
    for f in fs:
        print(f)
        zmf_read(f, l.session)
        l.session.rollback()
