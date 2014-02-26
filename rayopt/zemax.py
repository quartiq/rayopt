# -*- coding: utf-8 -*-
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

from __future__ import (print_function, absolute_import, division,
    unicode_literals)

from struct import Struct
from collections import namedtuple
import os
import codecs
import io
import time
import hashlib

import numpy as np

from .utils import sfloat, sint
from .material import get_material, air, SellmeierMaterial
from .elements import Spheroid
from .system import System


Lens = namedtuple("Lens", "name version elements shape "
        "aspheric grin toroidal length efl enp description")

def zmf_read(f):
    head = Struct("<I")
    lens = Struct("<100sIIIIIIIdd")
    shapes = "?EBPM"
    version, = head.unpack(f.read(head.size))
    assert version in (1001, )
    while True:
        li = f.read(lens.size)
        if len(li) != lens.size:
            if len(li) > 0:
                print(f, "additional data", repr(li))
            break
        li = list(lens.unpack(li))
        li[0] = li[0].decode("latin1").strip("\0")
        li[3] = shapes[li[3]]
        description = f.read(li[7])
        assert len(description) == li[7]
        description = zmf_obfuscate(description, li[8], li[9])
        description = description.decode("latin1")
        assert description.startswith("VERS {:06d}\n".format(li[1]))
        yield Lens(description=description, *li)


def zmf_obfuscate(data, a, b):
    iv = np.cos(6*a + 3*b)
    iv = np.cos(655*(np.pi/180)*iv) + iv
    p = np.arange(len(data))
    k = 13.2*(iv + np.sin(17*(p + 3)))*(p + 1)
    k = (int(("{:.8e}".format(_))[4:7]) for _ in k)
    data = np.fromstring(data, np.uint8)
    data ^= np.fromiter(k, np.uint8, len(data))
    return data.tostring()


def zmf_to_library(fil, library, collision="or replace"):
    stat = os.stat(fil)
    sha1 = hashlib.sha1()
    sha1.update(open(fil, "rb").read())
    sha1 = sha1.hexdigest()
    cu = library.conn.cursor()
    catalog = os.path.basename(fil)
    catalog = os.path.splitext(catalog)[0]
    cu.execute("""insert into catalog
        (name, type, format, version, file, date, size, sha1, import)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            catalog, "lens", "zmx", 1001, fil, stat.st_mtime,
            stat.st_size, sha1, time.time()))
    catalog_id = cu.lastrowid
    cat = list(zmf_read(open(fil, "rb")))
    cu.executemany("""insert %s into lens
        (name, catalog, version, elements, shape,
        aspheric, toroidal, grin, efl, enp, data)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" % collision, ((
            lens.name, catalog_id, lens.version,
            lens.elements, lens.shape,
            lens.aspheric, lens.toroidal, lens.grin,
            lens.efl, lens.enp, lens.description)
            for lens in cat))
    library.conn.commit()


def zmx_to_system(fil):
    s = System()
    next_pos = 0.
    for line in fil.splitlines():
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
            s.append(e)
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
            e.stop = True
        elif cmd == "WAVL":
            s.wavelengths = [float(i)*1e-6 for i in args.split() if i]
        elif cmd == "COAT":
            e.coating = args.split()[0]
        elif cmd == "CONI":
            e.conic = 1 + float(args.split()[0])
        elif cmd == "PARM":
            i, j = args.split()
            i = int(i) - 2
            j = float(j)
            if i < 0:
                if j != 0:
                    print("aspheric 2nd degree not supported", cmd, args)
                continue
            if e.aspherics is None:
                e.aspherics = []
            while len(e.aspherics) <= i:
                e.aspherics.append(0.)
            e.aspherics[i] = j
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
                     "PARM", # aspheric parameters
                     "SQAP", # square aperture?
                     "XDAT", "YDAT", # xy toroidal data
                     "OBNA", # object na
                     "PKUP", # pickup
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
                     "POPS", "COMM", "PZUP",
                     ):
            pass
        else:
            print(cmd, "not handled", args)
            continue
    return s


Glas = namedtuple("Glas", "name nd vd density code comment status "
        "tce description")


def agf_read(fil):
    raw = open(fil, "rb").read(32)
    if raw.startswith(codecs.BOM_UTF16):
        dat = io.open(fil, encoding="utf-16")
    else:
        dat = io.open(fil, encoding="latin1")
    g = []
    density, comment, status, tce = None, None, None, None
    for line in dat:
        cmd, args = line.split(" ", 1)
        if cmd == "CC":
            continue
        if cmd == "NM":
            if g:
                yield Glas(name, nd, vd, density, code, comment, status,
                        tce, "".join(g))
                g = []
                density, comment, status, tce = None, None, None, None
            args = args.split()
            name = args[0]
            nd = sfloat(args[3])
            vd = sfloat(args[4])
            code = args[2]
            if len(args) >= 7:
                status = sint(args[6])
        elif cmd == "GC":
            comment = args
        elif cmd == "ED":
            args = args.split()
            tce = sfloat(args[0])
            density = sfloat(args[2])
        g.append(line)
    if g:
        yield Glas(name, nd, vd, density, code, comment, status, tce,
                "".join(g))


def agf_to_library(fil, library, collision="or replace"):
    stat = os.stat(fil)
    sha1 = hashlib.sha1()
    sha1.update(open(fil, "rb").read())
    sha1 = sha1.hexdigest()
    cu = library.conn.cursor()
    catalog = os.path.basename(fil)
    catalog = os.path.splitext(catalog)[0]
    cu.execute("""insert into catalog
        (name, type, format, version, file, date, size, sha1, import)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            catalog, "glass", "agf", 0, fil, stat.st_mtime,
            stat.st_size, sha1, time.time()))
    catalog_id = cu.lastrowid
    cat = list(agf_read(fil))
    cu.executemany("""insert %s into glass
        (name, catalog, nd, vd, density, code, status, tce, comment, data)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" % collision, ((
            glass.name, catalog_id, glass.nd, glass.vd, glass.density,
            glass.code, glass.status, glass.tce, glass.comment,
            glass.description)
            for glass in cat))
    library.conn.commit()


def agf_to_material(dat):
    for line in dat.splitlines():
        if not line:
            continue
        cmd, args = line.split(" ", 1)
        if cmd == "NM":
            args = args.split()
            g = SellmeierMaterial(name=args[0], nd=sfloat(args[3]),
                    vd=sfloat(args[4]), sellmeier=[])
            g.glasscode = sfloat(args[2])
        elif cmd == "GC":
            g.comment = args.strip()
        elif cmd == "ED":
            args = map(sfloat, args.split())
            g.alpham3070, g.alpha20300, g.density = args[0:3]
        elif cmd == "CD":
            s = np.array(map(sfloat, args.split())).reshape((-1,2))
            g.sellmeier = np.array([si for si in s if not si[0] == 0])
        elif cmd == "TD":
            s = map(sfloat, args.split())
            g.thermal = s
        elif cmd == "OD":
            g.chemical = map(sfloat, args[1:])
            g.price = sfloat(args[0])
        elif cmd == "LD":
            s = map(sfloat, args.split())
        elif cmd == "IT":
            s = map(sfloat, args.split())
            if not hasattr(g, "transmission"):
                g.transmission = {}
            g.transmission[(s[0], s[2])] = s[1]
        else:
            print(cmd, args, "not handled")
    return g


if __name__ == "__main__":
    import glob, sys
    fs = sys.argv[1:]
    if not fs:
        p = "glass/Stockcat/"
        fs = glob.glob(p + "*.zmf") + glob.glob(p + "*.ZMF")
    from rayopt.library import Library
    l = Library()
    for f in fs:
        print(f)
        #c = zmf_to_system(f)
        #print(c)
        zmf_to_library(f, l)
