# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2013 Robert Jordens <jordens@phys.ethz.ch>
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
from collections import namedtuple
import os.path
import time
import io
import hashlib

import numpy as np

from .utils import sfloat, sint
from .elements import Spheroid
from .system import System
from .material import air, Material, SellmeierMaterial


Lens = namedtuple("Lens", "name elements efl radius thickness comment "
        "section description")


def olc_read(dir):
    prefix, ext = os.path.splitext(dir)
    assert ext.lower() == ".dir"
    # offset, length, name, efl, diameter, thickness
    dir = np.loadtxt(dir, delimiter=",", skiprows=1,
            dtype="i,i,i,S64,f,f,f", ndmin=1)
    lens = open("%s.dat" % prefix, "r")
    lens = [lens.read(i) for i in dir["f1"]]
    sections = {}
    sect_lens = []
    if os.access("%s.nam" % prefix, os.R_OK):
        try:
            # abbrev, description
            name = np.loadtxt("%s.nam" % prefix, delimiter=",", skiprows=1,
                      dtype="S64,S128", ndmin=1)
            for k, n in name:
                sect_lens.append(len(k))
                sections[k] = n.strip("\" '")
        except IndexError:
            pass
    sect_lens = sorted(sect_lens)[::-1]
    for i, (dirline, lensdat) in enumerate(zip(dir, lens)):
        of, le, ele, part, efl, dia, thick = dirline
        section = None
        comment = None
        for k in sect_lens:
            try:
                comment = sections[part[:k]]
                section = part[:k]
                break
            except KeyError:
                continue
        yield Lens(part, int(ele), float(efl), float(dia/2.), float(thick),
                comment, section, lensdat)


def olc_to_library(fil, library, collision="or replace"):
    stat = os.stat(fil)
    sha1 = hashlib.sha1()
    sha1.update(open(fil, "rb").read())
    sha1 = sha1.hexdigest()
    cu = library.conn.cursor()
    catalog = os.path.basename(fil)
    catalog = os.path.splitext(catalog)[0]
    cu.execute("""insert into catalog
        (name, type, source, format, version, file, date, size,
        sha1, import)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            catalog, "lens", "oslo", "olc", 0, fil,
            stat.st_mtime, stat.st_size, sha1, time.time()))
    catalog_id = cu.lastrowid
    cat = list(olc_read(fil))
    cu.executemany("""insert %s into lens
        (name, catalog, elements, thickness, comment,
        efl, radius, section, data)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)""" % collision, ((
            lens.name, catalog_id, lens.elements, lens.thickness,
            lens.comment, lens.efl, lens.radius, lens.section,
            lens.description)
            for lens in cat))
    library.conn.commit()


oslo_glass_map = {
    }


def olc_to_system(dat, glass_map=oslo_glass_map):
    sys = []
    s = Spheroid()
    sys.append(s)
    for cmd in dat.split(";"):
        cmd = cmd.strip()
        if not cmd:
            continue
        args = cmd.split()
        cmd, args = args[0], args[1:]
        if cmd == "RD":
            r = sfloat(args[0])
            s.curvature = 1/r if r else 0
        elif cmd == "GLA":
            mat = args[0].upper()
            mat = glass_map.get(mat, mat)
            try:
                mat = Material.make(mat)
            except KeyError:
                print("mat not found", cmd, args)
                mat = air
            s.material = mat
        elif cmd == "TH":
            th = sfloat(args[0]) or 0.
            # used on nxt and last
        elif cmd in "AP CVX APN AY1 AY2 AX1 AX2 ATP AAC".split():
            pass # cylindrical
        elif cmd == "CC":
            s.conic = sfloat(args[0])
        elif cmd == "ASP":
            assert args[0] in ("ASR", "ARA"), args
            s.aspherics = [0] * (int(args[1]) + 1)
        elif cmd[:2] == "AS":
            i = int(cmd[2])
            s.aspherics[i] = sfloat(args[0])
        elif cmd == "NXT":
            s = Spheroid(material=air, distance=th)
            sys.append(s)
        else:
            print("unhandled", cmd, args)
    return sys


def len_to_system(fil):
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
            e.material = Material.make(args[0])
        elif cmd == "AST":
            e.stop = True
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
    return s


Glas = namedtuple("Glas", "name nd vd density description")


def glc_read(f):
    f = io.open(f, "r")
    line = f.readline().split()
    ver, num, catalog = line[:3]
    #if len(line) > 3:
    #    print line
    for l in f:
        line = l.strip().split()
        if not line:
            continue
        name = line.pop(0)
        nd = sfloat(line.pop(0))
        vd = sfloat(line.pop(0))
        density = sfloat(line.pop(0))
        yield Glas(name, nd, vd, density, l.strip())


def glc_to_material(l):
    line = l.strip().split()
    name = line.pop(0)
    nd = sfloat(line.pop(0))
    vd = sfloat(line.pop(0))
    density = sfloat(line.pop(0))
    del line[:6]
    del line[:2]
    a, num = sint(line.pop(0)), sint(line.pop(0))
    sellmeier = np.array(map(sfloat, line[:num]))
    if a in (1, 2):
        sellmeier = sellmeier.reshape(2, -1).T
    del line[:num]
    mat = SellmeierMaterial(name=name, sellmeier=sellmeier, nd=nd, vd=vd)
    #if not np.allclose(nd, mat.nd):
    #    print(name, nd, mat.nd)
    mat.density = density
    return mat # weird remaining format
    if not line:
        return mat
    a, num = sint(line.pop(0)), sint(line.pop(0))
    if a != 1:
        del line[:num]
    a, num = sint(line.pop(0)), sint(line.pop(0))
    assert a == 1, l
    num *= 2
    transmission = np.array(map(sfloat, line[:num])).reshape(-1, 2)
    del line[:num]
    return mat


def glc_to_library(fil, library, collision="or replace"):
    stat = os.stat(fil)
    sha1 = hashlib.sha1()
    sha1.update(open(fil, "rb").read())
    sha1 = sha1.hexdigest()
    cu = library.conn.cursor()
    ver, num, catalog = io.open(fil, "r").readline().split()[:3]
    cu.execute("""insert into catalog
        (name, type, source, format, version, file, date, size,
        sha1, import)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            catalog, "glass", "oslo", "glc", float(ver), fil, 
            stat.st_mtime, stat.st_size, sha1, time.time()))
    catalog_id = cu.lastrowid
    cat = list(glc_read(fil))
    cu.executemany("""insert %s into glass
        (name, catalog, nd, vd, density, data)
        values (?, ?, ?, ?, ?, ?)""" % collision, ((
            glass.name, catalog_id, glass.nd, glass.vd, glass.density,
            glass.description) for glass in cat))
    n = cu.execute("select count() from glass where catalog = ?",
            (catalog_id,)).fetchone()[0]
    #assert n == int(num), (n, int(num))
    library.conn.commit()
