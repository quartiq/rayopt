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

import shelve, os.path, cPickle as pickle

import numpy as np

from .utils import sfloat
from .elements import Spheroid, Object, Aperture, Image
from .system import System
from .material import AllGlasses


def oslo_lenscat(prefix, name):
    dir_name = os.path.join(prefix, "%s.dir" % name)
    dir = np.loadtxt(dir_name, delimiter=",", skiprows=1, 
                     dtype="i,i,i,S64,f,f,f", ndmin=1)
    # offset, length, name, efl, diameter, thickness
    lens_name = os.path.join(prefix, "%s.dat" % name)
    lens = open(lens_name, "r")
    lens = [lens.read(i) for i in dir["f1"]]
    name_name = os.path.join(prefix, "%s.nam" % name)
    name_lens = {}
    if os.access(name_name, os.R_OK):
        name = np.loadtxt(name_name, delimiter=",", skiprows=1,
                      dtype="S64,S128", ndmin=1)
        # abbrev, description
        for k, n in name:
            name_lens.setdefault(len(k), {})[k] = n.strip("\"")
    lenscat = {}
    for i, (dirline, lensdat) in enumerate(zip(dir, lens)):
        of, le, ele, part, efl, dia, thick = dirline
        desc = None
        for k, kl in name_lens.items():
            cat = part[2:2+k]
            desc = kl.get(cat)
            if desc:
                break
        elems = read_oslo_lens(lensdat)
        for ele in elems:
            ele.radius = dia/2
        lenscat[part] = (efl, elems, dia, thick, desc, lensdat, elems)
    return lenscat


oslo_glass_map = {
    "AIR": "air",
    }


def read_oslo_lens(dat, glass_map=oslo_glass_map):
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
            mat = AllGlasses.get(mat)
            if not mat:
                #print("mat not found", cmd, args)
                mat = AllGlasses["air"]
            s.material = mat
        elif cmd == "TH":
            th = sfloat(args[0])
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
            s = Spheroid(material=AllGlasses["air"], thickness=th)
            sys.append(s)
        else:
            print("unhandled", cmd, args)
    return sys


def default_sys_from_elem(ele):
    obj = Object(infinite=True, radius=.1, material=AllGlasses["air"])
    ap = Aperture(thickness=1., radius=max(e.radius for e in ele))
    img = Image()
    sys = System([obj, ap] + ele + [img])
    return sys


def load_catalogs(all, prefix, catalogs):
    kw = dict(protocol=pickle.HIGHEST_PROTOCOL, writeback=False)
    try:
        db = shelve.open(all, "r", **kw)
        if not db.keys():
            db.close()
            raise
    except:
        # keeping it open writeable corrupts it
        db = shelve.open(all, "c", **kw)
        for f in catalogs:
            try:
                cf = oslo_lenscat(prefix, f)
                db[f] = cf
            except:
                pass
        db.close()
        db = shelve.open(all, "r", **kw)
    return db


catpath = os.path.expanduser("~/work/nist/pyrayopt/lenscat")
cats = []
for n in os.listdir(catpath):
    n, e = os.path.splitext(n)
    if e == ".dir":
        cats.append(n)
all = os.path.join(catpath, "all.shelve")
all_lenses = load_catalogs(all, catpath, cats)
AllLenses = all_lenses
