# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2015 Robert Jordens <jordens@phys.ethz.ch>
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

import os

import numpy as np
import yaml


Glass = namedtuple("Glass", "name typ range coefficients comments")


def yml_read(fil):
    lib = yaml.load(open(fil, "r").read())
    base, fil = os.path.split(fil)
    for shelf in lib:
        for item in shelf["content"]:
            if not "BOOK" in item:
                continue
            for page in item["content"]:
                dat = open(os.path.join(base, page["path"]), "r")
                dat = yaml.load(dat.read())
                yield Glass(name, nd, vd, density, code, comment, status,
                        tce, "".join(g))


def yml_to_material(dat):
    for line in dat.splitlines():
        if not line:
            continue
        cmd, args = line[:2], line[3:]
        if cmd == "NM":
            args = args.split()
            typ = typs[int(float(args[1])) - 1]
            g = CoefficientsMaterial(name=args[0], typ=typ, coefficients=[])
            g.glasscode = sfloat(args[2])
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



