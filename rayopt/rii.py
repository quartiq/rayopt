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

import yaml
import os
import time
import argparse
import logging
import subprocess
from collections import namedtuple

import numpy as np

from .material import Material, air, CoefficientsMaterial, Thermal
from .utils import sfloat, sint

logger = logging.getLogger(__name__)

Glass = namedtuple("Glass", "name section data comment")
Catalog = namedtuple("Catalog", "name sha1")


def yml_read(fil):
    path = os.path.split(fil)[0]
    sha1 = subprocess.check_output([
        "git", "-C", path, "describe", "--abbrev=0", "--always"
    ]).decode().strip()
    for shelf in yaml.safe_load(open(fil, "r")):
        cat = Catalog(sha1=sha1, name=shelf["SHELF"])
        yield cat
        div = None
        for book in shelf["content"]:
            if "DIVIDER" in book:
                div = book["DIVIDER"]
                continue
            for page in book["content"]:
                if "DIVIDER" in page:
                    continue
                fil = os.path.join(path, page["path"])
                try:
                    data = yaml.safe_load(open(fil, "r"))
                    data["BOOK"] = book["BOOK"]
                    data["PAGE"] = page["PAGE"]
                    data["name"] = page["name"]
                    data["div"] = div
                    data["path"] = page["path"]
                    yield Glass(name="{}/{}".format(book["BOOK"], page["PAGE"]),
                                section="{}/{}".format(div, book["name"]),
                                comment=page["path"], data=yaml.dump(data))
                except Exception as e:
                    print("error: {}: {}".format(page, e))

_typ_map = {
    "formula 1": "sellmeier_offset",
    "formula 2": "sellmeier_squared_offset",
    "formula 3": "polynomial",
    "formula 4": "refractiveindex_info",
    "formula 5": "cauchy",
    "formula 6": "gas_offset",
    "formula 7": "herzberger",
    "formula 8": "retro",
    "formula 9": "exotic",
}

def rii_to_material(dat):
    data = yaml.safe_load(dat)
    g = CoefficientsMaterial(name="{}/{}".format(data["BOOK"], data["PAGE"]),
                               coefficients=[])
    g.comment = data.get("COMMENTS", None)
    g.references = data.get("REFERENCES", None)
    for d in data["DATA"]:
        typ = d["type"]
        if typ.startswith("formula"):
            g.typ = _typ_map[typ]
            g.lambda_min, g.lambda_max = (sfloat(_) for _ in d["range"].split())
            c = np.array([sfloat(_) for _ in d["coefficients"].split()])
            g.coefficients = c
        if typ == "tabulated k":
            g.tabulated_k = np.array([sfloat(_) for _ in d["data"].split()])
    return g
