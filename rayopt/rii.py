#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2015 Robert Jordens <robert@joerdens.org>
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


import yaml
import os
import logging
import subprocess

import numpy as np

from .material import CoefficientsMaterial
from .utils import sfloat
from .library_items import Material, Catalog


logger = logging.getLogger(__name__)


def register_parsers():
    Catalog.parsers["library.yml"] = yml_read
    Material.parsers["rii"] = rii_to_material


def yml_read(fil, session):
    top = Catalog()
    data = top.load(fil)
    top.type, top.source = "material", "rii",
    top.format, top.name = "rii", "refractiveindex.info"
    session.add(top)

    path = os.path.split(fil)[0]
    sha1 = subprocess.check_output([
        "git", "-C", path, "describe", "--abbrev=0", "--always"
    ]).decode().strip()
    for shelf in yaml.safe_load(data):
        cat = Catalog(sha1=sha1, name=shelf["SHELF"],
                      source=top.source, type=top.type, format=top.format,
                      version=top.version, comment=str(top.id), file=top.file,
                      date=top.date, imported=top.imported)
        session.add(cat)
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
                    data = yaml.safe_load(open(fil))
                    data["BOOK"] = book["BOOK"]
                    data["PAGE"] = page["PAGE"]
                    data["name"] = page["name"]
                    data["div"] = div
                    data["path"] = page["path"]
                    g = Material(
                        name="{}|{}".format(book["BOOK"], page["PAGE"]),
                        section="{}|{}".format(div, book["name"]),
                        comment=page["path"], data=yaml.dump(data))
                    cat.materials.append(g)
                except Exception as e:
                    print(f"error: {page}: {e}")
    return top


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


def rii_to_material(dat, item=None):
    data = yaml.safe_load(dat)
    g = CoefficientsMaterial(name="{}|{}".format(data["BOOK"], data["PAGE"]),
                             coefficients=[])
    g.comment = data.get("COMMENTS", None)
    g.references = data.get("REFERENCES", None)
    for d in data["DATA"]:
        typ = d["type"]
        if typ.startswith("formula"):
            g.typ = _typ_map[typ]
            g.lambda_min, g.lambda_max = (
                sfloat(_) for _ in d["range"].split())
            c = np.array([sfloat(_) for _ in d["coefficients"].split()])
            g.coefficients = c
        if typ == "tabulated k":
            g.tabulated_k = np.array([sfloat(_) for _ in d["data"].split()])
    return g
