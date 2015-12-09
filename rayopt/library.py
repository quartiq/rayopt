# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
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

import os
import io
import hashlib
import time
import shutil
import site

from pkg_resources import Requirement, resource_filename

from sqlalchemy import (Column, Integer, String, Float,
                        ForeignKey, Boolean)
from sqlalchemy.engine import Engine
from sqlalchemy import event, create_engine, orm
from sqlalchemy.ext.declarative import declarative_base, declared_attr

from sqlalchemy.orm import relationship, Session

from . import zemax, oslo, rii
from .utils import public


class Tablename(object):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
Base = declarative_base(cls=Tablename)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("pragma page_size = 512")
    cursor.execute("pragma foreign_keys = on")
    cursor.close()


class LoaderParser:
    parsers = {}

    def parse(self):
        obj = self.parsers[self.catalog.format](self.data)
        obj.catalog = self.catalog.name
        obj.source = self.catalog.source
        return obj


@public
class Glass(Base, LoaderParser):
    id = Column(Integer, primary_key=True)
    name = Column(String(collation="nocase"), nullable=False)
    catalog_id = Column(Integer, ForeignKey("catalog.id"))
    comment = Column(String)
    section = Column(String)
    status = Column(Integer)
    version = Column(Float)
    code = Column(Integer)
    solid = Column(Boolean)
    mirror = Column(Boolean)
    nd = Column(Float)
    vd = Column(Float)
    density = Column(Float)
    tce = Column(Float)
    data = Column(String)
    #  unique (name, catalog)

    parsers = {
        "agf": zemax.agf_to_material,
        "glc": oslo.glc_to_material,
        "rii": rii.rii_to_material,
    }


@public
class Lens(Base, LoaderParser):
    id = Column(Integer, primary_key=True)
    name = Column(String(collation="nocase"), nullable=False)
    catalog_id = Column(Integer, ForeignKey("catalog.id"))
    comment = Column(String)
    section = Column(String)
    status = Column(Integer)
    version = Column(Float)
    elements = Column(Integer)
    thickness = Column(Float)
    radius = Column(Float)
    shape = Column(String(1))
    aspheric = Column(Boolean)
    toroidal = Column(Integer)
    grin = Column(Boolean)
    efl = Column(Float)
    enp = Column(Float)
    data = Column(String)
    #  unique (catalog, name)

    parsers = {
        "zmx": zemax.zmx_to_system,
        "olc": oslo.olc_to_system,
        "len": oslo.len_to_system,
    }


@public
class Catalog(Base):
    id = Column(Integer, primary_key=True)
    name = Column(String(collation="nocase"), nullable=False)
    type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    format = Column(String, nullable=False)
    version = Column(Float)
    comment = Column(String)
    file = Column(String)
    date = Column(Float)
    size = Column(Integer)
    sha1 = Column(String)
    imported = Column(Float)

    lenses = relationship(Lens, lazy="dynamic", backref="catalog",
                          cascade="all, delete-orphan")
    glasses = relationship(Glass, lazy="dynamic", backref="catalog",
                           cascade="all, delete-orphan")

    def load(self, fil, **kwargs):
        dir, base = os.path.split(fil)
        name, ext = os.path.splitext(base)
        loader = getattr(self, "load_%s" % ext[1:].lower(), None)
        if loader is None:
            return
        self.file = fil
        stat = os.stat(fil)
        self.date = stat.st_mtime
        self.size = stat.st_size
        sha1 = hashlib.sha1()
        sha1.update(io.open(fil, "rb").read())
        self.sha1 = sha1.hexdigest()
        self.imported = time.time()
        loader(**kwargs)
        return self

    def load_zmf(self, **kwargs):
        catalog = os.path.basename(self.file)
        catalog = os.path.splitext(catalog)[0]
        self.name = catalog
        self.type, self.source, self.format = "lens", "zemax", "zmx"
        self.version = 1001
        for lens in zemax.zmf_read(io.open(self.file, "rb")):
            l = Lens(name=lens.name, version=lens.version,
                     elements=lens.elements, shape=lens.shape,
                     aspheric=lens.aspheric, toroidal=lens.toroidal,
                     grin=lens.grin, efl=lens.efl, enp=lens.enp,
                     data=lens.description)
            self.lenses.append(l)

    def load_agf(self, **kwargs):
        catalog = os.path.basename(self.file)
        catalog = os.path.splitext(catalog)[0]
        self.name = catalog
        self.type, self.source, self.format = "glass", "zemax", "agf"
        self.version = 0
        for glass in zemax.agf_read(self.file):
            g = Glass(name=glass.name,
                      nd=glass.nd, vd=glass.vd, density=glass.density,
                      code=glass.code, status=glass.status, tce=glass.tce,
                      comment=glass.comment, data=glass.description)
            self.glasses.append(g)

    def load_dir(self, **kwargs):
        catalog = os.path.basename(self.file)
        catalog = os.path.splitext(catalog)[0]
        self.name = catalog
        self.type, self.source, self.format = "lens", "oslo", "olc"
        self.version = 0
        for lens in oslo.olc_read(self.file):
            l = Lens(name=lens.name,
                     elements=lens.elements, thickness=lens.thickness,
                     comment=lens.comment, efl=lens.efl, radius=lens.radius,
                     section=lens.section, data=lens.description)
            self.lenses.append(l)

    def load_glc(self, **kwargs):
        ver, num, catalog = io.open(self.file, "r").readline().split()[:3]
        self.name = catalog
        self.type, self.source, self.format = "glass", "oslo", "glc"
        self.version = float(ver)
        for glass in oslo.glc_read(self.file):
            g = Glass(name=glass.name, nd=glass.nd,
                      vd=glass.vd, density=glass.density,
                      data=glass.description)
            self.glasses.append(g)
        #assert len(self.glasses) == int(num), (len(self.glasses, num)

    def load_yml(self, **kwargs):
        session = Session.object_session(self)
        self.type, self.source = "glass", "rii",
        self.format, self.name = "rii", "refractiveindex.info"
        for glass in rii.yml_read(self.file):
            if isinstance(glass, rii.Catalog):
                cat = Catalog(source=self.source, type=self.type,
                              format=self.format, name=glass.name,
                              sha1=glass.sha1, version=0,
                              comment=str(self.id), file=self.file,
                              date=self.date, imported=self.imported)
                session.add(cat)
            else:
                g = Glass(name=glass.name, data=glass.data,
                          section=glass.section, comment=glass.comment)
                cat.glasses.append(g)


@public
class Library(object):
    def __init__(self, db=None):
        if db is None:
            db = "sqlite:///%s" % self.find_db()
        self.db_get(db)

    def find_db(self):
        name = "library.sqlite"
        dir = os.path.join(site.getuserbase(), "rayopt")
        main = os.path.join(dir, name)
        if os.path.exists(main):
                return main
        base = resource_filename(Requirement.parse("rayopt"), name)
        if not os.path.exists(base):
            base = os.path.join(os.path.split(__file__)[0], name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(base, main)
        return main

    def db_get(self, db):
        self.engine = create_engine(db)
        Base.metadata.create_all(self.engine)
        Session = orm.sessionmaker(bind=self.engine)
        self.session = Session()

    def load_all(self, paths, **kwargs):
        for path in paths:
            for i in os.listdir(path):
                try:
                    self.load(os.path.join(path, i), **kwargs)
                except KeyError:
                    pass

    def load(self, fil, mode="refresh"):
        res = self.session.query(Catalog).filter(Catalog.file == fil).first()
        if not res:
            pass
        elif mode == "refresh":
            stat = os.stat(fil)
            if stat.st_mtime <= res.date or stat.st_size == res.size:
                return
            self.sesstion.delete(res)
        elif mode == "reload":
            self.sesstion.delete(res)
        elif mode == "add":
            pass
        cat = Catalog()
        self.session.add(cat)
        if not cat.load(fil):
            self.session.expunge(cat)
        else:
            print("added %s" % fil)
        self.session.commit()

    def get(self, typ, name, catalog=None, source=None, **kwargs):
        Typ = {"glass": Glass, "lens": Lens}[typ]
        res = self.session.query(Typ).join(Catalog)
        if catalog is not None:
            res = res.filter(Catalog.name == catalog)
        if source is not None:
            res = res.filter(Catalog.source == source)
        res = res.filter(Typ.name == name).first()
        if not res:
            raise KeyError("{} {}/{}/{} not found".format(
                typ, source, catalog, name))
        return res.parse(**kwargs)

    @classmethod
    def one(cls):
        try:
            return cls._one
        except AttributeError:
            cls._one = cls()
            return cls._one


def _test(l):
    for glass in l.session.query(Glass):
        glass.parse()
    for lens in l.session.query(Lens):
        lens.parse()


def _test_nd(l):
    from .material import lambda_d
    for g in l.session.query(Glass):
        if hasattr(g, "nd"):
            try:
                nd0 = g.parse().refractive_index(lambda_d)
            except:
                continue
            if 1. in (nd0, g.nd) or 0. in (nd0, g.nd):
                continue
            if abs(g.nd - nd0) > .001:
                print(g.name, g.catalog.source, g.catalog.name,
                      g.nd, nd0, g.parse().coefficients, g.parse().typ)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--db", help="library database url", default=None)
    p.add_argument("-a", "--all", help="test-parse all entries",
                   action="store_true")
    p.add_argument("-g", "--glass", help="test-parse a glass")
    p.add_argument("-l", "--lens", help="test-parse a lens")
    p.add_argument("files", nargs="*")
    o = p.parse_args()
    l = Library(o.db)
    # fs = ["catalog/oslo_glass", "catalog/zemax_glass",
    # "catalog/oslo_lens", "catalog/zemax_lens"]
    l.load_all(o.files)
    #print(l.get("glass", "BK7"))
    #print(l.get("lens", "AC127-025-A", "thorlabs"))
    if o.glass:
        g = l.get("glass", o.glass)
        print(g)
    if o.lens:
        print(l.get("lens", o.lens))
    if o.all:
        _test(l)
    #_test_nd(l)
