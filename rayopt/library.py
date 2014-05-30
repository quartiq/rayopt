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

import os, sys, io, hashlib, time

from sqlalchemy import (Column, Integer, String, Float,
            ForeignKey, desc, asc, Boolean)
from sqlalchemy.engine import Engine
from sqlalchemy import event, create_engine, orm
from sqlalchemy.ext.declarative import declarative_base, declared_attr

from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.orm import relationship, backref

from rayopt import zemax, oslo


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

    def parse(self, **kwargs):
        try:
            return self.parsed
        except AttributeError:
            obj = self.parsers[self.catalog.format](self.data, **kwargs)
            self.parsed = obj
            return obj


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
    }


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
        sha1.update(open(fil, "rb").read())
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
        for lens in zemax.zmf_read(open(self.file, "rb")):
            self.lenses.append(Lens(name=lens.name, version=lens.version,
                elements=lens.elements, shape=lens.shape,
                aspheric=lens.aspheric, toroidal=lens.toroidal,
                grin=lens.grin, efl=lens.efl, enp=lens.enp,
                data=lens.description))

    def load_agf(self, **kwargs):
        catalog = os.path.basename(self.file)
        catalog = os.path.splitext(catalog)[0]
        self.name = catalog
        self.type, self.source, self.format = "glass", "zemax", "agf"
        self.version = 0
        for glass in zemax.agf_read(self.file):
            self.glasses.append(Glass(name=glass.name,
                nd=glass.nd, vd=glass.vd, density=glass.density,
                code=glass.code, status=glass.status, tce=glass.tce,
                comment=glass.comment, data=glass.description))

    def load_dir(self, **kwargs):
        catalog = os.path.basename(self.file)
        catalog = os.path.splitext(catalog)[0]
        self.name = catalog
        self.type, self.source, self.format = "lens", "oslo", "olc"
        self.version = 0
        for lens in oslo.olc_read(self.file):
            self.lenses.append(Lens(name=lens.name,
                elements=lens.elements, thickness=lens.thickness,
                comment=lens.comment, efl=lens.efl, radius=lens.radius,
                section=lens.section, data=lens.description))

    def load_glc(self, **kwargs):
        ver, num, catalog = io.open(self.file, "r").readline().split()[:3]
        self.name = catalog
        self.type, self.source, self.format = "glass", "oslo", "glc"
        self.version = float(ver)
        for glass in oslo.glc_read(self.file):
            self.glasses.append(Glass(name=glass.name, nd=glass.nd,
                vd=glass.vd, density=glass.density,
                data=glass.description))
        #assert len(self.glasses) == int(num), (len(self.glasses, num)


class Library(object):
    def __init__(self, db=None):
        if db is None:
            dir = os.path.split(__file__)[0]
            db = os.path.join(dir, "library.db")
            db = "sqlite:///%s" % db
        self.db_get(db)

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
        res = self.session.query(Catalog).filter(
                Catalog.file == fil).first()
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
        if not cat.load(fil):
            return
        print("adding %s" % fil)
        self.session.add(cat)
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
    print(l.get("glass", "BK7"))
    print(l.get("lens", "AC127-025-A", "thorlabs"))
    for cat in l.session.query(Catalog):
        for glass in cat.glasses:
            glass.parse()
        for lens in cat.lenses:
            lens.parse()


if __name__ == "__main__":
    import glob, sys
    fs = sys.argv[1:] or [
            "catalog/oslo_glass",
            "catalog/zemax_glass",
            "catalog/oslo_lens",
            "catalog/zemax_lens",
            ]
    l = Library.one()
    l.load_all(fs)
    _test(l)
