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


import os
import io
import hashlib
import time

from sqlalchemy import (Column, Integer, String, Float,
                        ForeignKey, Boolean)
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship

from .utils import public


class Tablename:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
Base = declarative_base(cls=Tablename)


class LoaderParser:
    parsers = {}

    def parse(self):
        try:
            return self.obj
        except AttributeError:
            pass
        obj = self.parsers[self.catalog.format](self.data, self)
        obj.item = self
        obj.catalog = self.catalog.name
        self.obj = obj
        return obj


@public
class Material(Base, LoaderParser):
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
    materials = relationship(Material, lazy="dynamic", backref="catalog",
                             cascade="all, delete-orphan")

    parsers = {
    }

    @classmethod
    def parse(self, fil, session):
        fill = fil.lower()
        for k, v in self.parsers.items():
            if fill.endswith(k):
                return v(fil, session)

    def load(self, fil):
        data = open(fil, "rb").read()
        self.file = fil
        stat = os.stat(fil)
        self.date = stat.st_mtime
        self.size = stat.st_size
        sha1 = hashlib.sha1()
        sha1.update(data)
        self.sha1 = sha1.hexdigest()
        self.imported = time.time()
        return data
