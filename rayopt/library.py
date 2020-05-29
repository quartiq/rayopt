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

import os
import shutil
import site
import warnings
import logging

from pkg_resources import Requirement, resource_filename

from sqlalchemy.engine import Engine
from sqlalchemy import event, create_engine, orm

from .utils import public
from .library_items import Material, Lens, Catalog, Base
from . import oslo, zemax, rii, codev

logger = logging.getLogger(__name__)

oslo.register_parsers()
zemax.register_parsers()
rii.register_parsers()
codev.register_parsers()


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("pragma page_size = 512")
    cursor.execute("pragma foreign_keys = on")
    cursor.close()


@public
class Library(object):
    _one = None

    @classmethod
    def one(cls, *args, **kwargs):
        if cls._one is None:
            cls._one = cls(*args, **kwargs)
        return cls._one

    def __init__(self, db=None):
        if self._one:
            warnings.warn("to guarantee consistency, Library should be "
                          "used as a singleton throughout: "
                          "use `Library.one()`", stacklevel=3)
        if db is None:
            db = "sqlite:///%s" % self.find_db()
        self.db_get(db)

    def find_db(self):
        name = "library.sqlite"
        dir = os.path.join(site.getuserbase(), "rayopt")
        main = os.path.join(dir, name)
        if not os.path.exists(main):
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
                file_path = os.path.join(path, i)
                try:
                    self.load(file_path, **kwargs)
                except KeyError:
                    pass
                except Exception as e:
                    logger.exception("Could not load %s.", file_path)
                    continue

    def load(self, fil, mode="refresh"):
        if mode in ("refresh", "reload"):
            res = self.session.query(Catalog).filter(
                Catalog.file == fil).first()
            if not res:
                pass
            elif mode == "refresh":
                stat = os.stat(fil)
                if stat.st_mtime <= res.date or stat.st_size == res.size:
                    return
                self.session.delete(res)
            elif mode == "reload":
                self.session.delete(res)

        try:
            if Catalog.parse(fil, self.session):
                self.session.commit()
                print("added %s" % fil)
        except:
            self.session.rollback()
            raise

    def get(self, *args, **kwargs):
        for k in self.get_all(*args, **kwargs):
            return k

    def get_all(self, typ, name=None, catalog=None, source=None, **kwargs):
        Typ = {"material": Material, "lens": Lens}[typ]
        res = self.session.query(Typ).join(Catalog)
        if catalog is not None:
            res = res.filter(Catalog.name == catalog)
        if source is not None:
            res = res.filter(Catalog.source == source)
        if name is not None:
            res = res.filter(Typ.name == name)
        res = res.order_by(Typ.name)
        if not res.count():
            raise KeyError("{} {}/{}/{} not found".format(
                typ, source, catalog, name))
        for item in res:
            yield item.parse()


def _test(l):
    for material in l.session.query(Material):
        material.parse()
    for lens in l.session.query(Lens):
        lens.parse()


def _test_nd(l):
    from .material import lambda_d
    for g in l.session.query(Material):
        if not getattr(g, "nd", None):
            continue
        try:
            m = g.parse()
            nd0 = m.refractive_index(lambda_d)
            if abs(g.nd - nd0) > .001:
                raise ValueError(g.nd, nd0, m.coefficients, m.typ)
        except Exception as e:
            print(g.catalog.source, g.catalog.name, g.name, e)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--db", help="library database url", default=None)
    p.add_argument("-a", "--all", help="test-parse all entries",
                   action="store_true")
    p.add_argument("-m", "--material", help="test-parse a material")
    p.add_argument("-l", "--lens", help="test-parse a lens")
    p.add_argument("-n", "--nd", help="test nd of materials",
                   action="store_true")
    p.add_argument("files", nargs="*")
    o = p.parse_args()
    l = Library(o.db)
    l.load_all(o.files)
    if o.material:
        print(l.get("material", o.material))
    if o.lens:
        print(l.get("lens", o.lens))
    if o.all:
        _test(l)
    if o.nd:
        _test_nd(l)
