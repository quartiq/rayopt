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

import sqlite3
import os

from rayopt import zemax, oslo


class Library(object):
    loaders = {
            ".zmf": zemax.zmf_to_library,
            ".agf": zemax.agf_to_library,
            ".dir": oslo.olc_to_library,
            ".glc": oslo.glc_to_library,
    }
    parsers = {
            "zmx": zemax.zmx_to_system,
            "agf": zemax.agf_to_material,
            "olc": oslo.olc_to_system,
            "len": oslo.len_to_system,
            "glc": oslo.glc_to_material,
    }

    def __init__(self, db=None):
        if db is None:
            dir = os.path.split(__file__)[0]
            db = os.path.join(dir, "library.db")
        self.db_get(db)
        self.db_init()
        self.cache = {}

    def db_get(self, db):
        conn = sqlite3.connect(db)
        conn.text_factory = str
        self.conn = conn
        self.cursor = conn.cursor()

    def db_init(self):
        cu = self.cursor
        cu.execute("pragma page_size = 512")
        cu.execute("pragma foreign_keys = on")
        cu.execute("""create table if not exists catalog (
            id integer primary key autoincrement,
            name text not null collate nocase,
            type text not null,
            source text not null,
            format text not null,
            version real,
            comment text,
            file text,
            date real,
            size integer,
            sha1 text,
            import real
            )""")
        cu.execute("""create table if not exists glass (
            id integer primary key autoincrement,
            name text not null collate nocase,
            catalog integer not null,
            comment text,
            section text,
            status integer,
            version real,
            code integer,
            solid boolean,
            mirror boolean,
            nd real,
            vd real,
            density real,
            tce real,
            data text,
            foreign key (catalog) references catalog(id),
            unique (name, catalog)
            )""")
        cu.execute("""create table if not exists lens (
            id integer primary key autoincrement,
            name text not null collate nocase,
            catalog integer not null,
            comment text,
            section text,
            status integer,
            version real,
            elements integer,
            thickness real,
            radius real,
            shape character,
            aspheric boolean,
            toroidal boolean,
            grin boolean,
            efl real,
            enp real,
            data text,
            foreign key (catalog) references catalog(id),
            unique (catalog, name)
            )""")
        self.conn.commit()

    def load_all(self, paths, **kwargs):
        for path in paths:
            for i in os.listdir(path):
                try:
                    self.load(os.path.join(path, i), **kwargs)
                except KeyError:
                    pass

    def delete_catalog(self, id):
        cu = self.cursor
        cu.execute("delete from lens where catalog = ?", (id,))
        cu.execute("delete from glass where catalog = ?", (id,))
        cu.execute("delete from catalog where id = ?", (id,))
        self.conn.commit()

    def load(self, fil, mode="refresh"):
        dir, base = os.path.split(fil)
        name, ext = os.path.splitext(base)
        extl = ext.lower()
        if extl not in self.loaders:
            return
        cu = self.cursor
        res = cu.execute("select id, date, size "
                "from catalog where file = ?", (fil,)).fetchone()
        if not res:
            pass
        elif mode == "refresh":
            stat = os.stat(fil)
            if stat.st_mtime <= res[1] or stat.st_size == res[2]:
                return
            self.delete_catalog(res[0])
        elif mode == "reload":
            self.delete_catalog(res[0])
        elif mode == "add":
            pass
        print("loading %s" % fil)
        self.loaders[extl](fil, self)

    def get(self, typ, name, catalog=None, source=None, **kwargs):
        cu = self.cursor
        if catalog is None:
            res = cu.execute("select id from {0} where name = ? "
                    "".format(typ), (name,))
        elif source is None:
            res = cu.execute("select {0}.id from {0}, catalog "
                    "where {0}.name = ? "
                    "and {0}.catalog = catalog.id "
                    "and catalog.name = ? "
                    "".format(typ), (name, catalog))
        else:
            res = cu.execute("select {0}.id from {0}, catalog "
                    "where {0}.name = ? "
                    "and {0}.catalog = catalog.id "
                    "and catalog.name = ? "
                    "and catalog.source = ? "
                    "".format(typ), (name, catalog, source))
        res = res.fetchone()
        if not res:
            raise KeyError("{} {}/{}/{} not found".format(
                typ, source, catalog, name))
        return self.parse(typ, res[0], **kwargs)

    def parse(self, typ, id, reload=False):
        if not reload:
            try:
                return self.cache[(typ, id)]
            except KeyError:
                pass
        cu = self.cursor
        res = cu.execute("select data, format from {0}, catalog "
                "where {0}.id = ? and {0}.catalog = catalog.id "
                "".format(typ),
            (id,)).fetchone()
        obj = self.parsers[res[1]](res[0])
        self.cache[(typ, id)] = obj
        return obj

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
    for t in "glass lens".split():
        for i in l.conn.cursor().execute("select id from %s" % t
                ).fetchall():
            #print(t, i[0])
            l.parse(t, i[0])


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
    #_test(l)
