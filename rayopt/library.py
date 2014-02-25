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

from . import zemax, oslo


class Library(object):
    loaders = {
            ".zmf": zemax.zmf_to_library,
            ".agf": zemax.agf_to_library,
            ".dir": oslo.olc_to_library,
            ".glc": oslo.glc_to_library,
    }

    def __init__(self, db="library.db"):
        self.db_get(db)
        self.db_init()

    def db_get(self, db="library.db"):
        conn = sqlite3.connect(db)
        conn.text_factory = str
        self.conn = conn

    def db_init(self):
        cu = self.conn.cursor()
        cu.execute("pragma page_size=512")
        cu.execute("pragma foreign_keys=on")
        cu.execute("""create table if not exists catalog (
            id integer primary key autoincrement,
            name text not null,
            comment text,
            type text,
            format text,
            version integer,
            file text,
            date real,
            import real
            )""")
        cu.execute("""create table if not exists glas (
            id integer primary key autoincrement,
            name text not null,
            catalog integer not null,
            section text,
            comment text,
            code integer,
            solid boolean,
            mirror boolean,
            nd real,
            vd real,
            density real,
            price real,
            data text,
            foreign key (catalog) references catalog(id),
            unique (name, catalog)
            )""")
        cu.execute("""create table if not exists lens (
            id integer primary key autoincrement,
            name text not null,
            catalog integer not null,
            section text,
            comment text,
            version integer,
            elements integer,
            thickness real,
            radius real,
            code character,
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
        cu = self.conn.cursor()
        cu.execute("drop from lens where catalog = ?", id)
        cu.execute("drop from glas where catalog = ?", id)
        cu.execute("drop from catalog where id = ?", id)
        self.conn.commit()

    def load(self, fil, mode="refresh"):
        dir, base = os.path.split(fil)
        name, ext = os.path.splitext(base)
        extl = ext.lower()
        if extl not in self.loaders:
            return
        cu = self.conn.cursor()
        res = cu.execute("select id, date from catalog where file = ?",
            (fil,)).fetchone()
        tim = os.stat(fil).st_mtime
        if res:
            if mode == "refresh" and tim <= res[1]:
                return
            elif mode == "add":
                pass
            elif mode == "reload":
                self.delete_catalog(res[0])
        print("loading %s" % fil)
        self.loaders[extl](fil, self)


if __name__ == "__main__":
    import glob, sys
    fs = sys.argv[1:] or ["glass/Stockcat", "lenscat", "glass/Glasscat", "glass/oslo"]
    Library().load_all(fs)
