# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2014 Robert Jordens <jordens@phys.ethz.ch>
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

__all__ = ["NameMixin"]


class NameMixin(object):
    _types = {}
    _default_type = None
    _nickname = None
    _type = None
    _typeletter = None

    @classmethod
    def register(cls, sub):
        if sub._type is None:
            sub._type = sub.__name__.lower()
        k = cls, sub._type
        assert k not in cls._types, (k, sub, cls._types)
        cls._types[k] = sub
        return sub

    def dict(self):
        dat = {}
        if self._type != self._default_type:
            dat["type"] = self._type
        if self._nickname:
            dat["nickname"] = self.nickname
        return dat

    @classmethod
    def make(cls, data):
        if isinstance(data, cls):
            return data
        typ = data.pop("type", cls._default_type)
        sub = cls._types[(cls, typ)]
        return sub(**data)

    @property
    def type(self):
        return self._type

    @property
    def typeletter(self):
        return self._typeletter or self._type[0].upper()

    @property
    def nickname(self):
        return self._nickname or hex(id(self))

    @nickname.setter
    def nickname(self, name):
        self._nickname = name

    def __str__(self):
        return "<%s/%s>" % (self.typeletter, self.nickname)
