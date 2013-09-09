class NameMixin(object):
    typ = "?"
    _nickname = None

    @property
    def nickname(self):
        return self._nickname or hex(id(self))

    @nickname.setter
    def nickname(self, name):
        self._nickname = name

    def __repr__(self):
        return "<%s/%s>" % (self.typ, self.nickname)
