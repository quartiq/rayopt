RayOpt
========

Introduction
------------

Optics design (lenses, cavities, gaussian optics, lasers).
Geometric, paraxial, and gaussian ray tracing.


Installation
------------

Install like any usual Python package using `pip`, `easy_install`, or plain
`setup.py`.
The materials and lenses catalogs can be obtained from the freely available
versions of Oslo and Zemax, copied to `catalog/` and then parsed using
`rayopt/library.py` (see there for details on where the files are expected)::

  get and install http://www.lambdares.com/images/OSLO/OSLO662_EDU_Installer.exe
  get and install http://downloads.radiantsourcemodels.com/Downloads/Zemax_2015-03-03_x32.exe
  $ python3 -m rayopt.library \
    Users/Public/Documents/OSLO66\ EDU/bin/lmo \
    Users/Public/Documents/OSLO66\ EDU/bin/glc \
    Users/$USER/My\ Documents/Zemax/Glasscat \
    Users/$USER/My\ Documents/Zemax/Stockcat

Examples
--------

Usage examples are at https://github.com/jordens/rayopt-notebooks as IPython
Notebooks. View at http://nbviewer.ipython.org/github/jordens/rayopt-notebooks/tree/master/
