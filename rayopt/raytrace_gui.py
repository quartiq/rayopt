#!/usr/bin/python
# -*- coding: utf8 -*-
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

from rayopt import (Rays, Spheroid, System,
        demerit_aberration3,
        demerit_rms_position, Parameter, MaterialThickness)

from chaco.api import Plot, ArrayPlotData
from chaco.tools.api import PanTool, ZoomTool

from traits.api import (Float, Enum, Instance, TraitRange,
        Trait, Instance, HasTraits)
from traits.ui.api import Item, View, HGroup

from enable.api import (Component, ColorTrait)
from enable.component_editor import ComponentEditor

from kiva import Font
from kiva import SWISS

from numpy import (linspace, array, alltrue, extract, isfinite,
        argwhere, zeros_like, inf)

font = Font(family=SWISS)

class SystemComp(Component):
    elementcolor = ColorTrait("gray")
    raycolor = ColorTrait("lightblue")
    bgcolor = "transparent"
    system = Instance(System)
    rays = Instance(Rays)
    projection = Enum((2,1,0), (2,0,1), (0,1,2))
    scale = Trait(10, TraitRange(.01, 100))

    #def _on_trait_change(self, *a, **k):
    #    print "do"
    #    self.request_redraw()

    def _draw_mainlayer(self, gc, view=None, mode="default"):
        gc.save_state()
        x,y = self.position
        dx, dy = self.bounds
        i, j, k = self.projection
        gc.translate_ctm(x+15, y+dy/2)
        gc.rotate_ctm(0.)
        gc.scale_ctm(self.scale, self.scale)
        gc.set_line_width(1.)
        gc.set_font(font)
        for element, inrays in self.system.propagate(self.rays):
            #print str(element), inrays.positions, inrays.angles, inrays.lengths
            p_start = inrays.positions[:,(i,j)]
            p_end = inrays.end_positions[:,(i,j)]
            p_good = alltrue(isfinite(p_start) & isfinite(p_end),
                    axis=1)
            gc.translate_ctm(element.origin[i], element.origin[j])
            gc.rotate_ctm(element.angles[k])
            gc.set_stroke_color(self.raycolor_)
            gc.begin_path()
            gc.line_set(p_start[p_good], p_end[p_good])
            gc.stroke_path()
            #s = str(element)
            #tx,ty,tdx,tdy = gc.get_text_extent(s)
            #gc.set_text_position(0-tdy, -.1-tdy)
            #gc.show_text(s)
            try:
                t = linspace(-element.radius, element.radius, 20)
                z = -element.shape_func(array((0*t, t, 0*t)).T)
                gc.set_stroke_color(self.elementcolor_)
                gc.begin_path()
                gc.lines(zip(z, t))
                gc.stroke_path()
            except AttributeError:
                try:
                    gc.move_to(0, -element.radius)
                    gc.line_to(0, element.radius)
                    gc.stroke_path()
                except AttributeError:
                    pass
        gc.restore_state()


class Raytrace(HasTraits):
    system = Instance(SystemComp)
    spot_diagram = Instance(Plot)
    x_transverse = Instance(Plot)
    y_transverse = Instance(Plot)
    traits_view = View(
            Item("system", editor=ComponentEditor(size=(1024,200)),
                show_label=False),
            HGroup(
                Item("object.system.projection", label="Proj"),
                Item("object.system.scale", label="Scale"),
                Item("object.system.x", label="X"),
                Item("object.system.y", label="Y"),
                ),
            HGroup(
                Item("spot_diagram",
                    editor=ComponentEditor(size=(200,200)),
                    show_label=False),
                Item("x_transverse",
                    editor=ComponentEditor(size=(200,200)),
                    show_label=False),
                Item("y_transverse",
                    editor=ComponentEditor(size=(200,200)),
                    show_label=False),
                ),
        resizable=True)

    def __init__(self, system, **k):
        super(Raytrace, self).__init__(**k)

        n = 30

        rays = system.get_ray_bundle(
                system.wavelengths[0],
                system.heights[-1], n, 
                paraxial_chief=True,
                paraxial_marginal=False)
        print rays.positions, rays.angles

        optrays = tuple(system.get_ray_bundle(w, xy, 7,
                paraxial_chief=True,
                paraxial_marginal=False)
                    for w in system.wavelengths
                    for xy in system.heights)
        optparams = (
            #"elements[1].curvature",
            Parameter("elements[1].curvature", (-1/10., 1/10.), 1e-3),
            #Parameter("elements[2].origin[2]", (1, 6), 1e-3),
            Parameter("elements[2].curvature", (-1/10., 1/10.), 1e-3),
            #Parameter("elements[3].origin[2]", (1, 6), 1e-3),
            Parameter("elements[3].curvature", (-1/10., 1/10.), 1e-3),
            Parameter("elements[5].curvature", (-1/10., 1/10.), 1e-3),
            Parameter("elements[6].curvature", (-1/10., 1/10.), 1e-3),
            Parameter("elements[7].curvature", (-1/10., 1/10.), 1e-3),
            Parameter("elements[8].curvature", (-1/10., 1/10.), 1e-3),
            #Parameter("elements[9].origin[2]", (1, 6), 1e-3),
            Parameter("elements[9].curvature", (-1/10., 1/10.), 1e-3),
            #Parameter("elements[10].origin[2]", (1, 6), 1e-3),
            Parameter("elements[10].curvature", (-1/10., 1/10.), 1e-3),
            #Parameter("elements[11].origin[2]", (5e-5, 6), 1e-3),
            Parameter("elements[11].curvature", (-1/10., 1/10.), 1e-3),
            #Parameter("elements[12].origin[2]", (1, 6), 1e-3),
            Parameter("elements[12].curvature", (-1/10., 1/10.), 1e-3),
            #Parameter("elements[13].origin[2]", (1, 6), 1e-3),
            Parameter("elements[13].curvature", (-1/10., 1/10.), 1e-3),
             )
        optparams = (
            Parameter("elements[2].curvature", (-1/20., 1/20.), 1e-4),
            Parameter("elements[3].curvature", (-1/10., -1/20.), 1e-4),
            Parameter("elements[3].conic", (.3, 1.2), 1e-3),
            Parameter("elements[3].aspherics[0]", (-1e-5, 1e-5), 1e-8),
            Parameter("elements[3].aspherics[1]", (-1e-7, 1e-7), 1e-10),
            Parameter("elements[3].aspherics[2]", (-1e-9, 1e-9), 1e-12),
            Parameter("elements[3].aspherics[3]", (-1e-11, 1e-11), 1e-14),
            #Parameter("elements[11].origin[2]", (160, 220), 1e-3),
            )
        optparams = (
            Parameter("elements[1].curvature", (1/20., 1/10.), 1e-4),
            Parameter("elements[1].conic", (.2, 1.2), 1e-3),
            Parameter("elements[1].aspherics[0]", (-1e-4, 1e-4), 1e-8),
            Parameter("elements[1].aspherics[1]", (-1e-6, 1e-6), 1e-10),
            Parameter("elements[1].aspherics[2]", (-1e-8, 1e-8), 1e-12),
            Parameter("elements[1].aspherics[3]", (-1e-10, 1e-10), 1e-14),
            Parameter("elements[2].curvature", (-1/20., 1/20.), 1e-4),
            )

        print str(system)
        #print system.optimize(optrays, optparams,
        #        (demerit_rms_position,), # demerit_aberration3), 
        #        #(MaterialThickness(),),
        #        method="ralg") #scipy_lbfgsb")
        print str(system)

        self.system = SystemComp(system=system, rays=rays,
                position=(0,0), bounds=(1024,400))
        self.system.projection = (2,1,0)
        self.system.scale = 5

        orays = system.propagate_through(rays)
        x,y = orays.positions[...,(0,1)].T * 1e3
        #x,y = orays.angles[...,(0,1)].T
        x -= x[0]
        y -= y[0]
        #x,y = zeros_like(x), zeros_like(y)

        ty = range(1,n+1)
        tx = range(n+1, 2*n+1)
        if system.object.radius == inf:
            ax, ay = rays.positions[:,(0,1)].T
        else:
            ax, ay = rays.angles[:,(0,1)].T
        #ax,ay = zeros_like(ax), zeros_like(ay)

        pd = ArrayPlotData(index=x)
        pd.set_data("y", y)
        plot = Plot(pd, padding=30, title="Spot Diagram",
                bgcolor="transparent")
        plot.plot(("index", "y"),
                color="green", type="scatter", marker_size=1)
        plot.tools.append(PanTool(component=plot))
        plot.tools.append(ZoomTool(component=plot,
            tool_mode="box", always_on=False))

        self.spot_diagram = plot

        pd = ArrayPlotData(index=ax[ty])
        pd.set_data("y", x[ty])
        plot = Plot(pd, padding=30, title="X Transverse",
                bgcolor="transparent")
        plot.plot(("index", "y"),
                color="green", type="line")
        plot.tools.append(PanTool(component=plot))
        plot.tools.append(ZoomTool(component=plot,
            tool_mode="box", always_on=False))

        self.x_transverse = plot

        pd = ArrayPlotData(index=ay[tx])
        pd.set_data("y", y[tx])
        plot = Plot(pd, padding=30, title="Y Transverse",
                bgcolor="transparent")
        plot.plot(("index", "y"),
                color="green", type="line")
        plot.tools.append(PanTool(component=plot))
        plot.tools.append(ZoomTool(component=plot,
            tool_mode="box", always_on=False))
        
        self.y_transverse = plot


if __name__ == "__main__":
    #s = slide_projector
    #s = tessar
    #s = lithograph
    #s = lithium
    #s = k_z_imaging
    #s = k_z_objective
    s = schwarzschild
    s.paraxial_trace()
    t = Raytrace(system=s)
    t.configure_traits()
