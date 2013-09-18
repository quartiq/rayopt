# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2013 Robert Jordens <jordens@phys.ethz.ch>
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.mlab import griddata
from matplotlib import gridspec

from .raytrace import ParaxialTrace, FullTrace
from .utils import tanarcsin, sinarctan


class Analysis(object):
    figwidth = 12.
    resize = True
    refocus = True
    print_system = True
    print_paraxial = True
    print_paraxial_full = False
    plot_paraxial = False
    plot_paraxial_full = False
    plot_heights = [1., .707, 0.]
    plot_rays = 3
    plot_transverse = True
    plot_longitudinal = True

    def __init__(self, system, run=True, **kwargs):
        self.system = system
        self.text = []
        self.figures = []
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)
        if run:
            self.run()

    def run(self):
        self.paraxial = ParaxialTrace(self.system)
        if self.refocus:
            self.paraxial.focal_plane_solve()
        if self.resize:
            self.paraxial.size_elements()
        self.system.image.radius = abs(self.paraxial.height[1])
        if self.print_system:
            self.text.append(str(self.system))
        if self.print_paraxial:
            self.text.append(str(self.paraxial))
        t = FullTrace(self.system)
        t.rays_paraxial(self.paraxial)
        if self.print_paraxial_full:
            self.text.append(str(t))

        figheight = 2*max(e.radius for e in self.system)
        figheight = min(figheight/self.paraxial.z[-1]*self.figwidth,
                2*self.figwidth/3)
        fig, ax = plt.subplots(figsize=(self.figwidth, figheight))
        self.figures.append(fig)
        self.system.plot(ax)
        if self.plot_paraxial:
            self.paraxial.plot(ax)
        if self.plot_paraxial_full:
            t.plot(ax)
        for h in self.plot_heights:
            t = FullTrace(self.system)
            t.rays_paraxial_point(self.paraxial, h,
                    nrays=self.plot_rays, clip=False)
            t.plot(ax)
        
        if self.plot_transverse is True:
            self.plot_transverse = self.plot_heights
        if self.plot_transverse:
            figheight = self.figwidth*len(self.plot_transverse)/6
            fig = plt.figure(figsize=(self.figwidth, figheight))
            self.figures.append(fig)
            self.transverse(fig, self.plot_transverse)
        if self.plot_longitudinal:
            fig = plt.figure(figsize=(self.figwidth/3, self.figwidth/6))
            self.figures.append(fig)
            self.longitudinal(fig, max(self.plot_transverse))
        return self.text, self.figures

    @staticmethod
    def setup_axes(ax, xlabel=None, ylabel=None, title=None):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.tick_params(bottom=True, top=False,
                left=True, right=False,
                labeltop=False, labelright=False,
                labelleft=True, labelbottom=True,
                direction="out", axis="both")
        ax.xaxis.set_smart_bounds(True)
        ax.yaxis.set_smart_bounds(True)
        kw = dict(rotation="horizontal",
                horizontalalignment="left",
                verticalalignment="bottom")
        if xlabel:
            ax.set_xlabel(xlabel, **kw)
        if ylabel:
            ax.set_ylabel(ylabel, **kw)
        if title:
            ax.set_title(title)
    
    @staticmethod
    def post_setup_axes(axi):
        axi.relim()
        #axi.autoscale_view(True, True, True)
        t = axi.get_xticks()
        axi.set_xticks((t[0], t[-1]))
        t = axi.get_yticks()
        axi.set_yticks((t[0], t[-1]))
        xl, xu = axi.get_xlim()
        yl, yu = axi.get_ylim()
        axi.xaxis.set_label_coords(xu, .02*(yu - yl),
                transform=axi.transData)
        axi.yaxis.set_label_coords(.02*(xu - xl), yu,
                transform=axi.transData)

    @classmethod
    def pre_setup_fanplot(cls, fig, n):
        gs = gridspec.GridSpec(n, 6)
        axpx0, axpy0, axex0, axey0 = None, None, None, None
        ax = []
        for i in range(n):
            axp = fig.add_subplot(gs.new_subplotspec((i, 0), 1, 1),
                    aspect="equal", sharex=axex0, sharey=axey0)
            axex0 = axex0 or axp
            axey0 = axey0 or axp
            axm = fig.add_subplot(gs.new_subplotspec((i, 1), 1, 2),
                    sharex=axpy0, sharey=axey0)
            axpy0 = axpy0 or axm
            axsm = fig.add_subplot(gs.new_subplotspec((i, 3), 1, 1),
                    sharex=axpx0, sharey=axey0)
            axpx0 = axpx0 or axsm
            axss = fig.add_subplot(gs.new_subplotspec((i, 4), 1, 1),
                    sharex=axpx0, sharey=axey0)
            axo = fig.add_subplot(gs.new_subplotspec((i, 5), 1, 1),
                    aspect="equal") #, sharex=axpy0, sharey=axpx0)
            ax.append((axp, axm, axsm, axss, axo))
            for axi, xl, yl in [
                    (axp, "EX", "EY"),
                    (axm, "PY", "EY"),
                    (axsm, "PX", "EY"),
                    (axss, "PX", "EX"),
                    (axo, "PX", "PY"),
                    ]:
                cls.setup_axes(axi, xl, yl)
            for axi in axp, axo:
                axi.spines["left"].set_visible(False)
                axi.spines["bottom"].set_visible(False)
                axi.tick_params(bottom=False, left=False,
                        labelbottom=False, labelleft=False)
                axi.set_aspect("equal")
        return ax

    def transverse(self, fig, heights=[1., .707, 0.],
            wavelengths=None, nrays_spot=100, nrays_line=32,
            colors="gbrcmyk"):
        paraxial = self.paraxial
        if wavelengths is None:
            wavelengths = self.system.object.wavelengths
        nh = len(heights)
        ia = self.system.aperture_index
        ax = self.pre_setup_fanplot(fig, nh)
        p = paraxial.pupil_distance[0]
        h = paraxial.pupil_height[0]
        r = paraxial.airy_radius[1]
        for hi, axi in zip(heights, ax):
            axp, axm, axsm, axss, axo = axi
            axp.add_patch(patches.Circle((0, 0), r, edgecolor="black",
                facecolor="none"))
            axp.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                    transform=axp.transAxes,
                    verticalalignment="center")
            for i, (wi, ci) in enumerate(zip(wavelengths, colors)):
                t = FullTrace(self.system)
                ref = t.rays_paraxial_point(paraxial, hi, wi,
                        nrays=nrays_spot, distribution="hexapolar",
                        clip=True)
                # plot transverse image plane hit pattern (ray spot)
                exy = t.y[-1, :, :2]
                exy = exy - exy[ref]
                axp.plot(exy[:, 0], exy[:, 1], ".%s" % ci,
                        markersize=3, markeredgewidth=0, label="%s" % wi)
                if i == 0:
                    # plot opd over entrance pupil
                    pxy = t.y[1, :, :2] + p*t.u[0, :, :2]/t.u[0, :, 2:]
                    pxy -= pxy[ref]
                    o = t.opd(ref)
                    # griddata barfs on nans
                    xyo = np.r_[pxy.T, [o]]
                    x, y, o = xyo[:, ~np.any(np.isnan(xyo), axis=0)]
                    n = 4*int(nrays_spot)**.5
                    xs, ys = np.mgrid[-1:1:1j*n, -1:1:1j*n]*h
                    os = griddata(x, y, o, xs, ys)
                    mm = np.fabs(os).max()
                    v = np.linspace(-mm, mm, 21)
                    axo.contour(xs, ys, os, v, cmap=plt.cm.RdBu_r)
                    #axo.set_title("max=%.2g" % mm)
                    # TODO normalize opd across heights
                t = FullTrace(self.system)
                ref = t.rays_paraxial_point(paraxial, hi, wi,
                        nrays=nrays_line, distribution="tee", clip=True)
                # plot transverse image plane versus entrance pupil
                # coordinates
                exy = t.y[-1, :, :2]
                exy = exy - exy[ref]
                pxy = t.y[1, :, :2] + p*t.u[0, :, :2]/t.u[0, :, 2:]
                pxy = pxy - pxy[ref]
                axm.plot(pxy[:ref, 1], exy[:ref, 1], "-%s" % ci, label="%s" % wi)
                axsm.plot(pxy[ref:, 0], exy[ref:, 1], "-%s" % ci, label="%s" % wi)
                axss.plot(pxy[ref:, 0], exy[ref:, 0], "-%s" % ci, label="%s" % wi)
        for axi in ax:
            for axii in axi:
                self.post_setup_axes(axii)

    def longitudinal(self, fig, height=1.,
            wavelengths=None, nrays=11, colors="gbrcmyk"):
        paraxial = self.paraxial
        if wavelengths is None:
            wavelengths = self.system.object.wavelengths
        gs = plt.GridSpec(1, 2)
        axl = fig.add_subplot(1, 2, 1)
        self.setup_axes(axl, "EY", "DEY") #, "distortion")
        axc = fig.add_subplot(1, 2, 2)
        self.setup_axes(axc, "EY", "EZ") #, "field")
        for i, (wi, ci) in enumerate(zip(wavelengths, colors)):
            t = FullTrace(self.system)
            t.rays_paraxial_line(paraxial, height, wi, nrays=nrays)
            a, b, c = np.split(t.y[-1].T, (nrays, 2*nrays), axis=1)
            p, q, r = np.split(t.u[-2].T, (nrays, 2*nrays), axis=1)
            xd = a[1] - np.linspace(0, height*paraxial.height[1], nrays)
            # tangential field curvature
            # -(real_y-parax_y)/(tanarcsin(real_u)-tanarcsin(parax_u))
            xt = -(b[1]-a[1])/(tanarcsin(q[1])-tanarcsin(p[1]))
            # sagittal field curvature
            # -(real_x-parax_x)/(tanarcsin(real_v)-tanarcsin(parax_v))
            xs = -(c[0]-a[0])/(tanarcsin(r[0])-tanarcsin(p[0]))
            axl.plot(a[1], xd, ci+"-", label="DEY")
            axc.plot(a[1], xt, ci+"-", label="EZt")
            axc.plot(a[1], xs, ci+"--", label="EZs")
        self.post_setup_axes(axl)
        self.post_setup_axes(axc)
        return fig
