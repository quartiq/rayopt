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

from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib import gridspec

from .raytrace import ParaxialTrace, FullTrace
from .utils import tanarcsin


class CenteredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center"."""
    center = 0.
    def __call__(self, value, pos=None):
        if value == self.center:
            return ""
        return super(CenteredFormatter, self).__call__(value, pos)


class Analysis(object):
    figwidth = 12.
    resize = True
    refocus = True
    print_system = True
    print_paraxial = True
    resize_full = False
    print_full = False
    plot_paraxial = False
    plot_full = False
    plot_heights = [1., .707, 0.]
    plot_rays = 3
    plot_transverse = True
    plot_spots = True
    defocus = 5
    plot_opds = True
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
            self.system.fix_sizes()
        self.system.image.radius = abs(self.paraxial.height[1])
        if self.print_system:
            self.text.append(unicode(self.system))
        if self.print_paraxial:
            self.text.append(unicode(self.paraxial))
        t = FullTrace(self.system)
        t.rays_paraxial(self.paraxial)
        if self.print_full:
            self.text.append(unicode(t))
        if self.resize_full:
            t.size_elements()
            self.system.fix_sizes()
        figheight = 2*max(e.radius for e in self.system)
        figheight = min(figheight/self.paraxial.z[-1]*self.figwidth,
                2*self.figwidth/3)
        fig, ax = plt.subplots(figsize=(self.figwidth, figheight))
        self.figures.append(fig)
        self.system.plot(ax)
        if self.plot_paraxial:
            self.paraxial.plot(ax)
        if self.plot_full:
            t.plot(ax)
        for h in 0, max(self.plot_heights):
            t = FullTrace(self.system)
            t.rays_paraxial_clipping(self.paraxial, h)
            t.plot(ax)
        
        if self.plot_transverse is True:
            self.plot_transverse = self.plot_heights
        if self.plot_transverse:
            figheight = self.figwidth*len(self.plot_transverse)/5
            fig = plt.figure(figsize=(self.figwidth, figheight))
            self.figures.append(fig)
            self.transverse(fig, self.plot_transverse)

        if self.plot_longitudinal:
            fig, ax = plt.subplots(1, 4,
                    figsize=(self.figwidth, self.figwidth/3))
            self.figures.append(fig)
            self.longitudinal(ax, max(self.plot_transverse))

        if self.plot_spots is True:
            self.plot_spots = self.plot_heights
        if self.plot_spots:
            figheight = self.figwidth*len(self.plot_spots)/self.defocus
            fig, ax = plt.subplots(len(self.plot_spots), self.defocus,
                    figsize=(self.figwidth, figheight), sharex=True,
                    sharey=True)
            self.figures.append(fig)
            self.spots(ax, self.plot_spots)

        if self.plot_opds is True:
            self.plot_opds = self.plot_heights
        if self.plot_opds:
            figheight = self.figwidth/len(self.plot_opds)
            fig, ax = plt.subplots(1, len(self.plot_opds),
                    figsize=(self.figwidth, figheight), sharex=True,
                    sharey=True)
            self.figures.append(fig)
            self.opds(ax, self.plot_opds)

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
        ax.xaxis.set_major_formatter(CenteredFormatter())
        ax.yaxis.set_major_formatter(CenteredFormatter())
        ax.locator_params(tight=True, nbins=4)
        kw = dict(rotation="horizontal")
        if xlabel:
            ax.set_xlabel(xlabel, horizontalalignment="right",
                    verticalalignment="bottom", **kw)
        if ylabel:
            ax.set_ylabel(ylabel, horizontalalignment="left",
                    verticalalignment="top", **kw)
        if title:
            ax.set_title(title)
    
    @staticmethod
    def post_setup_axes(axi):
        axi.relim()
        #axi.autoscale_view(True, True, True)
        xl, xu = axi.get_xlim()
        yl, yu = axi.get_ylim()
        axi.xaxis.set_label_coords(xu, 0, transform=axi.transData)
        axi.yaxis.set_label_coords(0, yu, transform=axi.transData)

    @classmethod
    def pre_setup_fanplot(cls, fig, n):
        gs = gridspec.GridSpec(n, 4)
        axpx, axe, axpy = None, None, None
        ax = []
        for i in range(n):
            axm = fig.add_subplot(gs.new_subplotspec((i, 0), 1, 2),
                    sharex=axpy, sharey=axe)
            axpy = axpy or axm
            axe = axe or axm
            axsm = fig.add_subplot(gs.new_subplotspec((i, 2), 1, 1),
                    sharex=axpx, sharey=axe)
            axpx = axpx or axsm
            axss = fig.add_subplot(gs.new_subplotspec((i, 3), 1, 1),
                    sharex=axpx, sharey=axe)
            ax.append((axm, axsm, axss))
            for axi, xl, yl in [
                    (axm, "PY", "EY"),
                    (axsm, "PX", "EY"),
                    (axss, "PX", "EX"),
                    ]:
                cls.setup_axes(axi, xl, yl)
        return ax

    @classmethod
    def pre_setup_xyplot(cls, axi):
        cls.setup_axes(axi)
        axi.set_aspect("equal")
        axi.spines["left"].set_visible(False)
        axi.spines["bottom"].set_visible(False)
        axi.tick_params(bottom=False, left=False,
                labelbottom=False, labelleft=False)

    def transverse(self, fig, heights=[1., .707, 0.],
            wavelengths=None, nrays_line=152,
            colors="grbcmyk"):
        paraxial = self.paraxial
        if wavelengths is None:
            wavelengths = self.system.object.wavelengths
        ax = self.pre_setup_fanplot(fig, len(heights))
        p = paraxial.pupil_distance[0]
        for hi, axi in zip(heights, ax):
            axm, axsm, axss = axi
            axm.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                    transform=axm.transAxes,
                    verticalalignment="center")
            for wi, ci in zip(wavelengths, colors):
                t = FullTrace(self.system)
                ref = t.rays_paraxial_point(paraxial, hi, wi,
                        nrays=nrays_line, distribution="tee", clip=True)
                # plot transverse image plane versus entrance pupil
                # coordinates
                y = t.y[-1, :, :2] - t.y[-1, ref, :2]
                py = t.y[1, :, :2] + p*tanarcsin(t.u[0])
                py -= py[ref]
                axm.plot(py[:ref, 1], y[:ref, 1], "-%s" % ci, label="%s" % wi)
                axsm.plot(py[ref:, 0], y[ref:, 1], "-%s" % ci, label="%s" % wi)
                axss.plot(py[ref:, 0], y[ref:, 0], "-%s" % ci, label="%s" % wi)
        for axi in ax:
            for axii in axi:
                self.post_setup_axes(axii)

    def spots(self, ax, heights=[1., .707, 0.],
            wavelengths=None, nrays=200, colors="grbcmyk"):
        paraxial = self.paraxial
        if wavelengths is None:
            wavelengths = self.system.object.wavelengths
        nh = len(heights)
        nd = ax.shape[1]
        for axi in ax.flat:
            self.pre_setup_xyplot(axi)
        z = paraxial.rayleigh_range[1]
        z = (np.arange(nd) - nd//2) * z
        for hi, axi in zip(heights, ax[:, 0]):
            axi.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                    transform=axi.transAxes,
                    verticalalignment="center")
        for zi, axi in zip(z, ax[-1, :]):
            axi.text(.5, -.1, "DZ=%.1g" % zi,
                    transform=axi.transAxes,
                    horizontalalignment="center")
        for hi, axi in zip(heights, ax):
            for wi, ci in zip(wavelengths, colors):
                r = paraxial.airy_radius[1]/paraxial.l*wi
                t = FullTrace(self.system)
                ref = t.rays_paraxial_point(paraxial, hi, wi,
                        nrays=nrays, distribution="hexapolar",
                        clip=True)
                # plot transverse image plane hit pattern (ray spot)
                y = t.y[-1, :, :2] - t.y[-1, ref, :2]
                u = tanarcsin(t.u[-2])
                for axij, zi in zip(axi, z):
                    axij.add_patch(mpl.patches.Circle((0, 0), r, edgecolor=ci,
                        facecolor="none"))
                    yi = y + zi*u
                    axij.plot(yi[:, 0], yi[:, 1], ".%s" % ci,
                            markersize=3, markeredgewidth=0, label="%s" % wi)
        for axi in ax:
            for axii in axi:
                self.post_setup_axes(axii)

    def opds(self, ax, heights=[1., .707, 0.],
            wavelength=None, nrays=200, colors="grbcmyk"):
        paraxial = self.paraxial
        if wavelength is None:
            wavelength = self.system.object.wavelengths[0]
        for axi in ax.flat:
            self.pre_setup_xyplot(axi)
        p = paraxial.pupil_distance[0]
        h = paraxial.pupil_height[0]
        mm = None
        for hi, axi in zip(heights, ax):
            axi.text(.5, 1., "OY=%s" % hi,
                    transform=axi.transAxes,
                    horizontalalignment="center")
            t = FullTrace(self.system)
            ref = t.rays_paraxial_point(paraxial, hi, wavelength,
                    nrays=nrays, distribution="hexapolar", clip=True)
            py = t.y[1, :, :2] + p*tanarcsin(t.u[0])
            # plot opd over entrance pupil
            #pp = paraxial.pupil_distance[1]
            #py = t.y[-2, :, :2] + pp*tanarcsin(t.u[-2, :])
            py -= py[ref]
            o = t.opd(ref)
            # griddata barfs on nans
            xyo = np.r_[py.T, [o]]
            x, y, o = xyo[:, np.all(np.isfinite(xyo), axis=0)]
            if len(o):
                n = 4*nrays**.5
                xs, ys = np.mgrid[-1:1:1j*n, -1:1:1j*n]*h
                os = griddata(x, y, o, xs, ys)
                if mm is None:
                    mm = np.fabs(o).max()
                    v = np.linspace(-mm, mm, 21)
                axi.contour(xs, ys, os, v, cmap=plt.cm.RdBu_r)
                axi.text(.5, -.1, u"PTP: %.3g" % o.ptp(),
                        transform=axi.transAxes,
                        horizontalalignment="center")
            # TODO normalize opd across heights
        for axi in ax:
            self.post_setup_axes(axi)

    def longitudinal(self, ax, height=1.,
            wavelengths=None, nrays=21, colors="grbcmyk"):
        paraxial = self.paraxial
        # lateral color: image relative to image at wl[0]
        # focus shift paraxial focus vs wl
        # longitudinal spherical: marginal focus vs height (vs wl)
        if wavelengths is None:
            wavelengths = self.system.object.wavelengths
        axd, axf, axs, axc = ax
        for axi, xl, yl, tl in [
                (axd, "EY", "DEY", "DIST"),
                (axf, "EY", "DEZ", "FIELD"),
                (axc, "EY", "DEY", "TCHA"),
                (axs, "EY", "DEZ", "SPHA"),
                ]:
            self.setup_axes(axi, xl, yl, tl)
        #m = paraxial.magnification[0]
        h = np.linspace(0, height*paraxial.height[1], nrays)
        for i, (wi, ci) in enumerate(zip(wavelengths, colors)):
            t = FullTrace(self.system)
            t.rays_paraxial_line(paraxial, height, wi, nrays=nrays)
            a, b, c = np.split(t.y[-1].T, (nrays, 2*nrays), axis=1)
            p, q, r = np.split(tanarcsin(t.u[-2]).T, (nrays, 2*nrays), axis=1)
            xd = (a[1] - h)/h
            xd[0] = np.nan
            axd.plot(a[1], xd, ci+"-", label="%s" % wi)
            xt = -(b[1]-a[1])/(q[1]-p[1])
            axf.plot(a[1], xt, ci+"-", label="EZt %s" % wi)
            xs = -(c[0]-a[0])/(r[0]-p[0])
            axf.plot(a[1], xs, ci+"--", label="EZs %s" % wi)
            #xs = (t.n[0, :nrays]*u[0, /t.n[-2]* - m)/m
            #axa.plot(a[1], xa, ci+"-", label="DM")
        for axi in ax:
            self.post_setup_axes(axi)
