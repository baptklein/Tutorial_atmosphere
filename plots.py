# -*- coding: utf-8 -*-
"""
Created on Sep 2021
@authors: Baptiste KLEIN
"""

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from scipy.optimize import minimize
import src


font = {'size'   : 16,
        'weight': 'light'}
axes = {'labelsize': 16,
        'labelweight': 'light'}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)



def plot_spec_fit(W,I,I_pred=[],eps=0.03,nb=10):

    plt.figure(figsize=(20,14))
    for ii in range(nb):
        plt.plot(W,I[ii]-ii*eps,"-k",lw=2.0,label="Data")
        if len(I_pred)>0: plt.plot(W,I_pred[ii]-ii*eps,"-r",lw=1.0,label="Adjusted ref. spect")
        if ii == 0: plt.legend(ncol=2)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("FLux")
    plt.show()




def plot_normalization(Ws,Is,I_corr,I_med_tot,eps=0.04,nb=10):

    plt.figure(figsize=(15,20))
    for nn in range(nb):
        plt.plot(Ws,Is[nn]-eps*nn,"-m",lw=0.7)
        plt.plot(Ws,I_corr[nn]-eps*nn,"-k",lw=0.7)                
        plt.plot(Ws,I_med_tot[nn]-eps*nn,"-b",lw=2.0)
    plt.ylabel("Flux")
    plt.ylim(0.55,1.05)
    plt.xlabel("Wavelength [nm]")
    plt.show()                






def plot_timeseries(T_obs,T0,Porb,phase,flux,n_ini,n_end,airmass,Vc,snr_mat):

    T_wrt  = 24.*(T_obs-T0 - int((T_obs[-1]-T0)/Porb)*Porb)

    ### 1 - Time series
    ypad = 10

    plt.figure(figsize=(12,10))
    ax  = plt.subplot(411)

    ax.plot(T_wrt,flux,"-+k")
    ax.axvline(T_wrt[n_ini],ls=":",color="r")
    ax.axvline(T_wrt[n_end],ls=":",color="r")
    ax.set_xticks([])

    ax2 = ax.twiny() 
    ax2.plot(phase,flux,"-",lw=0.0)
    ax2.set_xlabel("Orbital phase")

    ax.set_ylabel("Transit curve\n", labelpad=ypad)


    ax = plt.subplot(412)
    plt.plot(T_wrt,airmass,"-k")
    plt.axvline(T_wrt[n_ini],ls=":",color="r")
    plt.axvline(T_wrt[n_end],ls=":",color="r")
    plt.xticks([])
    ax.set_ylabel("Airmass\n", labelpad=ypad)

    ax = plt.subplot(413)
    plt.plot(T_wrt,Vc,"-k")
    plt.axvline(T_wrt[n_ini],ls=":",color="r")
    plt.axvline(T_wrt[n_end],ls=":",color="r")
    plt.xticks([])
    ax.set_ylabel("RV correction\n[km/s]", labelpad=ypad)

    ax = plt.subplot(414)
    plt.plot(T_wrt,np.max(snr_mat,axis=1),"+k")
    plt.axvline(T_wrt[n_ini],ls=":",color="r")
    plt.axvline(T_wrt[n_end],ls=":",color="r")
    plt.xlabel("Time wrt transit [h]")
    ax.set_ylabel("Peak S/N\n", labelpad=ypad)

    plt.subplots_adjust(hspace=0.02)
    plt.show()


def plot_2D(x,y,Z,lab,cmap="gist_heat",n_ini=-1,n_end=-1,title="",size=(12,7)):

    X,Y  = np.meshgrid(x,y)
    zmin = np.median(Z) - 3.*Z.std()
    zmax = np.median(Z) + 3.*Z.std()

    plt.figure(figsize=size)
    ax   = plt.subplot(111)
    c    = plt.pcolor(X,Y,Z,cmap=cmap,vmin=zmin,vmax=zmax)
    cb   = plt.colorbar(c,ax=ax)
    ax.set_ylim(np.min(y),np.max(y))
    ax.set_xlim(np.min(x),np.max(x))
    if n_ini>=0: plt.axhline(y[n_ini],color="#EFFBFF",ls="--",lw=5.0)
    if n_end>=0: plt.axhline(y[n_end],color="#EFFBFF",ls="--",lw=5.0)
    ax.set_xlabel(lab[0])
    ax.set_ylabel(lab[1])
    cb.set_label(lab[2],rotation=270,labelpad=40)
    if len(title)>0: ax.set_title(title)
    plt.show()


def compare_2D(x,y,Z,lab,cmap="gist_heat",n_ini=-1,n_end=-1,title="",size=(12,7)):


    X,Y  = np.meshgrid(x,y)
    zmin = np.median(Z) - 2.*Z.std()
    zmax = np.median(Z) + 2.*Z.std()

    plt.figure(figsize=size)
    ax   = plt.subplot(111)
    c    = plt.pcolor(X,Y,Z,cmap=cmap,vmin=zmin,vmax=zmax)
    cb   = plt.colorbar(c,ax=ax)
    ax.set_ylim(np.min(y),np.max(y))
    ax.set_xlim(np.min(x),np.max(x))
    if n_ini>=0: plt.axhline(y[n_ini],color="#EFFBFF",ls="--",lw=5.0)
    if n_end>=0: plt.axhline(y[n_end],color="#EFFBFF",ls="--",lw=5.0)
    ax.set_xlabel(lab[0])
    ax.set_ylabel(lab[1])
    cb.set_label(lab[2],rotation=270,labelpad=40)
    if len(title)>0: ax.set_title(title)
    plt.show()







def plot_orders(z,wmean,orders,ind_rem=[],laby=""):

    WW,LO_pred,LO_predt = src.fit_order_wave(orders,wmean)

    plt.figure(figsize=(12,5))
    ax = plt.subplot(111)
    ax.plot(orders,z,"*k")
    if len(ind_rem) > 0:
        ax.plot(np.array(orders)[ind_rem],z[ind_rem],"^r")
    ax.set_ylabel(laby)
    ax.set_xlabel("Order")
    ax2 = ax.twiny()
    ax2.set_xticks(LO_pred)
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_xticklabels(WW)
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(LO_predt))
    ax.set_xlim(30,80)
    ax2.set_xlim(30,80)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.show()



def plot_distrib(W,rms_px,rms_pred,W_fin,rms_fin,numb):

    plt.figure(figsize=(8,5))
    plt.plot(W,rms_px,"+",color="r",alpha=0.75)
    plt.plot(W_fin,rms_fin,"+k")
    plt.plot(W,rms_pred,"-b",lw=2.0)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Dispersion")
    tt = "Order " + str(numb) + ": pixel dispersion"
    plt.title(tt)
    plt.show()

        

def plot_spectrum_dispersion(lord):


    rms_sp     = np.zeros(len(lord))
    rms_sp_s   = np.zeros(len(lord))
    rms_drs    = np.zeros(len(lord))
    rms_drs_s  = np.zeros(len(lord))
    wmean      = np.zeros(len(lord))
    LO         = np.zeros(len(lord),dtype=int)

    for kk in range(len(lord)):
        O              = lord[kk]
        disp_mes       = 1./O.SNR_mes
        disp_drs       = 1./O.SNR
        rms_sp[kk]     = np.mean(disp_mes)
        rms_sp_s[kk]   = np.std(disp_mes)
        rms_drs[kk]    = np.mean(disp_drs)
        rms_drs_s[kk]  = np.std(disp_drs)
        wmean[kk]      = O.W_mean
        LO[kk]         = O.number

    WW,LO_pred,LO_predt = src.fit_order_wave(LO,wmean)
    plt.figure(figsize=(12,5))
    ax = plt.subplot(111)
    ax.errorbar(LO,rms_sp,rms_sp_s,fmt="*",color="k",label="Reduced data",capsize=10.0)
    ax.errorbar(LO,rms_drs,rms_drs_s,fmt="o",color="m",label="DRS",capsize=5.0)
    ax.legend(ncol=2)
    ax2 = ax.twiny()
    ax2.set_xticks(LO_pred)
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_xticklabels(WW)
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(LO_predt))
    ax.set_xlim(30,80)
    ax2.set_xlim(30,80)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel("Spectrum dispersion")
    ax.set_xlabel("Order number")
    ax.set_yscale("log")
    plt.subplots_adjust(wspace=0.5,hspace = 0.)
    plt.show()



def plot_correlation_map(Vsys,Kp,sn_map,V_inj=0.0,K_inj=0.0,cmap="gist_heat",sn_cutx=[],sn_cuty=[],levels=10):

    if len(sn_cutx) == 0:
        plt.figure(figsize=(10,7))
        plt.contourf(Vsys,Kp,sn_map,levels=levels,cmap=cmap)
        plt.ylabel(r"K$_{\rm{p}}$ [km/s]")
        plt.xlabel(r"V$_{\rm{sys}}$ [km/s]")
        plt.axhline(K_inj,ls="--",lw=1.0,color="w")
        plt.axvline(V_inj,ls="--",lw=1.0,color="w")
        cb = plt.colorbar()
        cb.set_label(r"Significance [$\sigma$]",rotation=270,labelpad=40)
        plt.show()

    else:
        fig     = plt.figure(figsize=(12,12))
        grid    = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
        main_ax = fig.add_subplot(grid[1:,:3])
        y_hist  = fig.add_subplot(grid[1:,-1], yticklabels=[])
        x_hist  = fig.add_subplot(grid[:1, :3], xticklabels=[])

        main_ax.contourf(Vsys,Kp,sn_map,levels=levels,cmap=cmap)
        main_ax.axhline(K_inj,ls=":",color="w",lw=2.5)
        main_ax.axvline(V_inj,ls=":",color="w",lw=2.5)
        main_ax.set_ylabel(r"K$_{\rm{p}}$ [km/s]")
        main_ax.set_xlabel(r"V$_{\rm{sys}}$ [km/s]")

        # histogram on the attached axes
        x_hist.plot(Vsys,sn_cuty,"-k")
        x_hist.axhline(0.0,ls=":",color="gray")
        x_hist.axvline(V_inj,ls=":",color="r")
        title_x = r"Cut at K$_{\rm{p}}$ = " + str(K_inj) + " km/s"
        x_hist.set_xlabel(title_x,rotation=0,labelpad=-180)
        x_hist.set_ylabel(r"Significance [$\sigma$]")

        y_hist.plot(sn_cutx,Kp,"-k")
        y_hist.axvline(0.0,ls=":",color="gray")
        y_hist.axhline(K_inj,ls=":",color="r")
        title_y = r"Cut at V$_{\rm{sys}}$ = " + str(V_inj) + " km/s"
        y_hist.set_ylabel(title_y,rotation=270,labelpad=-170)
        y_hist.set_xlabel(r"Significance [$\sigma$]")

        plt.show()







