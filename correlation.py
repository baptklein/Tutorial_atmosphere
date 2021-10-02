# -*- coding: utf-8 -*-
"""
Created on Sep 2021
@authors: Baptiste KLEIN, Florian DEBRAS
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal
from scipy import stats
import src
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from plots import *





def compute_correlation(list_ord,window,phase,Kp,Vsys):
    

    t0           = time.time()
    c0           = src.Constants().c0
    pixel_window = np.linspace(-1.14,1.14,15)
    weights      = scipy.signal.gaussian(15,std=1.14)

    ### INITIALISATION
    print("\nInitialization")
    data_tot = []
    wl       = []
    Stdtot   = []
    F        = []
    nord_tmp = len(list_ord)
    for kk in range(nord_tmp):
        std_px = list_ord[kk].I_pca.std(axis=0)
        fit_px = np.polyfit(list_ord[kk].W_red,std_px,2)
        Stdfit = np.poly1d(fit_px)
        Stdtot.append(Stdfit(list_ord[kk].W_red))
        wl.append(list_ord[kk].W_red)
        data_tot.append(list_ord[kk].I_pca)
        f = interp.interp1d(list_ord[kk].Wm,list_ord[kk].Im)
        F.append(f)

    #This is the maxi speed up of the code. We create an interpolation
    #of the model as a function of speed, integrated ovver a
    # pixel size. We just need to call this function afterwards instead of integrating
    print("Interpolate model")
    Vtot = np.linspace(-120,120,1001)
    F2D  = []   
    for i in range(nord_tmp):
        mod_int = np.zeros((len(Vtot),len(wl[i])))
        for j in range(len(Vtot)):
            mod_int[j]= np.average(F[i](list(map(lambda x: wl[i]/(1.0+(Vtot[j]+x)/c0),pixel_window))),weights=weights,axis=0)
        f2D = interp.interp1d(Vtot,mod_int.T,kind='linear')
        F2D.append(f2D)
        
    #And now the correlation, following boucher+ Submitted
    print("Compute correlation for",nord_tmp,"orders")
    Nkp            = len(Kp)
    Nv             = len(Vsys)
    correl         = np.zeros((Nkp,Nv,len(list_ord)))
    for no in range(nord_tmp):
        for ii in range(Nkp):
            for jj in range(Nv):
                vp               = src.rvp(phase,Kp[ii],Vsys[jj])
                modelij          = np.zeros((len(wl[no]),len(phase)))
                interpmod        = F2D[no](vp)
                modelij          = interpmod - np.mean(interpmod,axis=0)
                dataij           = (data_tot[no].T-np.mean(data_tot[no].T,axis=0))
                correl[ii,jj,no] = np.sum(np.sum(dataij*modelij*window,axis=1)/Stdtot[no]**2)

    print("DONE!")
    t1 = time.time()
    print("Duration:",(t1-t0)/60.,"min")
    return correl


def get_snrmap(orders_fin,Kp,Vsys,correl,Kp_lim=[120,180],Vsys_lim=[-15.,15.]):

    sel = []
    for i in orders_fin:
        sel.append(np.where(np.array(orders_fin,dtype=int)==i)[0][0])
    a      = np.sum(correl[:,:,sel],axis=2)
    b      = a[np.where((Kp<Kp_lim[0])|(Kp>Kp_lim[1]))]
    c      = b.T[np.where((Vsys<Vsys_lim[0])|(Vsys>Vsys_lim[1]))]
    snrmap = np.std(c)
    return snrmap

            
def multi_var(param,X,Y):
    amp = param[0]
    mu  = np.array([param[1],param[2]],dtype=float)
    cov = np.array([[param[3]**(2),param[5]],[param[5],param[4]**(2)]],dtype=float)
    pv  = multivariate_normal(mu,cov)
    pos = np.dstack((X,Y))
    mv  = amp*pv.pdf(pos)
    return np.array(mv,dtype=float)






    
def crit(param,X,Y,C):
    mv = multi_var(param,X,Y)
    cr = np.linalg.norm(C-mv) ### 2-norm
    return cr




def fit_multivar(x,y,C):

    sigma_x   = 2.0
    sigma_y   = 20.0
    sigma_xy  = 0.0
    mu_x      = 0.0
    mu_y      = 150.0
    amp       = 750.0
    param0    = [amp,mu_x,mu_y,sigma_x,sigma_y,sigma_xy]
    X_s,Y_s   = np.meshgrid(x,y)
    print("Fit bivariate normal law on significance map")
    res       = minimize(crit,param0,args=(X_s,Y_s,C),method="Nelder-Mead")
    p_best    = res.x
    mv_best   = multi_var(p_best,X_s,Y_s)
    return p_best,mv_best



def get_statistics(x,y,C):
    
    p_best,mv_best = fit_multivar(x,y,C)

    ### Get error bars and best parameters
    V_best  = p_best[1]
    Kp_best = p_best[2]

    V_big   =  np.linspace(x.min(),x.max(),101)
    Kp_big  =  np.linspace(y.min(),y.max(),101)
    Xb,Yb   = np.meshgrid(V_big,Kp_big)
    mv_tot  = multi_var(p_best,Xb,Yb)

    ### Max significance
    sn_max = mv_tot.max()
    print("Maximum detection:",round(sn_max,1),"sigma")

    ib1         = np.argmin(np.abs(V_big-V_best))
    ib2         = np.argmin(np.abs(Kp_big-Kp_best))
    K_sup,K_inf = 0.0,0.0
    V_sup,V_inf = 0.0,0.0

    ii    = ib2
    while mv_tot[ii,ib1]>sn_max-1.0 and ii<len(Kp_big)-1: ii+=1
    if ii == len(Kp_big)-1: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    K_sup = Kp_big[ii]-Kp_best
    ii    = ib2
    while mv_tot[ii,ib1]>sn_max-1.0 and ii>0: ii-=1
    if ii == 0: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    K_inf = Kp_best-Kp_big[ii]     
    ii    = ib1
    while mv_tot[ib2,ii]>sn_max-1.0 and ii<len(V_big)-1: ii+=1
    if ii == len(V_big)-1: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    V_sup = V_big[ii]-V_best
    ii    = ib1
    while mv_tot[ib2,ii] > sn_max-1.0 and ii>0: ii -= 1
    if ii == 0: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    V_inf = V_best-V_big[ii]        

    ### Display results
    print("Best estimates:")
    print("Kp:",round(Kp_best,1),"(+",round(K_sup,1),",-",round(K_inf,1),") km/s")
    print("V0:",round(V_best,1),"(+",round(V_sup,1),",-",round(V_inf,1),") km/s")

    return p_best,Kp_best,K_sup,K_inf,V_best,V_sup,V_inf






