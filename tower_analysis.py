#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:02:05 2024

@author: giordano
"""

######################
### IMPORT MODULES ###
######################

from IPython import get_ipython    
### Reset all variables at execution
#get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import curve_fit
import datetime
from tqdm import tqdm
import matplotlib.animation as animation
from functools import partial
import pandas as pd
import logging
import matplotlib as mpl


# Some styling changes
from pylab import rcParams
plt.style.use('default')

rcParams['axes.labelweight'] = 'bold'
rcParams['axes.labelsize'] = 'x-large'
rcParams['axes.titlesize'] = 'xx-large'
rcParams['axes.titleweight'] = 'bold'

plt.close('all')

########################
### DEFINE FUNCTIONS ###
########################

def clean_data(data_0, levels = 3, snowfox=[], wind_grad=[], turb_max_height=[], temp_grad = [], wind_log=[], temp_log=[], start_time=None, end_time=None, qc_threshold=0, fill_gaps = False):
    """
    Returns data with snowfox temperature measurements, 
    selects only periods with a high enough number of non-nan measurements, 
    applies different corrections to compute latent and sensible heat fluxes (SNB and WPL),
    computes vertical gradients of different variables.
    The variables are 10 min averaged to follow the structure of snowfox temperature measurements. 

    Parameters
    ----------
    data_0 : dataset
        A coordinate "heights_coords" stores the levels of measurement
    levels : int, optional
        Number of measurement levels for the tower. The default is 3.
    snowfox : dataframe, optional
        Contains the temperature information from the snowfox instruments. The default is [].
    wind_grad : dataframe, optional
        Contains the prandtl fit parameters for wind speed and maximum jet height previously computed. The default is [].
    turb_max_height : dataframe, optional
        Contains the maximum jet heights computed with uw and uT. The default is [].
    temp_grad : TYPE, optional
        Contains the prandtl fit parameters for temperature previously computed. The default is [].
    start_time : datetime, optional
        The default is None.
    end_time : datetime, optional
        The default is None.
    qc_threshold : float, optional
        Between 0 and 1. Only measurements periods with at least qc_threshold*1200 (20Hz) 
        non-nan values are kept. The default is 0.
    fill_gaps : bool, optional
        If True, the 10 min snowfox gaps are linearly interpolated to have 1 min temperature data. The default is False.

    Returns
    -------
    data : dataset
        Completed dataset.
    wind_grad : dataset
        Containing the prandtl parameters for wind speed.
    turb_max_height : dataset
        Containing the prandtl parameters for temperature.

    """


    data = data_0.copy()
    
    data['height_coords'] = data.height_coords.astype(str)
    
    data = data.sel(time=slice(start_time, end_time))
    #keep only mean +/- 5 std for pressure measurements
    data['meanPirga'] = data['meanPirga'].where((data.meanPirga.mean(dim='time', skipna=True) - 5*data.meanPirga.std(
        dim='time', skipna=True) < data['meanPirga']) &  (data['meanPirga'] < data.meanPirga.mean(dim='time', skipna=True) + 5*data.meanPirga.std(dim='time', skipna=True)))
    #data['meanTirga'] = data['meanTirga'].rolling(time = 10).mean(skipna=True) # compute the mean of the irga sensor so it is consistent with snowfox instruments
    #compute rolling average over the whole dataset
    data = data.rolling(time=10, center = True).mean(skipna=True)
    
    data = data.where(data.QCnan >= 1200 * qc_threshold) # threshold between 0 and 1
    data = data.sel(height_coords = data.height_coords[:levels])
    data = data.dropna(dim='time', subset=['QCnan'])

    if len(snowfox)>0:
        snowfox['TIMESTAMP'] = pd.to_datetime(snowfox['TIMESTAMP'])
    
        for col in snowfox.columns[1:]:
            
            snowfox[col] = snowfox[col].astype(float)
        
        da = xr.Dataset(snowfox).set_index({'dim_0':'TIMESTAMP'}).rename({'dim_0':'time'})
        da = da.sel(time=slice(start_time, end_time)) # using the same time window as the data
        
        data = xr.merge([data, da])
    
        if fill_gaps:
            ###Filling the 10-minutes gaps between snowfox measurement points
            data = data.interpolate_na(dim='time', max_gap=datetime.timedelta(0,600), method = 'linear') # what would the right method be? 

    
        data = data.assign(temp=(['time', 'height_coords'], np.array(
            [data.meanTirga[:, 0]+273.15, data['Tair_1_Avg'] + 273.15, data['Tair_2_Avg'] + 273.15]).T))
        data['temp_diff'] = data.temp - data.temp[:, 0] #difference between a level and level 0
        
        ### Compute surface temperature with stefan boltzmann law and outgoing longwave
        data['T_surf'] = (data['LWoutCor_Avg']/(5.67e-8))**0.25
        
    data['meanTirga'] = data.meanTirga + 273.15
    
    
    data['u_diff_level_0'] = data.meanU - data.meanU[:,0] #difference between a level and level 0
    if levels > 1:
        data['u_diff_level_1'] = data.meanU - data.meanU[:,1] #difference between a level and level 1
    
    ###Compute gradients of the following variables:
    grad_vars = ['meanU', 'meanT', 'uw', 'uT', 'wT', 'uu', 'vv', 'ww', 'dir', 'temp']
    
    for v in grad_vars:
        if v in data.keys() and 'height_coords' in list(data[v].coords):
            da = []
            for j in range(levels - 1):
                da.append(((data[v][:,j+1] - data[v][:,j])/(data.heights[:,j+1] - data.heights[:,j])).values)
            l = 'grad_' + v
            data[l] = (['time', 'height_coords'], np.array([[0]*len(data[v])] + da).T)

    ###Read wind gradients from prandtl fits
    if len(wind_grad)>0:
        wind_grad['Timestamp'] = pd.to_datetime(wind_grad['Timestamp'])
        
        da = xr.Dataset()
        
        wind_grad = da.from_dataframe(wind_grad).set_index(index = 'Timestamp').rename({'index': 'time'}).drop_duplicates(dim = 'time')
        wind_grad = wind_grad.sel(time=slice(start_time, end_time)) # using the same time window as the data
        ### Perform 10 min rolling average if jet data has a 1 minute time step
        if (pd.Timestamp(wind_grad.time.values[1]) - pd.Timestamp(wind_grad.time.values[0])).seconds == 60:
            wind_grad = wind_grad.rolling(time = 10).mean(skipna=True)
        
        data = xr.merge([data, wind_grad])
       
        data['max_jet_height'] = (['time', 'height_coords'], np.array([data['max jet height']]*levels).T)
        data['dudz'] = (['time', 'height_coords'], np.array([data['gradient 1 m'], data['gradient 2 m'], data['gradient 4 m']]).T)
        
    ###Read temperature gradients from prandtl fits
    if len(temp_grad)>0:
        temp_grad['Timestamp'] = pd.to_datetime(temp_grad['Timestamp'])
        
        da = xr.Dataset()
        
        temp_grad = da.from_dataframe(temp_grad).set_index(index = 'Timestamp').rename({'index': 'time'}).drop_duplicates(dim = 'time')
        temp_grad = temp_grad.sel(time=slice(start_time, end_time)) # using the same time window as the data
        ### Perform 10 min rolling average if jet data has a 1 minute time step
        if (pd.Timestamp(temp_grad.time.values[1]) - pd.Timestamp(temp_grad.time.values[0])).seconds == 60:
            temp_grad = temp_grad.rolling(time = 10).mean(skipna=True)
        
        data = xr.merge([data, temp_grad])
       
        data['dTdz'] = (['time', 'height_coords'], np.array([data['T gradient 1 m'], data['T gradient 2 m'], data['T gradient 4 m']]).T)
        
    ###Read turbulent maximum jet heights
    if len(turb_max_height) >0:
        turb_max_height['time'] = pd.to_datetime(turb_max_height['time'])
        da = xr.Dataset()
        turb_max_height = da.from_dataframe(turb_max_height).set_index(index = 'time').rename({'index': 'time'}).drop_duplicates(dim = 'time')
        turb_max_height = turb_max_height.sel(time=slice(start_time, end_time)) # using the same time window as the data
        ### Perform 10 min rolling average if jet data has a 1 minute time step
        if (pd.Timestamp(turb_max_height.time.values[1]) - pd.Timestamp(turb_max_height.time.values[0])).seconds == 60:
            turb_max_height = turb_max_height.rolling(time = 10).mean(skipna=True)
        
        data = xr.merge([data, turb_max_height])
        data['zmax_uT'] = ('time', data['zmax_uT'].values)
        data['zmax_uw'] = ('time', data['zmax_uw'].values)
    
    ###Read wind speed profiles from logarithmic fits
    if len(wind_log)>0:
        wind_log['Timestamp'] = pd.to_datetime(wind_log['Timestamp'])
        
        da = xr.Dataset()
        
        wind_log = da.from_dataframe(wind_log).set_index(index = 'Timestamp').rename({'index': 'time'}).drop_duplicates(dim = 'time')
        wind_log = wind_log.sel(time=slice(start_time, end_time)) # using the same time window as the data
        ### Perform 10 min rolling average if jet data has a 1 minute time step
        if (pd.Timestamp(wind_log.time.values[1]) - pd.Timestamp(wind_log.time.values[0])).seconds == 60:
            wind_log = wind_log.rolling(time = 10).mean(skipna=True)
        
        data = xr.merge([data, wind_log])
    
    ###Read temperature profiles from logarithmic fits
    if len(temp_log)>0:
        temp_log['Timestamp'] = pd.to_datetime(temp_log['Timestamp'])
        
        da = xr.Dataset()
        
        temp_log = da.from_dataframe(temp_log).set_index(index = 'Timestamp').rename({'index': 'time'}).drop_duplicates(dim = 'time')
        temp_log = temp_log.sel(time=slice(start_time, end_time)) # using the same time window as the data
        ### Perform 10 min rolling average if jet data has a 1 minute time step
        if (pd.Timestamp(temp_grad.time.values[1]) - pd.Timestamp(temp_grad.time.values[0])).seconds == 60:
            temp_log = temp_log.rolling(time = 10).mean(skipna=True)
        
        data = xr.merge([data, temp_log])
       
    
    ### Add time of day in hours in the dataset
    data = data.assign(tod = (['time'], (data.time.values - data.time.values.astype('datetime64[D]').astype(data.time[0].values.dtype)) / np.timedelta64(3600, 's') ))
    
    ###Calculate and correct sensible (and latent) heat fluxes
    cp = 1010
    Rv = 8.41/0.018
    Rd = 8.41/0.029
    mu = 29/18
    
    if 'meanQirga' in data.keys() and data.meanQirga.notnull().sum()>0:
        rhov  = data.meanQirga[:,0] * 0.001
        wrv = data.wQ[:,0] * 0.001
    else:
        rhov = np.zeros(len(data.time))*np.nan
        wrv = np.zeros(len(data.time))*np.nan
        rho = (data.meanPirga[:,0] * 100)/(Rd*data.meanTirga[:,0])
        data['rho'] = rho

    p = data.meanPirga[:,0] * 100
    T = data.meanTirga[:,0]
    e = rhov*Rv*T
    Tv = T/(1-e/p*0.378)
    rhod = p/(Rd*T)
    rho = p/(Rd*Tv)
    sigma = rhov/rhod
    
    if rho.notnull().sum()>0:
        data['rho'] = rho
    
    if 'rho' not in data.keys(): # still hansn't created attribute rho
        data['rho'] = (data.meanPirga[:,0] * 100)/(Rd*data.meanTirga[:,0])
    
    l = (3147 - 2.372*T) * 1000
    l[T<273.15] = 2838 * 1000 #sublimation
    wTs = data.wT[:,0]
    
    H_no_cor = data.rho*cp*data.wT
    H_hz = data.rho*cp*data.uT
    data['H_no_cor'] = H_no_cor
    data['H_hz'] = H_hz
    H = (rho*cp*wTs - 0.51*cp*T*(1 + mu*sigma)*wrv)/(1 + 0.51*(1 + mu*sigma)*rhov/rho)
    
    if 'meanQirga' in data.keys():
        E_no_cor = l*data.wQ
        data['E_no_cor'] = E_no_cor
    E = l*(1+mu*sigma) * (wrv + rhov/T * wTs)/(1 + 0.51 * (1 + mu*sigma)* rhov/rho)

    
    data['H'] = (['time', 'height_coords'], np.array([H] + [[np.nan]*len(H)]*(levels - 1)).T)
    data['E'] = (['time', 'height_coords'], np.array([E] + [[np.nan]*len(E)]*(levels - 1)).T)
    
    
    ### Decompose wind direction in northing and easting components
    data['northing'] = np.cos(data['dir']*np.pi/180) #wind coming from N (0 deg) has a northing of 1
    data['easting'] = np.sin(data['dir']*np.pi/180) #wind coming from E (90 deg) has an easting of 1

    
    ### Decide which fit is better between prandtl and log-linear
    #diff_log = 


    return data, wind_grad, turb_max_height, wind_log, temp_log


def calc_mean_dir(d, quantile = None):
    """
    Calculates the mean wind direction from an array of wind direction. 
    Possible to also calculate median and quantiles with "quantile"

    Parameters
    ----------

    d : 1-d array
        Wind direction measurements.
    quantile : float, optional
        If a float between 0 and 1, clculate the quantile of the data instead. The default is None.

    Returns
    -------
    meanD : float
        The mean wind direction.

    """
    
    s = np.sin(d*np.pi/180)
    c = np.cos(d*np.pi/180)
        
    if quantile != None:
        meanU = np.nanquantile(s, quantile, axis=0)
        meanV = np.nanquantile(c, quantile, axis=0)
    else:
        meanU = np.mean(s, axis=0)
        meanV = np.mean(c, axis=0)
    
    meanD = np.arctan2(meanU,meanV)*180/np.pi%360

    return meanD


def calc_std_dir(d):
    
    m = [~np.isnan(d)[i].any() for i in range(len(d))]
    d = d[m]
    d = d*np.pi/180
    c = np.cos(d)
    s = np.sin(d)
    
    std = np.sqrt(-2*np.log(np.sqrt(np.sum(c, axis = 0)**2 + np.sum(s, axis = 0)**2) / len(d)))
    
    return std*180/np.pi



def daily_cycle(data, var, levels, direction = False, window = 1, y_label = '', plot = False, ymin = None, ymax = None, title = None):
    """
    Computes the daily cycle of a given variable at the given levels, with one point per minute. 

    Parameters
    ----------
    data : dataset
        Must contain a variable "tod", indicating time of day in minutes.
    var : str
        The name of the variable in the dataset.
    levels : list
        The list of the levels required (possibly values of "height_coords" dimension).
    direction : bool, optional
        Set True if variable is a circular variable. The default is False.
    window : int, optional
        How many minutes to perform a running average on. The default is 1.
    y_label : str, optional
        For plotting. The default is ''.
    plot : bool, optional
        Wether to display the cycle. The default is False.
    ymin : float, optional
        To set the limits of the plot. The default is None.
    ymax : float, optional
        To set the limits of the plot. The default is None.\
    title : str, optional
        For plotting. The default is None.

    Returns
    -------
    cycle : list of dataframes
        Contains the mean daily cycle as a sepparate dataframe fr each level, in the order
        of "levels".

    """
    res = []
    
    if 'height_coords' not in data[var].dims:
        h = {levels[0]:'solid'}
        d = {levels[0]:'s'}
    
    if len(levels) == 3:
        h = {levels[0]:'dotted', levels[1]:'dashdot', levels[2]:'solid'}
        d = {levels[0]:'.', levels[1]:'s', levels[2]:'^'}
    elif len(levels) == 2:
        h = {levels[0]:'dotted', levels[1]:'dashdot'}
        d = {levels[0]:'.', levels[1]:'s'}
    else:
        h = {levels[0]:'solid'}
        d = {levels[0]:'s'}
     
    if plot:
        fig, ax = plt.subplots(figsize = (10, 7), dpi=100)
        
    for height in h:
        if not direction:
            if 'height_coords' in data[var].dims:
                daily_mean = data[[var, 'tod']].sel(height_coords=height).groupby('tod').mean(
            )[var].dropna(dim='tod').pad(tod=window, mode='wrap').rolling(tod=window, center=True).mean(skipna=True)[window:-window]
            else:
                daily_mean = data[[var, 'tod']].groupby('tod').mean(
            )[var].dropna(dim='tod').pad(tod=window, mode='wrap').rolling(tod=window, center=True).mean(skipna=True)[window:-window] 
        else:
            c = np.sin(data[var]*np.pi/180)
            s = np.cos(data[var]*np.pi/180)
            c = c.to_dataset(name='c')
            c['tod'] = data['tod']
            s = s.to_dataset(name='s')
            s['tod'] = data['tod']
            if 'height_coords' in data[var].dims:
                daily_c = c.sel(height_coords=height).groupby('tod').mean(
                ).c.dropna(dim='tod').pad(tod=window, mode='wrap').rolling(tod=window, center=True).mean(skipna=True)[window:-window]
                daily_s = s.sel(height_coords=height).groupby('tod').mean(
                ).s.dropna(dim='tod').pad(tod=window, mode='wrap').rolling(tod=window, center=True).mean(skipna=True)[window:-window]
            else:
                daily_c = c.groupby('tod').mean(
                ).c.dropna(dim='tod').pad(tod=window, mode='wrap').rolling(tod=window, center=True).mean(skipna=True)[window:-window]
                daily_s = s.groupby('tod').mean(
                ).s.dropna(dim='tod').pad(tod=window, mode='wrap').rolling(tod=window, center=True).mean(skipna=True)[window:-window]
            daily_mean = np.arctan2(daily_c, daily_s)*180/np.pi % 360
        
        res.append(daily_mean)
        
        if plot:
            if direction:
                ax.plot(daily_mean.tod, daily_mean, c = 'k', marker = d[height], linewidth = 0, label = height)
            else:
                ax.plot(daily_mean.tod, daily_mean, c = 'k', linestyle = h[height], label = height)
    if plot: 
        ax.set_xlabel('Time of day (h)')
        ax.set_xlim(0, 24)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Daily cycle')
        if window > 1:
            dt = daily_mean.tod.values[1] - daily_mean.tod.values[0]
            #ax.annotate('rolling average over \n{} minutes'.format(round(window*dt*60, 2)), (0.7, 0.9), weight='bold', xycoords='axes fraction')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
    
    cycle = res[0] 
    return cycle


def surface_energy_balance(data_orig, h = 1, T_var = 'meanT', u_var = 'meanU', T_up = np.array([None]*2), T_down = np.array([None]*2), dx_up = None, dx_down = None, plot = False):
    """
    Computes the surface energy balance from radiative and turbulent fluxes measurements, as well as
    advection if needed.

    Parameters
    ----------
    data_orig : dataset
        
    h : int, optional
        Measurement height. The default is 1.
    T_var : str, optional
        Variable for temperature used for the advection calculation. The default is 'meanT'.
    u_var : str, optional
        Variable for wind speed used for the advection calculation. The default is 'meanU'.
    T_up : 1-d array, optional
        Temperature measuements from the station uphill (for advection). 
        The default is np.array([None]*2) in case advection is not computed.
    T_down : 1-d array, optional
        Temperature measuements from the station downhill (for advection). 
        The default is np.array([None]*2) in case advection is not computed.
    dx_up : float, optional
        Distance between station and uphill station. The default is None.
    dx_down : float, optional
        Distance between station and downhill station. The default is None.
    plot : bool, optional
        Wwther to plot the result. The default is False.

    Returns
    -------
    residual : data array
        Contains the residual of the surface energy balance.
    A : data array
        Contains the values computed for advection in the surface energy balance. 
        (zeros if no advection is computed).

    """
    data = data_orig.dropna(dim = 'time', subset = ['SWin_Avg'])
    
    if 'height_coords' in data.dims:
        data = data.sel(height_coords = '1 m')

        
    A = np.zeros(len(data.SWin_Avg))
    if (T_up != None).any() and (T_up != None).any() and dx_up != None and dx_down != None:
        
        T_up = T_up[data_orig.SWin_Avg.notnull()]
        T_down = T_down[data_orig.SWin_Avg.notnull()]
        
        U = data[u_var].values
        
        dTdx = ((T_down - data[T_var])*dx_up**2 + (data[T_var] - T_down)* dx_down**2)/((dx_up + dx_down)*dx_up*dx_down)
        A = data.rho.values * 1005 * h * U * dTdx.values
    
    SWin = data.SWin_Avg
    SWout = data.SWout_Avg
    LWin = data.LWinCor_Avg
    LWout = data.LWoutCor_Avg
    Rn = SWin + LWin - SWout - LWout
    H = data.H
    E = data.E
    residual = Rn - H - E + A
    
    t = residual.time
    
    if plot:
        fig, ax = plt.subplots(figsize = (10,6), dpi = 150, layout = 'constrained')
        #plt.plot(t, SWin, 'y', label = 'SWin')
        #plt.plot(t, SWout, 'y:', label = 'SWout')
        #plt.plot(t, LWin, 'b', label = 'LWin')
        #plt.plot(t, LWout, 'b:', label = 'LWout')
        ax.plot(t, Rn, 'r', label = 'Net radiation')
        ax.plot(t, H, 'magenta', label = 'Sensible heat')
        ax.plot(t, E, 'darkgreen', label = 'Latent heat')
        if (A != 0).any():
            ax.plot(t, A, 'slategray', label = 'Advection')
        ax.set_xlabel('Timestamp', size = 30)
        ax.set_ylabel('Flux (W m$^{-2}$)', size = 30)
        ax.tick_params(labelsize = 20)
        ax.plot(t, np.zeros(len(t)), 'k:')
        ax.plot(t, residual, 'k', linewidth = 2, label = 'Residual')
        ax.legend(fontsize = 20)
    
    return residual, A


def plot_timeseries(data, variable, n_tower, varname='', ylabel=''):
    """
    Plots the timeseries of a given variable for a given tower

    Parameters
    ----------
    data : xarray dataset
        contains the data
    variable : str
        name of the variable to plot
    n_tower : int
        number identifier of the tower
    varname : str, optional
        name of the variable in the title
    ylabel : str, optional
        to put units in the y-label. The default is ''.

    Returns
    -------
    fig : figure

    ax : axes

    """

    if varname == '':
        varname = variable

    if variable == 'dir' or variable == 'stddir':
        ls = '.'
    else:
        ls = '-'

    fig, ax = plt.subplots(1, 1, figsize=(15, 7), dpi=150)
    ax.plot(data.coords['time'].values, data[variable],
            ls, label=data.coords['height_coords'].values)
    ax.legend()
    ax.set_title('Timeseries of {} for tower {}'.format(varname, n_tower))
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    return fig, ax


def Prandtl_U(z, aL, aV, Km_min):
    """
    Prandtl model with Km(z) for the wind speed.
    The wind profile is given by Charrondiere et al. 2020.
    
    Parameters
    ----------
    z : 1d array
        contains the heights at which to calculate the model
    aL : float
    aV : float
    Km_min : float

    Returns
    -------
    U_prandtl : 1d array
        the values of the wind speed calculated at z

    """

    U_prandtl = np.zeros(len(z))

    Km_L0 = Km_min + aL*z
    Km_V0 = Km_min + aV*z

    L0 = (1/Pr**0.25) * np.sqrt(2 * Km_L0 / (N_ref * np.sin(alpha)))
    V0 = Pr**0.25 * ((np.sqrt(2) * g * np.abs(Fs)) /
                     (Ts * np.sqrt(Km_V0 * N_ref**3 * np.sin(alpha))))

    for i, h in enumerate(z):
        if h < z0:
            U_prandtl[i] = V0[i] * h/z0
        else:
            U_prandtl[i] = V0[i]*np.sin((h-z0)/L0[i])*np.exp(-(h-z0)/L0[i])

    return U_prandtl


def Prandtl_T(z, aL, aT, Km_min):
    """
    Prandtl model with Km(z) for the air temperature in celsius.
    The temperature profile is given by Brun et al. 2017.

    Parameters
    ----------
    z : 1d array
         the heights at which to calculate the model
    aL : float
    aT : float
    Km_min : float

    Returns
    -------
    UTprandtl : 1d array
        the values of the temperature calculated at z

    """

    Km_L0 = Km_min+aL*z
    Km_T0 = Km_min+aT*z

    L0 = (1/Pr**0.25) * np.sqrt(2 * Km_L0 / (N_ref * np.sin(alpha)))
    T0 = Pr**0.75 * Fs * np.sqrt(2) / np.sqrt(Km_T0 * N_ref * np.sin(alpha))

    T_ref_sol = -T0[0]
    T_ref_sol = T_ref_sol + Ts - 273.15
    T_ref = T_ref_sol + z*dTz_ref

    # At the ground (z=0) T_Prandtl = T_ref
    T_prandtl = T0*np.cos(z/L0)*np.exp(-z/L0) + T_ref
    return (T_prandtl)


def log_u(z, a, b, c):
    
    u_log = a*np.log(z) + b*z + c
    return u_log


def log_T(z, a, b, c):
    
    T_log = a*np.log(z) + b*z + c
    return T_log
    
    
def bootstrap_fit_Prandtl_U(data, n, x, z):
    """
    Computes the bootstrapped prandtl fit for wind speed.

    Parameters
    ----------
    data : xarray dataframe
        
    n : int
        Number of bootstrap iterations
    x : float
        Fraction of the data to be used in an iteration (0<=x<=1)
    z : 1d-array
        Height coordinates for the prandtl profile 

    Returns
    -------
    mean_profile : 1-d array
        Bootstrapped prandtl profile

    """
    indexes = np.arange(len(data.time))
    store_profiles = np.zeros((n, len(z)))
    for i in tqdm(range(n)):
        index_subset = np.random.choice(indexes, int(x*len(indexes)))
        subset = data.isel(time = index_subset)
        subset_mean = subset.mean(dim = 'time')
        try:
            [aL, aV, Km_min], _ = curve_fit(Prandtl_U, [0] + list(subset_mean.heights.values), [0] + list(subset_mean.meanU.values), bounds=(
    0, np.inf), check_finite=True, p0=[0.01, 0.01, 0.01], maxfev=max_ev)
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue
        profile = Prandtl_U(z, aL, aV, Km_min)
        store_profiles[i] = profile
    mean_profile = np.mean(store_profiles, axis = 0)
    return mean_profile


def plot_fit_Prandtl(dataset_T1, dataset_T2, z, data_kaiji = [], c_list = ['gray']*20):
    """
    Plots and returns the Prandtl model fits for given datasets. 

    Parameters
    ----------
    dataset_T1 : list
        The datasets selected by time windows for tower 1 
    dataset_T1 : list
        The datasets selected by time windows for tower 2
    z : 1d array
        The heights at which to calculate the model.
    data_kaiji : list, optional
        The datasets select for the kaijo instrument at 0.3-0.5 m. The default is [].
    c_list : list, optional
        To color the lines according to time. The default is all gray.

    Returns
    -------
    fig : figure

    axs : axes

    U_fit_list : list
        fits of wind speed for each element of the datasets
    T_fit_list : list
        fits of wind speed for each element of the datasets

    """

    fig, axs = plt.subplots(1, 2, figsize=(16, 7), layout = 'constrained')
    

    U_fit_list = []
    T_fit_list = []

    for i in range(len(dataset_T1)):
        
        if i == 0:
            axs[0].scatter(np.nan, np.nan,marker='^', c=color_stations['T303'], label = 'T303')
            axs[1].scatter(np.nan, np.nan,marker='^', c=color_stations['T303'], label = 'T303')

        data_katabatic = dataset_T1[i].mean(dim='time')
        try:
            [aL, aV, Km_min], _ = curve_fit(Prandtl_U, np.concatenate(([0], data_katabatic.heights)), np.concatenate(
                ([0], data_katabatic.meanU)), bounds=(0, np.inf), maxfev=max_ev)
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue
        U_fit = Prandtl_U(z, aL, aV, Km_min)
        try:
            [aL, aT, Km_min], _ = curve_fit(Prandtl_T, np.concatenate(([0], data_katabatic.heights)), np.concatenate(
                ([273.15], data_katabatic.temp))-273.15, bounds=(0, np.inf), maxfev=max_ev)
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue
        T_fit = Prandtl_T(z, aL, aT, Km_min)

        axs[0].scatter(data_katabatic.meanU,
                       data_katabatic.heights, marker='^', c=color_stations['T303'])
        axs[0].plot(U_fit, z, c=c_list[i], zorder=0)

        axs[1].scatter(data_katabatic.temp-273.15,
                       data_katabatic.heights, marker='^', c=color_stations['T303'])
        axs[1].plot(T_fit, z, c=c_list[i], zorder=0)

        U_fit_list.append(U_fit)
        T_fit_list.append(T_fit)
        
        print(dataset_T1[i].time[0].values)
        
    for i in range(len(dataset_T2)):

        if i == 0:
            if i == 0:
                axs[0].scatter(np.nan, np.nan, marker='^', c=color_stations['T275'], label = 'T275')
                axs[1].scatter(np.nan, np.nan, marker='^', c=color_stations['T275'], label = 'T275')
            lab2 = 'Short-path sonic anemometer'
        else:
            lab2 = ''

        data_katabatic = dataset_T2[i].mean(dim='time')
        try:
            if len(data_kaiji) > 0:
                data_k = data_kaiji[i].mean(dim='time')
                #[aL, aV, Km_min], _ = curve_fit(Prandtl_U, np.concatenate(([0.0], data_k.heights.values, data_katabatic.heights.values)), np.concatenate(
                #    ([0], data_k.meanU.values, data_katabatic.meanU.values)), bounds=(0, np.inf), maxfev=max_ev, p0=[0.01, 0.01, 0.01])
            else:
                [aL, aV, Km_min], _ = curve_fit(Prandtl_U, np.concatenate(([0.0], data_katabatic.heights.values)), np.concatenate(
                    ([0], data_katabatic.meanU.values)), bounds=(0, np.inf), maxfev=max_ev, p0=[0.01, 0.01, 0.01])
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue
        U_fit = Prandtl_U(z, aL, aV, Km_min)
        try:
            [aL, aT, Km_min], _ = curve_fit(Prandtl_T, np.concatenate(([0], data_katabatic.heights.values)), np.concatenate(
                ([273.15], data_katabatic.temp.values))-273.15, bounds=(0, np.inf), maxfev=max_ev, p0=[0.01, 0.01, 0.01])
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue
        T_fit = Prandtl_T(z, aL, aT, Km_min)

        axs[0].scatter(data_katabatic.meanU,
                       data_katabatic.heights, marker='^', c=color_stations['T275'])
        axs[0].plot(U_fit, z, c=c_list[i], zorder=0)

        axs[1].scatter(data_katabatic.temp-273.15,
                       data_katabatic.heights, marker='^', c=color_stations['T275'])
        axs[1].plot(T_fit, z, c=c_list[i], zorder=0)

        U_fit_list.append(U_fit)
        T_fit_list.append(T_fit)
        
        if len(data_kaiji) > 0: 
            axs[0].scatter(data_kaiji[i].meanU.mean(), data_kaiji[i].heights.mean(), c = 'r', marker = 's', label = lab2)
            #axs[0].scatter(data_kaiji[i].meanT.mean(), data_kaiji[i].heights.mean(), c = 'r', marker = 's', label = lab2)

    axs[0].set_xlabel('Wind speed (m s$^{-1}$)', size = 20)
    axs[0].set_ylabel('Height (m)', size = 20)

    axs[1].set_xlabel('Temperature (°C)', size = 20)
    axs[1].set_ylabel('Height (m)', size = 20)

    axs[0].tick_params(labelsize = 20)
    axs[1].tick_params(labelsize = 20)
    
    axs[0].legend(loc = 'upper left')
    axs[1].legend()

    fig.suptitle('Prandtl model fit', weight='bold', size='xx-large')


    return fig, axs, U_fit_list, T_fit_list


def fit_prandtl_u(data, path):
    """
    Calculates the wind speed profile, the wind gradient and the maximum jet height for a timeseries of wind measurements.

    Parameters
    ----------
    data : DataFrame
        Contains the wind speed data at 3 elevations.
    path : str
        The location where to save the calculations.

    """

    file = open(path, 'w')
    file.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('gradient 1 m', 'gradient 2 m', 'gradient 4 m',
                          'max jet height', 'max jet speed', 'aL_U', 'aV_U', 'Km_U'))
    
    #(len(data.time)
    # contains the gradient at the 3 levels, the maximum jet height and speed, the 3 parameters for the fit
    grads = np.zeros(8)
    grads[:] = np.nan

    #maxgrads = np.zeros(len(data.time))
    j = 0

    for i in tqdm(range(len(data.time))):

        u0 = 0
        h0 = 0  

        U = u1, u2, u4 = data.meanU[i].values  # get the wind speeds at the levels
        H = h1, h2, h4 = data.heights[i].values  # get the real heights

        j = j+1

        try:
            fit = [aL, aV, Km_min], _ = curve_fit(Prandtl_U, np.conatenate([[h0], H]), np.concatenate([[u0, U]]), bounds=(
                0, np.inf), maxfev=max_ev, check_finite=True, p0=[0.01, 0.01, 0.01])

        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue

        U_fit = Prandtl_U(z, aL, aV, Km_min)

        grad = np.gradient(U_fit, z[1]-z[0])

        grads[:3] = [round(grad[int(h/max(z)*len(z))], 5)
                        for h in [h1, h2, h4]]
        grads[3] = round(z[np.argmax(U_fit)], 3)
        if grads[3] >= h4:
            grads[3] = np.nan

        #maxgrads[j-1] = np.max(grad)
        
        grads[4] = round(np.max(U_fit), 3)

        grads[5] = round(aL, 5)
        grads[6] = round(aV, 5)
        grads[7] = round(Km_min, 5)
        
        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(pd.Timestamp(Data_T2.time.values[i]), grads[0], grads[1], grads[2], grads[3],
                                                          grads[4], grads[5], grads[6], grads[7]))
        
    file.close()
    #res_df = pd.DataFrame(grads, columns=['gradient 1 m', 'gradient 2 m', 'gradient 4 m',
    #                      'max jet height', 'max jet speed', 'aL', 'aV', 'km'], index=data.time.values[:])


def fit_prandtl_T(data, path):
    """
    Calculates the temperature profile for a timeseries of temperature measurements, with the Prandtl model.

    Parameters
    ----------
    data : DataFrame
        Contains the temperature data at 3 elevations.
    path : str
        The location where to save the calculations.

    """

    file = open(path, 'w')
    file.write('\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('T gradient 1 m', 'T gradient 2 m', 'T gradient 4 m', 'aL_T', 'aT_T', 'Km_T'))
    
    #(len(data.time)
    # contains the gradient at the 3 levels, the 3 parameters for the fit
    grads = np.zeros(6)
    grads[:] = np.nan

    #maxgrads = np.zeros(len(data.time))
    j = 0

    for i in tqdm(range(len(data.time))):
        
        T0 = 273.15
        if data.T_surf[i] < 273.15:
            T0 = data.T_surf[i]
            
        #h0 = z0
        h0 = 0

        T = T1, T2, T4 = data.temp[i].values  # get the temperature at the levels
        H = h1, h2, h4 = data.heights[i].values  # get the real heights

        j = j+1

        try:
            fit = [aL, aT, Km_min], _ = curve_fit(Prandtl_T, np.concatenate([[h0], H]), np.concatenate([[T0], T])-273.15, 
                                                  bounds=(0, np.inf), maxfev=max_ev, p0=[0.01, 0.01, 0.01])
            
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue

        T_fit = Prandtl_T(z, aL, aT, Km_min)

        grad = np.gradient(T_fit, z[1]-z[0])

        grads[:3] = [round(grad[int(h/max(z)*len(z))], 5)
                        for h in [h1, h2, h4]]

        grads[3] = round(aL, 5)
        grads[4] = round(aT, 5)
        grads[5] = round(Km_min, 5)
        
        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(pd.Timestamp(Data_T2.time.values[i]), grads[0], grads[1], grads[2], grads[3],
                                                          grads[4], grads[5]))
        
    file.close()  
    
  
def fit_log_u(data, path):
    """
    Calculates the wind speed profile for a timeseries of wind speed measurements, with a lin-log model.

    Parameters
    ----------
    data : DataFrame
        Contains the temperature data at 3 elevations.
    path : str
        The location where to save the calculations.

    """

    file = open(path, 'w')
    file.write('\t{}\t{}\t{}\n'.format('aU', 'bU', 'cU'))
    
    #(len(data.time)
    # contains the gradient at the 3 levels, the 3 parameters for the fit
    fits = np.zeros(3)
    fits[:] = np.nan
    
    #maxgrads = np.zeros(len(data.time))
    j = 0

    for i in tqdm(range(len(data.time))):


        U = u1, u2, u4 = data.meanU[i].values  # get the wind speeds at the levels
        H = h1, h2, h4 = data.heights[i].values  # get the real heights

        j = j+1
        
        try:
            fit = [a,b,c], _ = curve_fit(log_u, np.concatenate([[z0], H]), np.concatenate([[0], U]), maxfev=max_ev)
            
        except (RuntimeError, TypeError, ValueError) as e:
             logging.debug(...)
             continue
         
        u_fit_log = log_u(np.array(H), a, b, c)
        
        fits[0] = round(a, 5)
        fits[1] = round(b, 5)
        fits[2] = round(c, 5)
        
        
        file.write('{}\t{}\t{}\t{}\n'.format(pd.Timestamp(Data_T2.time.values[i]), *fits))
        
        fits[:] = np.nan
        
    file.close()

    
def fit_log_T(data, path):
    """
    Calculates the temperature profile for a timeseries of temperature measurements, with a lin-log model.

    Parameters
    ----------
    data : DataFrame
        Contains the temperature data at 3 elevations.
    path : str
        The location where to save the calculations.

    """

    file = open(path, 'w')
    file.write('\t{}\t{}\t{}\n'.format('aT', 'bT', 'cT'))
    
    #(len(data.time)
    # contains the gradient at the 3 levels, the 3 parameters for the fit
    fits = np.zeros(3)
    fits[:] = np.nan
    
    #maxgrads = np.zeros(len(data.time))
    j = 0

    for i in tqdm(range(len(data.time))):

        T0 = 273.15
        if data.T_surf[i] < 273.15:
            T0 = data.T_surf[i]
        
        T = T1, T2, T4 = data.temp[i].values  # get the wind speeds at the levels
        H = h1, h2, h4 = data.heights[i].values  # get the real heights

        j = j+1
        
        try:
            fit = [a,b,c], _ = curve_fit(log_T, np.concatenate([[z0], H]), np.concatenate([[T0], T]), maxfev=max_ev)
            
        except (RuntimeError, TypeError, ValueError) as e:
             logging.debug(...)
             continue
         
        u_fit_log = log_u(np.array(H), a, b, c)
        
        fits[0] = round(a, 5)
        fits[1] = round(b, 5)
        fits[2] = round(c, 5)
        
        
        file.write('{}\t{}\t{}\t{}\n'.format(pd.Timestamp(Data_T2.time.values[i]), *fits))
        
        fits[:] = np.nan
        
    file.close()
    


def best_fit_u(data, path):
    
    file = open(path, 'w')
    file.write('\t{}\t{}\t{}\n'.format('best_fit','prandtl_rmse', 'log_rmse'))
    
    #(len(data.time)
    fits = np.zeros(1)
    fits[:] = np.nan
    
    #maxgrads = np.zeros(len(data.time))
    j = 0

    for i in tqdm(range(len(data.time))):

        u = data.meanU[i].values
        fit_log = log_u(data.heights[i].values, data.aU[i].values, data.bU[i].values, data.cU[i].values)
        fit_pr  = Prandtl_U(data.heights[i].values, data.aL_U[i].values, data.aV_U[i].values, data.Km_U[i].values)
        diff_log = np.sqrt(np.sum((fit_log - u)**2))
        diff_pr  = np.sqrt(np.sum((fit_pr - u)**2))
        
        if np.isnan(data.aL_U[i]) and np.isnan(data.aU[i]):
            best_fit = np.nan
        elif np.isnan(data.aL_U[i]):
            best_fit = 'log'
        else:
            if diff_log < diff_pr:
                best_fit = 'log'
            if diff_log > diff_pr:
                best_fit = 'prd'
        
        file.write('{}\t{}\t{}\t{}\n'.format(pd.Timestamp(Data_T2.time.values[i]), best_fit, diff_pr, diff_log))
        
        fits[:] = np.nan
        
    file.close()
    
    
    
def best_fit_T(data, path):
    
    file = open(path, 'w')
    file.write('\t{}\t{}\t{}\n'.format('best_fit','prandtl_rmse', 'log_rmse'))
    
    #(len(data.time)
    # contains the gradient at the 3 levels, the 3 parameters for the fit
    fits = np.zeros(1)
    fits[:] = np.nan
    
    #maxgrads = np.zeros(len(data.time))
    j = 0

    for i in tqdm(range(len(data.time))):

        T = data.temp[i].values
        fit_log = log_T(data.heights[i].values, data.aT[i].values, data.bT[i].values, data.cT[i].values)
        fit_pr  = Prandtl_T(data.heights[i].values, data.aL_T[i].values, data.aT_T[i].values, data.Km_T[i].values)
        diff_log = np.sqrt(np.sum((fit_log - T)**2))
        diff_pr  = np.sqrt(np.sum((fit_pr - (T - 273.15))**2))
        
            
        if np.isnan(data.aL_T[i]) and np.isnan(data.aT[i]):
            best_fit = np.nan
        elif np.isnan(data.aL_T[i]):
            best_fit = 'log'
        else:
            if diff_log < diff_pr:
                best_fit = 'log'
            else:
                best_fit = 'prd'
        
        file.write('{}\t{}\t{}\t{}\n'.format(pd.Timestamp(Data_T2.time.values[i]), best_fit, diff_pr, diff_log))
        
        fits[:] = np.nan
        
    file.close()



def animate_katabatic_prandtl(data_k, period, z, dt, n_tower):
    """
    Animates the Prandtl fit for a dataset over a period of time for a given tower

    Parameters
    ----------
    data_k : xarray dataset
        contains the data
    period : tuple of datetime.datetime 
        the start and the end of the period which is animated
    z : 1d array
        the heights at which to calculate the model
    dt : int
        timestep in seconds
    n_tower : int
        number identifier of the tower

    Returns
    -------
    ani : animation
   
    u_fit_list : list
        the fits for windspeed at each timestep
    T_fit_list : list
        the fits for temperature at each timestep
    time_list : list
        the timestamps of each timestep

    """

    col = ['green', 'blue']

    dt = datetime.timedelta(0, dt)

    time_list = []
    u_data_list = []
    T_data_list = []
    heights_list = []
    u_fit_list = []
    T_fit_list = []

    for i in tqdm(range(0, int((period[-1]-period[0])/dt))):

        time_slice = slice(period[0] + i*dt, period[0] + (i+1)*dt)
        data_katabatic = data_k.sel(time=time_slice).mean(dim='time')

        try:
            [aL, aV, Km_min], _ = curve_fit(Prandtl_U, np.concatenate(
                ([0], data_katabatic.heights)), np.concatenate(([0], data_katabatic.meanU)), bounds=(0, np.inf))
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue
        U_fit = Prandtl_U(z, aL, aV, Km_min)

        try:
            [aL, aT, Km_min], _ = curve_fit(Prandtl_T, np.concatenate(([0], data_katabatic.heights)), np.concatenate(
                ([273.15], data_katabatic.temp))-273.15, bounds=(0, np.inf))
        except (RuntimeError, TypeError, ValueError) as e:
            logging.debug(...)
            continue
        T_fit = Prandtl_T(z, aL, aT, Km_min)

        time_list.append(datetime.datetime.utcfromtimestamp((data_k.sel(
            time=time_slice).time[0].values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')))

        u_data_list.append(list(data_katabatic.meanU.values))
        T_data_list.append(list(data_katabatic.temp.values - 273.15))
        heights_list.append(list(data_katabatic.heights.values))

        u_fit_list.append(U_fit)
        T_fit_list.append(T_fit)

    fig, axs = plt.subplots(1, 2, figsize=(12, 10), dpi=200)

    sc0 = axs[0].scatter(u_data_list[0], heights_list[0],
                         c=col[n_tower-1], marker='^')
    f0 = axs[0].plot(u_fit_list[0], z, c='gray')
    sc1 = axs[1].scatter(T_data_list[0], heights_list[0],
                         c=col[n_tower-1], marker='^')
    f1 = axs[1].plot(T_fit_list[0], z, c='gray')

    umax = 1.1*np.max(u_fit_list)
    Tmin = 0.9*np.min(T_fit_list)
    Tmax = 1.1*np.max(T_fit_list)

    fig.suptitle('Prandtl model fit for tower {} \n {}'.format(
        n_tower, time_list[0]), weight='bold', size='xx-large')

    axs[0].set_xlim(0, umax)
    axs[1].set_xlim(Tmin, Tmax)

    axs[0].set_xlabel('Wind speed (m/s)')
    axs[0].set_ylabel('Height (m)')

    axs[1].set_xlabel('Temperature (°C)')
    axs[1].set_ylabel('Height (m)')
    
    plt.tight_layout()

    def update_fig(i):
        axs[0].cla()
        axs[1].cla()
        sc0 = axs[0].scatter(u_data_list[i], heights_list[i],
                             c=col[n_tower-1], marker='^')
        f0 = axs[0].plot(u_fit_list[i], z, c='gray')
        sc1 = axs[1].scatter(T_data_list[i], heights_list[i],
                             c=col[n_tower-1], marker='^')
        f1 = axs[1].plot(T_fit_list[i], z, c='gray')

        axs[0].set_xlim(0, umax)
        axs[1].set_xlim(Tmin, Tmax)

        axs[0].set_xlabel('Wind speed (m/s)')
        axs[0].set_ylabel('Height (m)')

        axs[1].set_xlabel('Temperature (°C)')
        axs[1].set_ylabel('Height (m)')

        fig.suptitle('Prandtl model fit for tower {} \n {}'.format(
            n_tower, time_list[i]), weight='bold', size='xx-large')

    ani = animation.FuncAnimation(fig, update_fig, frames=len(
        u_fit_list), interval=250, repeat_delay=1500, blit=False)  # create the animation

    return ani, u_fit_list, T_fit_list, time_list

def plot_4_timeseries(path):
    """
    Plots timeseries of temperature, wind speed, wind direction and pressure.

    Parameters
    ----------
    path : str
        Path for saving the figure.

    Returns
    -------
    None.

    """
    lab_size = 20
    m_size = 4
    l = [['(a)', '(b)'], ['(c)', '(d)']]
    col = [mpl.colormaps['viridis'].resampled(4)(i) for i in range(3)]
    fig, ax = plt.subplots(2,2, figsize = (16,10), sharex = True, layout = 'constrained')
    for i,h in enumerate(Data_T2.height_coords.values):
        ax[0,0].plot(Data_T2.time, Data_T2.sel(height_coords = h).temp - 273.15, '.', markersize = m_size, c = col[i], label = h)
        ax[0,1].plot(Data_T2.time, Data_T2.sel(height_coords = h).meanU, '.', markersize = m_size, c = col[i], label = h)
        ax[1,0].plot(Data_T2.time, Data_T2.sel(height_coords = h).dir, '.', markersize = m_size, c = col[i], label = h)
    ax[0,0].legend(loc = 'upper left')
    ax[0,1].legend(loc = 'upper left')
    ax[1,0].legend(loc = 'upper left')
    ax[1,1].plot(Data_T2.time, Data_T2.meanPirga[:,0], '.', markersize = m_size, c = 'k')
    ax[0,0].set_xlim(pd.Timestamp(2023,8,16,12), pd.Timestamp(2023,9, 7, 12))
    ax[1,0].set_xlabel('Time (UTC)', size = lab_size)
    ax[1,1].set_xlabel('Time (UTC)', size = lab_size)
    ax[0,0].set_ylabel('Mean \nTemperature ($^{\circ}$C)' , size = lab_size)
    ax[0,1].set_ylabel('Mean wind \nspeed (m.s$^{-1}$)', size = lab_size)
    ax[1,0].set_ylabel('Mean wind \ndirection ($^{\circ}$)', size = lab_size)
    ax[1,1].set_ylabel('Mean atmospheric \npressure (hPa)', size = lab_size)
    ax[0,0].set_ylim(-2, 18)
    ax[0,1].set_ylim(-1, 12)
    ax[1,0].set_ylim(0, 360)
    ax[1,1].set_ylim(725, 755)
    for i in range(2):
        for j in range(2):
            ax[i,j].tick_params(labelsize = lab_size)
            ax[i,j].annotate(l[i][j], (0.9, 0.9), fontweight = 'bold', fontsize = '20', xycoords='axes fraction')
            ax[i,j].set_yticks(ax[i,j].get_yticks()[1::2])

    t = ax[0,0].get_xticks()
    ax[0,0].set_xticks(t[1::2])

    fig.align_labels()
    
    plt.savefig(path, dpi = 200)
    

def get_max_jet_height(data, z, which_fit = 'both'):
    """
    Computes the maximum jet height (height of maximum wind speed) for each time step 
    based on different wind profile fitting methods.

    Parameters:
    ----------
    data : xarray.Dataset
        The dataset containing wind profile parameters and best-fit information.
    z : numpy.ndarray
        Array of height levels (altitudes) corresponding to the wind profile.
    which_fit : str, optional
        Specifies the fitting method to use. Options are:
        - 'log': Uses the logarithmic wind profile fit.
        - 'prd': Uses the Prandtl wind profile fit.
        - 'both': Uses the best-fit method specified in `data.best_fit_u`.
        Default is 'both'.

    Returns:
    -------
    numpy.ndarray
        An array of maximum jet heights for each time step, with NaN values where 
        a valid jet height could not be determined.

    Notes:
    ------
    - For each time step, the function determines the wind profile using either:
      - Logarithmic wind profile (`log_u` function)
      - Prandtl wind profile (`Prandtl_U` function)
      - The best-fit method provided in `data.best_fit_u` (if `which_fit='both'`).
    - The function finds the height (`zmax`) where the fitted wind speed is maximal.
    - If `zmax` is within the provided height range (`z[0] < zmax < z[-1]`), it is stored;
      otherwise, it remains NaN.
    - A plot of `max_heights` over time is generated for visualization.

    """
    
    
    fit  = data.best_fit_u.values
    max_heights = np.zeros(len(data.time))*np.nan
    for i in tqdm(range(len(fit))):
        if which_fit == 'log':
            a = data.aU[i].values
            b = data.bU[i].values
            c = data.cU[i].values
            u_fit = log_u(z, a, b, c)
            zmax = z[np.argmax(u_fit)]
            if z[0] < zmax < z[-1]:
                max_heights[i] = zmax
                
        if which_fit == 'prd':
            aL = data.aL_U[i].values
            aV = data.aV_U[i].values
            Km_min = data.Km_U[i].values
            u_fit = Prandtl_U(z, aL, aV, Km_min)
            zmax = z[np.argmax(u_fit)]
            if z[0] < zmax < z[-1]:
                max_heights[i] = zmax
                
        if which_fit == 'both':
            if fit[i] == 'prd':
                aL = data.aL_U[i].values
                aV = data.aV_U[i].values
                Km_min = data.Km_U[i].values
                u_fit = Prandtl_U(z, aL, aV, Km_min)
                zmax = z[np.argmax(u_fit)]
                if z[0] < zmax < z[-1]:
                    max_heights[i] = zmax
                    
            if fit[i] == 'log':
                a = data.aU[i].values
                b = data.bU[i].values
                c = data.cU[i].values
                u_fit = log_u(z, a, b, c)
                zmax = z[np.argmax(u_fit)]
                if z[0] < zmax < z[-1]:
                    max_heights[i] = zmax
        
    plt.figure()
    plt.plot(data.time, max_heights, '.')
    
    return max_heights
    

##################
### PARAMETERS ###
##################

Pr = 1  # Prandtl number, keep equal to 1
# 0.01 (s-1) what is N ref value ? It is not that important. Looking at the data, we can see that it is definitely > 0, and around sqrt(0.2) = 0.44
N_ref = 0.01
g = 9.81  # gravity acceleration (m/s2)
# slope in radians what is slope ? modify latter if it works.
alpha = 4*np.pi/180
R = 287  # perfect gas constant for air (J/K/kg)
# sensible surface heat flux (K.m/s). what are Fs values? between -0.05 and -0.01
Fs = -0.05
# surface temperature (K). Ts varies during night and days but roughly 273.15 K.
Ts = 273.15
# temperature gradient in the atmosphere (K/m) (reference so adiabatic).
dTz_ref = 0.01
z0 = 0.003  # surface roughness length

z = np.linspace(0, 5, 500)  # vertical grid for the fits

max_ev = 200  # number of calls to the fit function



dic = {'time': slice(datetime.datetime(2023, 8, 22, 0),
                     datetime.datetime(2023, 8, 22, 6))}

if __name__ == "__main__":

    ################
    ### SETTINGS ###
    ################

    #If set to True, computes all the fits and saves them in different files, it should only be used once
    first_time = False
    

    start_T1 = datetime.datetime(2023, 8, 19, 7, 0)
    end_T1 = datetime.datetime(2023, 9, 9, 10, 9)

    start_T2 = datetime.datetime(2023, 8, 17, 20, 00)
    end_T2 = datetime.datetime(2023, 9, 9, 18, 00)

    path = '/home/giordano/Work/ENS2/stage/'
    figpath = path + 'figures/tower_analysis/'

    Data_T1_orig = xr.open_dataset(
        path + 'data/HEFEXII/HEFEX2023_T303_DetrendDR_1min.nc')
    Data_T2_orig = xr.open_dataset(
        path + 'data/HEFEXII/HEFEX2023_T275_DetrendDR_1min.nc')
    
    snowfox_T1 = pd.read_csv(
        path + 'data/HEFEXII/T1_snowfox_lf.dat', skiprows=1).iloc[2:]
    snowfox_T2 = pd.read_csv(
        path + 'data/HEFEXII/T2_snowfox_lf.dat', skiprows=1).iloc[2:]

    
    ### Compute the prandtl fits at each time step (only once!)
    if first_time: 
        Data_T1, _, _ = clean_data(
            Data_T1_orig, levels=3, snowfox=snowfox_T1, start_time=pd.Timestamp(
                2023, 8, 17, 0), end_time=pd.Timestamp(2023, 9, 26, 0), qc_threshold=0.9)
        Data_T2, _, _ = clean_data(
            Data_T2_orig, levels=3, snowfox=snowfox_T2, start_time=pd.Timestamp(
                2023, 8, 17, 0), end_time=pd.Timestamp(2023, 9, 26, 0), qc_threshold=0.9)
        
        fit_prandtl_u(Data_T1, '/home/giordano/Work/ENS2/stage/results/tower_analysis/wind_gradients_T303.txt')
        fit_prandtl_u(Data_T2, '/home/giordano/Work/ENS2/stage/results/tower_analysis/wind_gradients_T275.txt')
        fit_prandtl_T(Data_T1, '/home/giordano/Work/ENS2/stage/results/tower_analysis/temperature_gradients_T303.txt')
        fit_prandtl_T(Data_T2, '/home/giordano/Work/ENS2/stage/results/tower_analysis/temperature_gradients_T275.txt')
        
        fit_log_u(Data_T1, '/home/giordano/Work/ENS2/stage/results/tower_analysis/wind_speed_log_T303.txt', path)
        fit_log_u(Data_T2, '/home/giordano/Work/ENS2/stage/results/tower_analysis/wind_speed_log_T275.txt', path)
        fit_log_T(Data_T1, '/home/giordano/Work/ENS2/stage/results/tower_analysis/temperature_log_T303.txt', path)
        fit_log_T(Data_T2, '/home/giordano/Work/ENS2/stage/results/tower_analysis/temperature_log_T275.txt', path)

        
        best_fit_u(Data_T1, '/home/giordano/Work/ENS2/stage/results/tower_analysis/best_fit_wind_T303.txt')
        best_fit_u(Data_T2, '/home/giordano/Work/ENS2/stage/results/tower_analysis/best_fit_wind_T275.txt')
        best_fit_T(Data_T1, '/home/giordano/Work/ENS2/stage/results/tower_analysis/best_fit_temp_T_surf_T303.txt')
        best_fit_T(Data_T2, '/home/giordano/Work/ENS2/stage/results/tower_analysis/best_fit_temp_T_surf_T275.txt')
    
    else:
        wind_grad_T1 = pd.read_csv(
            path + 'results/tower_analysis/wind_gradients_T303.txt', sep='\t').rename(columns = {'Unnamed: 0': 'Timestamp'})
        wind_grad_T2 = pd.read_csv(
            path + 'results/tower_analysis/wind_gradients_T275.txt', sep='\t').rename(columns = {'Unnamed: 0': 'Timestamp'})
        
        temp_grad_T1 = pd.read_csv(
            path + 'results/tower_analysis/temperature_gradients_T_surf_T303.txt', sep='\t').rename(columns = {'Unnamed: 0': 'Timestamp'})
        temp_grad_T2 = pd.read_csv(
            path + 'results/tower_analysis/temperature_gradients_T_surf_T275.txt', sep='\t').rename(columns = {'Unnamed: 0': 'Timestamp'})
        
        wind_log_T1 = pd.read_csv(
            path + 'results/tower_analysis/wind_speed_log_T303.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'})
        wind_log_T2 = pd.read_csv(
            path + 'results/tower_analysis/wind_speed_log_T275.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'})
    
        temp_log_T1 = pd.read_csv(
            path + 'results/tower_analysis/temperature_log_T_surf_T303.txt', sep='\t').rename(columns = {'Unnamed: 0': 'Timestamp'})
        temp_log_T2 = pd.read_csv(
            path + 'results/tower_analysis/temperature_log_T_surf_T275.txt', sep='\t').rename(columns = {'Unnamed: 0': 'Timestamp'})
            
        best_fits_wind_T1 = pd.read_csv(
           path + 'results/tower_analysis/best_fit_wind_T303.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'}).to_xarray().set_index(
               {'index': 'Timestamp'}).rename({'index': 'time', 'best_fit': 'best_fit_u'})
        best_fits_wind_T2 = pd.read_csv(
           path + 'results/tower_analysis/best_fit_wind_T275.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'}).to_xarray().set_index(
               {'index': 'Timestamp'}).rename({'index': 'time', 'best_fit': 'best_fit_u'})
        
        best_fits_temp_T1 = pd.read_csv(
           path + 'results/tower_analysis/best_fit_temp_T_surf_T303.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'}).to_xarray().set_index(
               {'index': 'Timestamp'}).rename({'index': 'time', 'best_fit': 'best_fit_T'})
        best_fits_temp_T2 = pd.read_csv(
           path + 'results/tower_analysis/best_fit_temp_T_surf_T275.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'}).to_xarray().set_index(
               {'index': 'Timestamp'}).rename({'index': 'time', 'best_fit': 'best_fit_T'})
        
        turb_max_height_T1 = pd.read_csv(
            '/home/giordano/Work/ENS2/stage/results/tower_analysis/turbulent_max_jet_height_T303.txt', sep='\t')
        
        turb_max_height_T2 = pd.read_csv(
            '/home/giordano/Work/ENS2/stage/results/tower_analysis/turbulent_max_jet_height_T275.txt', sep='\t')
                
        fit_max_heights_T1 = pd.read_csv(
            '/home/giordano/Work/ENS2/stage/results/tower_analysis/max_jet_heights_T303.txt')
        
        fit_max_heights_T2 = pd.read_csv(
            '/home/giordano/Work/ENS2/stage/results/tower_analysis/max_jet_heights_T275.txt')
        
        
        Data_T1, wind_grad_T1, turb_max_height_T1, wind_log_T1, temp_log_T1 = clean_data(
            Data_T1_orig, levels=3, snowfox=snowfox_T1, wind_grad=wind_grad_T1, turb_max_height=turb_max_height_T1, temp_grad=temp_grad_T1, wind_log=wind_log_T1, temp_log=temp_log_T1,
            start_time=pd.Timestamp(
                2023, 8, 17, 0), end_time=pd.Timestamp(2023, 9, 26, 0), qc_threshold=0.9, fill_gaps=False)
        Data_T2, wind_grad_T2, turb_max_height_T2, wind_log_T2, temp_log_T2 = clean_data(
            Data_T2_orig, levels=3, snowfox=snowfox_T2, wind_grad=wind_grad_T2, turb_max_height=turb_max_height_T2, temp_grad=temp_grad_T2, wind_log=wind_log_T2, temp_log=temp_log_T2,
            start_time=pd.Timestamp(
                2023, 8, 17, 0), end_time=pd.Timestamp(2023, 9, 26, 0), qc_threshold=0.9, fill_gaps=False)
        Data_K = xr.open_dataset('/home/giordano/Work/ENS2/stage/data/HEFEXII/HEFEX2023_T275_Kaijo_DetrendDR_1min.mat.nc')
    
        Data_T1_orig['temp'] = Data_T1_orig.meanT
        Data_T2_orig['temp'] = Data_T2_orig.meanT
        
        best_fits_temp_T1['time'] = pd.to_datetime( best_fits_temp_T1['time'])
        best_fits_temp_T2['time'] = pd.to_datetime( best_fits_temp_T2['time'])
        best_fits_wind_T1['time'] = pd.to_datetime( best_fits_wind_T1['time'])
        best_fits_wind_T2['time'] = pd.to_datetime( best_fits_wind_T2['time'])
        
        Data_T1['best_fit_u'] = best_fits_wind_T1.best_fit_u
        Data_T2['best_fit_u'] = best_fits_wind_T2.best_fit_u
        
        Data_T1['fit_log_u_rmse'] = best_fits_wind_T1.log_rmse
        Data_T1['fit_prd_u_rmse'] = best_fits_wind_T1.prandtl_rmse
        Data_T2['fit_log_u_rmse'] = best_fits_wind_T2.log_rmse
        Data_T2['fit_prd_u_rmse'] = best_fits_wind_T2.prandtl_rmse
    
        Data_T1['best_fit_T'] = best_fits_temp_T1.best_fit_T
        Data_T2['best_fit_T'] = best_fits_temp_T2.best_fit_T
        
        Data_T1['fit_log_T_rmse'] = best_fits_temp_T1.log_rmse
        Data_T1['fit_prd_T_rmse'] = best_fits_temp_T1.prandtl_rmse
        Data_T2['fit_log_T_rmse'] = best_fits_temp_T2.log_rmse
        Data_T2['fit_prd_T_rmse'] = best_fits_temp_T2.prandtl_rmse
        
        variables = list(Data_T1.keys())
        
        n_colors = 6
        c_list = [mpl.colormaps['copper'].resampled(n_colors)(i) for i in range(n_colors)]
        
        color_stations = {'T275':'#8DA279','T303':'#497784'}
    
    
        ###################
        ### ACTUAL CODE ###
        ###################
    
        katabatic_periods = [slice(datetime.datetime(2023, 8, 19, 20, i), datetime.datetime(
            2023, 8, 19, 21, i)) for i in range(0, 59, 10)]
        
        data_katabatic_T1 = [Data_T1.sel(time=katabatic_periods[i])
                             for i in range(len(katabatic_periods))]
        data_katabatic_T2 = [Data_T2.sel(time=katabatic_periods[i])
                             for i in range(len(katabatic_periods))]
        
        #fig, ax, ufit, tfit = plot_fit_Prandtl(data_katabatic_T1, data_katabatic_T2, z, c_list = c_list)
        #plt.savefig(figpath + 'prandtl_fit_4.png')
        
        
    
        #daily_T1 = Data_T1.groupby(Data_T1['time.time']).mean(dim='time')
        #daily_T2 = Data_T2.groupby(Data_T2['time.time']).mean(dim='time')
        #rolling_T1 = daily_T1.rolling(time = 60, center= True).mean()
        #rolling_T2 = daily_T2.rolling(time = 60, center= True).mean()
        #plt.figure()
        #plt.plot(np.linspace(0,24,1440), rolling_T1.meanU)
        #plt.figure()
        #plt.plot(np.linspace(0,24,1440), rolling_T2.meanU).
    
        #plot_timeseries(Data_T2, 'meanQirga', 2)
    
    
        #anim = animate_katabatic_prandtl(Data_T2, (datetime.datetime(2023,8,20,16,00), datetime.datetime(2023,8,21,00,00)), z, 60, 2)
        #anim[0].save(figpath + 'ani_prandtl_T2_1min_filled.gif', fps = 10)
        #anim[0].save(figpath + 'ani_prandtl_T2_1min_filled.mp4', fps = 10)
        
       
        #anim = animate_katabatic_prandtl(Data_T2, (datetime.datetime(2023,8,20,16,00), datetime.datetime(2023,8,21,16,00)), z, 600, 2)
        #anim[0].save(figpath + 'ani_prandtl_T2_10min.gif', fps = 5)
        #anim[0].save(figpath + 'ani_prandtl_T2_10min.mp4', fps = 5)
    
    
        #anim2 = animate_katabatic_prandtl(Data_T2, (datetime.datetime(2023,8,19,17,20), datetime.datetime(2023,8,20,00)), z, 600, 2)
        #anim2[0].save(figpath + 'ani_prandtl_T2_10min.gif')
    
       
        #hb = surface_energy_balance(Data_T2, True)
        
        
        ###Compare Prandtl fits with 3 or 4 levels
        '''
        start = pd.Timestamp(2023, 8, 22, 8,0)
        c_list = ['C{}'.format(i) for i in range(10)]
        DT = [Data_T2.sel(time = slice(start + np.timedelta64(600, 's') * i, start + np.timedelta64(600, 's') * (i+1))).mean(dim='time') for i in range(10)]
        DK = [Data_K.sel(time = slice(start + np.timedelta64(600, 's') * i, start + np.timedelta64(600, 's') * (i+1))).mean(dim='time') for i in range(10)]
        fig = plt.figure(figsize = (12,8), dpi = 100)
        for i in range(8):
            h = np.concatenate([[DK[i].heights.values], DT[i].heights.values])
            u = np.concatenate([[DK[i].meanU.values], DT[i].meanU.values])
            [aL, aV, Km_min], _ = curve_fit(Prandtl_U, h, u, bounds=(0, np.inf), maxfev=max_ev)
            ufit = Prandtl_U(z, aL, aV, Km_min)
            plt.plot(ufit, z, c = c_list[i])
            plt.scatter(u, h, c = c_list[i])
            h2 = DT[i].heights.values
            u2 = DT[i].meanU.values
            [aL2, aV2, Km_min2], _ = curve_fit(Prandtl_U, h2, u2, bounds=(0, np.inf), maxfev=max_ev)
            ufit = Prandtl_U(z, aL2, aV2, Km_min2)
            plt.plot(ufit, z, ':', c = c_list[i])
        plt.xlabel('U (m/s)')
        plt.ylabel('z (m)')
        plt.plot([], [], '-k', label = 'Fit with 4 levels')
        plt.plot([], [], ':k', label = 'Fit with 3 levels')
        plt.legend()
        plt.tight_layout()'''
        
        '''
        ###Compare Prandtl fits with 3 or 4 levels, a bit more toroughly
        d1 = Data_T2.time
        d2 = Data_K.time
        d1.name = 'time_T'
        d2.name = 'time_K'
        time = xr.merge([d1, d2], join = 'inner').time
        Data_T2_n = Data_T2.sel(time = time)
        u_list4l = []
        u_list3l = []
        h_list4l = []
        h_list3l = []
        max_z4l = []
        max_z3l = []
        time_4l = []
        time_3l = []
        for i,t in tqdm(enumerate(time.values)):
            DT = Data_T2_n.sel(time = t)
            DK = Data_K.sel(time = t)
            h4 = np.concatenate([[DK.heights.values], DT.heights.values])
            u4 = np.concatenate([[DK.meanU.values], DT.meanU.values])
            h3 = DT.heights.values
            u3 = DT.meanU.values
            try:
                [aL, aV, Km_min], _ = curve_fit(Prandtl_U, h4, u4)
                ufit4 = Prandtl_U(z, aL, aV, Km_min)
                z4 = z[np.argmax(ufit4)]
                if z4<4:
                    u_list4l.append(ufit4)
                    h_list4l.append(h4)
                    max_z4l.append(z4)
                    time_4l.append(t)
            except (RuntimeError, TypeError, ValueError) as e:
                logging.debug(...)
            continue 
            try:
                [aL, aV, Km_min], _ = curve_fit(Prandtl_U, h3, u3)
                ufit3 = Prandtl_U(z, aL, aV, Km_min)
                z3 = z[np.argmax(ufit3)]
                if z3 < 4:
                    u_list3l.append(ufit3)
                    h_list3l.append(h3)
                    max_z3l.append(z3)
                    time_3l.append(t)
            except (RuntimeError, TypeError, ValueError) as e:
                logging.debug(...)
            continue
        
        x = Data_T2_n.zmax_uw[:,0].sel(time = time_4l)
        y3 = Data_T2_n.max_jet_height[:,0].sel(time = time_4l)
        y4 = np.array(max_z4l)
        yT = Data_T2_n.zmax_uT[:,0].sel(time = time_4l)
        idx3 = x.notnull() & y3.notnull()
        idx4 = x.notnull() & ~np.isnan(y4)
        idxT = x.notnull() & yT.notnull()
        a3, b3 = np.polyfit(x[idx3], y3[idx3], 1)
        a4, b4 = np.polyfit(x[idx4], y4[idx4], 1)
        aT, bT = np.polyfit(x[idxT], y4[idxT], 1)
        dev3 = np.nanmedian(np.abs(y3 - x*a3 + b3))
        dev4 = np.nanmedian(np.abs(y4 - x*a4 + b4))
        devT = np.nanmedian(np.abs(yT - x*aT + bT))
        plt.figure(figsize=(12,12), dpi = 100)
        plt.scatter(x, y3, label = '3 level fit, median deviation = {}'.format(round(dev3, 2)), s = 50)
        plt.scatter(x, y4, label = '4 level fit, median deviation = {}'.format(round(dev4, 2)), s = 50)
        plt.scatter(x, yT, label = '$\overline{u\'w\'}$' +  ', median deviation = {}'.format(round(devT, 2)), s=50)
        plt.plot(x, x, '-k', label = '1:1')
        plt.plot(x, a3*x+b3, 'C0')
        plt.plot(x, a4*x+b4, 'C1')
        plt.plot(x, aT*x+bT, 'C2')
        plt.xlabel('Max jet height, $\overline{u\'w\'}$ (m)')
        plt.ylabel('Max jet height (m)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/giordano/Work/ENS2/stage/figures/tower_analysis/compare_3_4_fits_uw.png')
        '''
        
        '''
        ### Compute max jet heights for the 2 models and the best fit, writing rmse
        max_heights_T1_prd = get_max_jet_height(Data_T1, z, 'prd')
        max_heights_T1_log = get_max_jet_height(Data_T1, z, 'log')
        max_heights_T1_both = get_max_jet_height(Data_T1, z, 'both')
        da = xr.Dataset(data_vars = dict(max_jet_log=(["time"], max_heights_T1_log),
                                       max_jet_prd=(["time"], max_heights_T1_prd), 
                                       max_jet_best=(["time"], max_heights_T1_both),
                                       log_rmse=(["time"], Data_T1.fit_log_u_rmse.data), 
                                       prd_rmse=(["time"], Data_T1.fit_prd_u_rmse.data)),
                        coords = dict(time=Data_T1.time)).to_dataframe()
        da.to_csv('/home/giordano/Work/ENS2/stage/results/tower_analysis/max_jet_heights_T303.txt')
        
        max_heights_T2_prd = get_max_jet_height(Data_T2, z, 'prd')
        max_heights_T2_log = get_max_jet_height(Data_T2, z, 'log')
        max_heights_T2_both = get_max_jet_height(Data_T2, z, 'both')
        da = xr.Dataset(data_vars = dict(max_jet_log=(["time"], max_heights_T2_log),
                                       max_jet_prd=(["time"], max_heights_T2_prd), 
                                       max_jet_best=(["time"], max_heights_T2_both),
                                       log_rmse=(["time"], Data_T2.fit_log_u_rmse.data), 
                                       prd_rmse=(["time"], Data_T2.fit_prd_u_rmse.data)),
                        coords = dict(time=Data_T2.time)).to_dataframe()
        da.to_csv('/home/giordano/Work/ENS2/stage/results/tower_analysis/max_jet_heights_T275.txt')
        '''
        
        plt.show()
        
