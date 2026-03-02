#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:07:48 2024

@author: giordano
"""

######################
### IMPORT MODULES ###
######################

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import xarray as xr
import scipy.cluster.hierarchy as shc
import datetime
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl

import tower_analysis as ta
import dendro


# Some styling changes
from pylab import rcParams
plt.style.use('default')

rcParams['axes.labelweight'] = 'bold'
rcParams['axes.labelsize'] = 'x-large'
rcParams['axes.titlesize'] = 'xx-large'
rcParams['axes.titleweight'] = 'bold'

plt.close('all')

warnings.filterwarnings("default")

########################
### DEFINE FUNCTIONS ###
########################


def normalize(data, cluster_variables, independent=True, n_lev=3):
    """
    Normalize the data to zero mean and unit standard deviation.

    Parameters
    ----------
    data : 2-d array
        shape (length of the data in time, n_levels*n_variables)
    cluster_variables : list of str
        The names of the variables to normalize.
    independent : l, optional
        Independent normaization or not. The default is True.
    n_lev : int, optional
        The number of levels of the data. The default is 3.

    Returns
    -------
    data : 2-d array
        The normalized data, shape (length of the data in time, n_levels*n_variables).
    stds : 1-d array
        The standard deviations of the data, shape (n_levels*n_variables).
    means : 1-d array
        The means of the data, shape (n_levels*n_variables).

    """

    # Where to use circular mean and standard deviation ('dir')
    pos_dir = []
    for v in cluster_variables:
        if v[:3] == 'dir':
            pos_dir.append(cluster_variables.index(v))

    # Normalize strategically.
    # Here, independent normalization is kept
    # Independent normalization is an option to be played with, but (I (Cole) found) lead to less interpretable results

    if not independent:
        means = np.array([[np.mean(data[:, i*n_lev:i*n_lev+n_lev])]
                         * n_lev for i in range(len(cluster_variables))]).flatten()

        stds = np.array([[np.std(data[:, i*n_lev:i*n_lev+n_lev])]
                        * n_lev for i in range(len(cluster_variables))]).flatten()

    if independent:
        #Independent normalization

        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

    for i in range(len(data[0])):
        if stds[i] != 0:
            data[:, i] = (data[:, i] - means[i])/stds[i]
        else:
            data[:, i] = data[:, i] - means[i]

    return data, stds, means


def pca(cluster_variables, data, n_eigs=4, no_plot=True, independent_normalization=True, levels_u = [1,2,3], levels_T=[1,2,3]):
    """
    Computes principal component analysis (PCA) on tower data.

    Parameters
    ----------
    cluster_variables : list
        names of the variables to select
    data : dataset
    n_eigs : int, optional
        Number of eigenvectors to compute. The default is 4.
    no_plot : bool, optional
        If True, no plot is displayed. The default is True.
    independent_normalization : bool, optional
        Independent normaization or not. The default is True.. The default is True.
    levels_u : list, optional
        Levels for the wind speed measurements (the lenght is used in the caclulation, the values for plotting).
        The default is [1,2,3].
    levels_T : list, optional
        Levels for the temperature measurements (the lenght is used in the caclulation, the values for plotting).
        The default is [1,2,3].

    Returns
    -------
    dat_orig : dataset
        The original data.
    dat_reduc : 2-d array
        The original data, projectted onto th EOFs. 
        The shape is (n_variables, n_levels).
    eofs : nd-array
        The non-dimensional Empirical Orthogonal Functions (EOFs). 
        The shape is (n_eigs, n_variables, n_levels).
    dim_eofs : nd-array
        The dimensional Empirical Orthogonal Functions (EOFs). 
        The shape is (n_eigs, n_variables, n_levels).
    pcr : 1d-array
        The explained variance ratio of the eofs
    X : 2d-array
        The matrix such as X . dim_eofs is a reduction of data_orig. The shape is (n_observations, n_eigs)


    """

    
    n_lev = len(levels_u)

    # load in data
    D = data[cluster_variables]

    # Read out columns
    dat_orig = []
    for i in range(len(D)):
        dat_orig.append(D[cluster_variables[i]].values.T)
    dat_orig = np.array(dat_orig)

    dat_orig = np.reshape(dat_orig.flatten(),
                          (n_lev*len(D), len(dat_orig[0, 0]))).T

    dat_array = np.copy(dat_orig)

    # Normalize data
    dat_array, stds, means = normalize(dat_array, cluster_variables, independent = independent_normalization)

    # pca.fit WILL further normalize (as PCA needs data with 0 mean)
    pca = PCA(n_components=n_eigs)
    X = pca.fit_transform(dat_array)
    pcr = pca.explained_variance_ratio_

    eofs = pca.components_
    dim_eofs = eofs * stds + means

    # Dimensionality reduction

    dat_reduc = (np.dot(X[:, :], eofs[:])[:, :])*stds[:] + means[:]
    print('Dimensionality reduction with {} % of explained variance'.format(
        round(np.sum(pcr)*100, 2)))

    eofs = np.reshape(eofs, (n_eigs, len(D), n_lev))
    dim_eofs = np.reshape(dim_eofs, (n_eigs, len(D), n_lev))
    
    add_zero_U  = 0

    if not no_plot:

        fig, axes = plt.subplots(2, n_eigs, figsize=(15, 10), dpi=100)

        for i in range(n_eigs):

            ax = axes[0, i]

            ax.set_ylim(0, 5)
            if np.isnan(add_zero_U):
                ax.set_xlim(0.7*np.min(dim_eofs[:, cluster_variables.index('meanU')]), 1.1*np.max(
                    dim_eofs[:, cluster_variables.index('meanU')]))
            else:
                ax.set_xlim(
                    0, 1.2*np.max(dim_eofs[:, cluster_variables.index('meanU')]))

            ax.plot(np.concatenate(
                ([add_zero_U], dim_eofs[i, cluster_variables.index('meanU')])), [0] + levels_u, ':.b')
            ax.set_xlabel('Wind speed (m/s)', color='blue')
            if i == 0:
                ax.set_ylabel('Height (m)')
            
            if 'dir' in cluster_variables:
                ax.barbs([ax.get_xlim()[0]*0.1 + ax.get_xlim()[1]*0.9]*n_lev, levels_u,
                         np.sin(np.deg2rad(
                             dim_eofs[i, cluster_variables.index('dir')]) + np.pi),
                         np.cos(np.deg2rad(
                             dim_eofs[i, cluster_variables.index('dir')]) + np.pi),
                         barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=6)
            
            elif 'easting' in cluster_variables and 'northing' in cluster_variables:
                ax.barbs([ax.get_xlim()[0]*0.1 + ax.get_xlim()[1]*0.9]*n_lev, levels_u,
                             dim_eofs[i, -cluster_variables.index('easting')],
                         dim_eofs[i, -cluster_variables.index('northing')],
                         barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=6)

            ax.scatter([ax.get_xlim()[0]*0.1 + ax.get_xlim()
                       [1]*0.9]*n_lev, [1, 2, 4], c='k', s=15)

            ax.set_title('EOF {} ({} %)'.format(i+1, round(100*pcr[i], 1)))

        for i in range(n_eigs):

            ax = axes[1, i]

            ax.set_ylim(0, 5)
            ax.set_xlim(np.min(dim_eofs[:, cluster_variables.index(
                    temperature)] - 2), np.max(dim_eofs[:, cluster_variables.index(temperature)] + 5))
            #else:
            #    ax.set_xlim(
            #       add_zero_T + 273.15, np.max(dim_eofs[:, cluster_variables.index(temperature)]) + 5)

            ax.plot(np.concatenate(([np.nan], dim_eofs[i, cluster_variables.index(
                temperature)])), [0] +  levels_T, ':.g')
            ax.set_xlabel('Temperature (K)', color='green')
            if i == 0:
                ax.set_ylabel('Height (m)')

        plt.tight_layout()

    else:
        fig, axes = None, None

    return dat_orig, dat_reduc, eofs, dim_eofs, pcr, X


    
def clustering(cluster_variables, data_0, tower, c_list, n_clusters=5, dim_reduction=-1, no_plot=False, independent_normalization=True, levels_u = [1,2,3], levels_T=[1,2,3], lab_size=15, fig1_size = (30,15), save_fig_profiles='', save_fig_variables='', save_fig_daily='', save_fig_pca='', save_fig_dendro='', save_fig_eofs_var='', save_fig_prandtl_profiles='', bams = False):
    """
    Performs hierarchial linking clustering on selected variables in 
    a given dataset, with different measurements levels.

    Parameters
    ----------
    cluster_variables : list
        names of the variables to select
    data_0 : dataset
        Already selected the interesting periods.
    tower : int 
        identifier of the tower, used for temperature sensors height which dffer from T1 to T2
    c_list : list
        List of colors to use for plotting
    n_clusters : int, optional
        Number of clusters to find. The default is 5.
    dim_reduction : int, optional
        Number of EOFs to keep for dimensionality reduction. Default is -1, which corresponds to keeping all the EOFs (as many as len(cluster_variables) * n_levels)
    no_plot : bool, optional
        If True, no figure will be displayed. The default is False.
    independent_normalization : bool, optional
        Independent normaization or not. The default is True.. The default is True.
    levels_u : list, optional
        Levels for the wind speed measurements (the lenght is used in the caclulation, the values for plotting).
        The default is [1,2,3].
    levels_T : list, optional
        Levels for the temperature measurements (the lenght is used in the caclulation, the values for plotting).
        The default is [1,2,3].
    lab_size : float, optional
        Label size on the figures. The default is 15.
    fig1_size : tuple, optional
        Size of the clusters profiles figure. The default is (30,15).
    save_fig_profiles : str, optional
        Path for saving the profiles figure. The default is ''. In this case, the figure is not saved.
    save_fig_variables : str, optional
        Path for saving the variables timeseries figure. The default is ''. In this case, the figure is not saved.
    save_fig_daily : str, optional
        Path for saving the clusters daily cycle figure. The default is ''. In this case, the figure is not saved.
    save_fig_pca : str, optional
        Path for saving the eofs profiles figure. The default is ''. In this case, the figure is not saved.
    save_fig_dendro : str, optional
        Path for saving the dendrogram figure. The default is ''. In this case, the figure is not saved.
    save_fig_eofs_var : str, optional
        Path for saving the eofs variables figure. The default is ''. In this case, the figure is not saved.
    save_fig_prandtl_profiles : str, optional
        Path for saving the clusters profiles (with Prandtl models) figure. The default is ''. In this case, the figure is not saved.
    bams : bool, optional
        If Ttrue, some clusters are grouped with one another to recreate the
        figures from the bams article. In this case, the number of clusters
        should be set to 5. The default is False.

    Returns
    -------
    data : dataset 
        Modified dataset which includes the cluster each data point is located in.
    clus_ids : 1d-array
        Contains the identifiers of the cluster in which each measurement is located. The shape is (n_observations)
    clus_values : list
        Each element is a dataset containing the meadian value of each variable for a given cluster, in order of significance.
    clus_q1 : list
        Each element is a dataset containing the first quartile deviation of each variable for a given cluster, in order of significance.
    clus_q3 : list
        Each element is a dataset containing the third quartile deviation of each variable for a given cluster, in order of significance.
    X : 2d-array
        The matrix such as X . dim_eofs is a reduction of data_orig, on which the clustering is performed. The shape is (n_observations, n_eigs).


    """
    
    n_lev = len(levels_u)

    data = data_0.dropna(dim='time', subset=cluster_variables)

    c_list = c_list[:n_clusters]

    if dim_reduction == -1:
        dim_reduction = len(cluster_variables)*n_lev

    # Load in data
    D = data[cluster_variables]

    # Compute dimensionality reduction by PCA
    dat_orig, dat_reduc, eofs, dim_eofs, pcr, X = pca(
        cluster_variables, D, n_eigs=dim_reduction, no_plot=no_plot, independent_normalization=independent_normalization, levels_u=levels_u, levels_T=levels_T)
    if save_fig_pca != '':
        plt.savefig(save_fig_pca, dpi=200)

    # Plot EOFs variables
    if not no_plot and dim_reduction>0:
        lab_list = []
        for k in cluster_variables:
            lab_list.append(k + ' 1 m')
            lab_list.append(k + ' 2 m')
            lab_list.append(k + ' 4 m')
        fig, ax = plt.subplots(2, dim_reduction//2 + dim_reduction%2, figsize=(10,10), sharex = True, layout = 'constrained')
        for i in range(dim_reduction):
            ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].set_ylim(np.min(eofs)*1.1, np.max(eofs)*1.1)
            ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].plot(lab_list, eofs[i].flatten(), 's', c = 'k')
            ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].plot(lab_list, [0]*len(lab_list), ':', c = 'k')
            ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].set_title('EOF {} ({} %)'.format(i+1, int(pcr[i]*100)))
            ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].tick_params(axis = 'x', rotation = 90)
            l = ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].get_ylim()
            ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].bar(list(range(len(cluster_variables)*n_lev)), [l[1]-l[0]]*len(cluster_variables)*n_lev, bottom = l[0], width = 1, color = np.array(['b', 'white'])[np.array([(i//n_lev)%2 for i in range(len(cluster_variables)*n_lev)])], alpha = 0.2)
            ax[int(i >= dim_reduction//2 + dim_reduction%2), i%(dim_reduction//2 + dim_reduction%2)].set_ylim(l)
        for i in range(dim_reduction//2, dim_reduction//2 + dim_reduction%2):
            ax[1, i%(dim_reduction//2 + dim_reduction%2)].remove()
        if save_fig_eofs_var != '':
            plt.savefig(save_fig_eofs_var, dpi = 200)
    
    # Get data from dimensionality reduction
    dat_array = np.copy(dat_reduc)
    
    # Normalize data
    dat_array, stds, means = normalize(dat_array, cluster_variables, independent = independent_normalization)

    print('\nClustering')

    Z = shc.linkage(X, 'ward')
    clus = shc.fcluster(Z, n_clusters, criterion='maxclust')-1

    data = data.assign(cluster=(['time'], clus))

    unsorted_clus_values = []  # 9 parameters
    unsorted_clus_q1 = []
    unsorted_clus_q3 = []

    clus_freqs = np.zeros(n_clusters)

    print('Sorting clusters')

    if bams: 
        ###Messing up manually for the BAMS figures (I know it's bad)
        data['cluster'][data['cluster'] == 1] = 0
        data['cluster'][data['cluster'] == 2] = 2
        data['cluster'][data['cluster'] == 3] = 1
        data['cluster'][data['cluster'] == 4] = 1


    
        data['cluster']  = data['cluster'].astype('float')
        data['cluster'][data['cluster']<0] = np.nan
        data = data.dropna(dim='time', subset = ['cluster'])
        clus = data['cluster'].values
        n_clusters = len(np.unique(clus))
        # calculate cluster frequencies
        clus_freqs = np.zeros(n_clusters)
        for i in range(n_clusters):
            clus_freqs[i] = np.sum(clus == i)/len(clus)
        sf = np.argsort(-clus_freqs)  # get the ordering from the frequencies
        
        c_list[2:4] = c_list[3:1:-1] # change order of colors

    clus_values = []
    clus_q1 = []
    clus_q3 = []
    
    for i in range(n_clusters):
        
        # Calculate median and quantiles for each cluster
        variables = list(data_0.keys())
        variables.remove('best_fit_u')
        variables.remove('best_fit_T')
        unsorted_clus_values.append(data_0[variables].where(data.cluster == i).median(
            dim='time', skipna=True))  # select by cluster values
        unsorted_clus_q1.append(data_0[variables].where(
            data.cluster == i).quantile(0.75, dim='time', skipna=True))
        unsorted_clus_q3.append(data_0[variables].where(
            data.cluster == i).quantile(0.25, dim='time', skipna=True))
            
        # calculate cluster frequencies
        clus_freqs[i] = np.sum(clus == i)/len(clus)

    sf = np.argsort(-clus_freqs)  # get the ordering from the frequencies
    data = data.assign(cluster=(['time'], np.array([np.where(data.cluster.values[i] == sf)[
                       0][0] for i in range(len(clus))])))  # reassign clusters

    for i in np.arange(n_clusters)[sf]:
        # reassign ordered cluster values
        clus_values.append(unsorted_clus_values[i])
        clus_q1.append(unsorted_clus_q1[i])
        clus_q3.append(unsorted_clus_q3[i])
    
    '''
    ###Now the temperature profiles got mixed up??
    t = clus_values[2][temperature]
    tlow = clus_q1[2][temperature]
    thigh = clus_q3[2][temperature]
    clus_values[1]['dir'] = clus_values[1]['dir']
    clus_q1[2]['dir'] = clus_q1[1]['dir']
    clus_q3[2]['dir'] = clus_q3[1]['dir']
    clus_values[1]['dir'] = d
    clus_q1[1]['dir'] = dlow
    clus_q3[1]['dir'] = dhigh
    '''
        
    '''
    ###Switching the directions from two clusters because they got mixed up (how?)
    d = clus_values[2]['dir']
    dlow = clus_q1[2]['dir']
    dhigh = clus_q3[2]['dir']
    clus_values[2]['dir'] = clus_values[1]['dir']
    clus_q1[2]['dir'] = clus_q1[1]['dir']
    clus_q3[2]['dir'] = clus_q3[1]['dir']
    clus_values[1]['dir'] = d
    clus_q1[1]['dir'] = dlow
    clus_q3[1]['dir'] = dhigh
    '''

    clus_freqs = np.sort(clus_freqs)[::-1]

    sil_scores = []
    
    print('Clusters silhouette scores')
    for i in range(n_clusters):
        score = round(100*silhouette_score(dat_array, clus == i), 2)
        sil_scores.append(score)
        print('{} {} %'.format(i+1, score))
        
    # Plot dendrogram
    if not no_plot:
        fig, ax = plt.subplots(figsize=(10,6))
        ax, clus = dendro.improved_dendrogram(ax, X, n_clusters, c_list = c_list, sc = sil_scores)
        if save_fig_dendro != '':
            plt.savefig(save_fig_dendro, dpi = 200)
        
    
    occurences = [0]
    times = np.linspace(0, 24, 1440)

    # Counting daily occurences
    print('\nCounting daily occurences\n')
    occurences = []
    
    for i in range(n_clusters):
        
        sub = ''
        for v in list(data.variables):
            if v[:5] == 'meanU':
                sub = v
        
        if 'tod' in data.variables:
            day_t = data.where(data.cluster == i).dropna(dim='time', subset = [sub]).tod
            
            hist = np.histogram(day_t, bins = 144, density = False) # 10-minutes bins
            x_t = hist[1][1:]
            y_t = (hist[0]/len(day_t)) # 5 hours rolling average
            
            window_in_hours = 6 # change this value if needed
            win = int(window_in_hours*6) #one point every 10 min gives 6 points per hour
            
        
            ds = xr.Dataset(
                data_vars=dict(val=(["tod"], y_t)),
                coords=dict(tod=x_t))
            ds = ds.pad(tod=win, mode='wrap').rolling(tod=win, center=True).mean()
            
            occurences.append((ds.tod[win:-win], ds.val[win:-win]))
        

    if not no_plot:
        
        wind_fits = data[['best_fit_u', 'aU', 'bU',
                          'cU', 'aL_U', 'aV_U', 'Km_U']].copy()
        
        temp_fits = data[['best_fit_T', 'aT', 'bT',
                          'cT', 'aL_T', 'aT_T', 'Km_T']].copy()
        
        temperature = 'temp'
        add_zero_T = 0

        print('Plotting\n')

        # Axes limits
        maxT = np.min((15, np.nanquantile(data_0[temperature].max(dim = 'height_coords'), 0.99)))
        #minT = np.nanquantile(data_0[temperature].min(dim = 'height_coords'), 0.1)
        #minT = np.min((-3, np.nanquantile(data_0[temperature].min(dim = 'height_coords'), 0.02)))
        minT = np.min((-2, np.min(data_0[temperature])))
        maxU = np.nanquantile(data_0['meanU'].max(dim = 'height_coords'), 0.95)
        maxC = np.nanmax([np.nanmax(o) for o in occurences])
    
        # Profiles figure
        fig, axes = plt.subplots(2, 1, figsize=fig1_size, dpi=100, sharex='row')
        for j in range(2):
            axes[j].spines[['top', 'left', 'right', 'bottom']].set_visible(False)
            axes[j].set_xticks([])
            axes[j].set_yticks([])
        #lab = 'Temperature difference \n with first level (°C)'
        axes[0].set_xlabel('Wind speed (m s$^{-1}$)', size = lab_size)
        lab = 'Temperature (°C)'            
        axes[1].set_xlabel(lab, size = lab_size)
        
        wind_fits = wind_fits.where(wind_fits.time.isin(data.time), drop=True)

        for i in range(n_clusters):
            
            ax = axes[0].inset_axes([0.01 + i*(1/n_clusters), 0.05, (1/n_clusters) - 0.01, 0.95])

            ax.set_ylim(0, 4.5)
            ax.set_xlim(0, 1.2*maxU)
            
            ax.plot(np.concatenate(([0], clus_values[i]['meanU'])), np.concatenate(([0],levels_u)), c=c_list[i])
            ax.fill_betweenx([0, 1, 2, 4], np.concatenate(([0], clus_q1[i]['meanU'])),
                             np.concatenate(
                                 ([0], clus_q3[i]['meanU'])),
                             color=c_list[i], alpha=0.5)

            ax.barbs([ax.get_xlim()[0]*0.1 + ax.get_xlim()[1]*0.9]*n_lev, levels_u, np.sin(np.deg2rad(clus_values[i]['dir']) + np.pi),
                     np.cos(np.deg2rad(clus_values[i]['dir']) + np.pi), barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=10, linewidth = 4, zorder=6)
            ax.barbs([ax.get_xlim()[0]*0.15 + ax.get_xlim()[1]*0.9]*n_lev, levels_u, np.sin(np.deg2rad(clus_q1[i]['dir']) + np.pi),
                     np.cos(np.deg2rad(clus_q1[i]['dir']) + np.pi), barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=9, linewidth = 4, alpha=0.6, zorder=6)
            ax.barbs([ax.get_xlim()[0]*0.1 + ax.get_xlim()[1]*0.9]*n_lev, levels_u, np.sin(np.deg2rad(clus_q3[i]['dir']) + np.pi),
                     np.cos(np.deg2rad(clus_q3[i]['dir']) + np.pi), barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=9, linewidth = 4, alpha=0.6, zorder=6)

            ax.scatter([ax.get_xlim()[0]*0.1 + ax.get_xlim()
                       [1]*0.9]*n_lev, levels_u, c='k', s=15, zorder=6)

            ax.set_title('Cluster {} ({} %)'.format(
                i+1, round(100*clus_freqs[i], 1)), size = lab_size)
            
            if i == 0:
                ax.set_ylabel('Height (m)', size = lab_size)
                ax.set_yticks(levels_u)
            if i > 0:
                ax.set_yticks([])
            ax.tick_params(labelsize = lab_size)

        
        for i in range(n_clusters):
            
            ax = axes[1].inset_axes([0.01 + i*(1/n_clusters), 0.05, (1/n_clusters) - 0.01, 0.95])
            
            ax.set_ylim(0, 5)
            #ax.set_xlim(np.nanmin([minT-2, add_zero_T-0.5]), maxT + 2)
            ax.set_xlim(minT, maxT)

            ax.plot(np.concatenate(
                ([add_zero_T], clus_values[i][temperature])), np.concatenate(([0],levels_T)), c=c_list[i], linewidth = 3)
            ax.fill_betweenx(np.concatenate(([0],levels_T)), np.concatenate(([add_zero_T], clus_q1[i][temperature])),
                             np.concatenate(
                                 ([add_zero_T], clus_q3[i][temperature])),
                             color=c_list[i], alpha=0.3)
            #lab = 'Temperature difference \n with first level (°C)'

            if i == 0:
                ax.set_ylabel('Height (m)', size = lab_size)
                ax.set_yticks(levels_T)
            if i > 0:
                ax.set_yticks([])
            ax.tick_params(labelsize = lab_size)


        plt.tight_layout()

        if save_fig_profiles != '':
            plt.savefig(save_fig_profiles)
    
        else:
            fig, ax = None, None
            
        # Prandtl profiles figure
        fig, axes = plt.subplots(2, 1, figsize=fig1_size, dpi=100, sharex='row')
        for j in range(2):
            axes[j].spines[['top', 'left', 'right', 'bottom']].set_visible(False)
            axes[j].set_xticks([])
            axes[j].set_yticks([])
        #lab = 'Temperature difference \n with first level (°C)'
        axes[0].set_xlabel('Wind speed (m s$^{-1}$)', size = lab_size)
        lab = 'Temperature (°C)'            
        axes[1].set_xlabel(lab, size = lab_size)


        for i in range(n_clusters):
            
            ax = axes[0].inset_axes([0.01 + i*(1/n_clusters), 0.05, (1/n_clusters) - 0.01, 0.95])

            ax.set_ylim(0, 4.5)
            ax.set_xlim(0, 1.2*maxU)

            U = []
            z = np.linspace(0, 4, 500)[1:]
            
            aU = wind_fits.where(wind_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).aU
            bU = wind_fits.where(wind_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).bU
            cU = wind_fits.where(wind_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).cU
            aL_U = wind_fits.where(wind_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).aL_U
            aV_U = wind_fits.where(wind_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).aV_U
            Km_U = wind_fits.where(wind_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).Km_U
            
            for j in tqdm(range(len(aU))):
                best_fit = wind_fits['best_fit_u'][j]
                if best_fit == 'log':
                    u_fit = ta.log_u(
                        z, float(aU[j]), float(bU[j]), float(cU[j]))
                    ax.plot(u_fit, z, c_list[i], linewidth=0.2)
                    U.append(u_fit)
                if best_fit == 'prd':
                    u_fit = ta.Prandtl_U(
                        z, float(aL_U[j]), float(aV_U[j]), float(Km_U[j]))
                    ax.plot(u_fit, z, c_list[i], linewidth=0.2)
                    U.append(u_fit)
            U = np.array(U)
            u_med = np.nanquantile(U, 0.5, axis=0)
            u_q1 = np.nanquantile(U, 0.25, axis=0)
            u_q3 = np.nanquantile(U, 0.75, axis=0)
            ax.plot(u_med, z, 'k', linewidth=3)
            ax.plot(u_q1, z, 'k', linewidth=1.5)
            ax.plot(u_q3, z, 'k', linewidth=1.5)
            
            ax.barbs([ax.get_xlim()[0]*0.1 + ax.get_xlim()[1]*0.9]*n_lev, levels_u, np.sin(np.deg2rad(clus_values[i]['dir']) + np.pi),
                     np.cos(np.deg2rad(clus_values[i]['dir']) + np.pi), barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=10, linewidth = 4, zorder=6)
            ax.barbs([ax.get_xlim()[0]*0.15 + ax.get_xlim()[1]*0.9]*n_lev, levels_u, np.sin(np.deg2rad(clus_q1[i]['dir']) + np.pi),
                     np.cos(np.deg2rad(clus_q1[i]['dir']) + np.pi), barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=9, linewidth = 4, alpha=0.6, zorder=6)
            ax.barbs([ax.get_xlim()[0]*0.1 + ax.get_xlim()[1]*0.9]*n_lev, levels_u, np.sin(np.deg2rad(clus_q3[i]['dir']) + np.pi),
                     np.cos(np.deg2rad(clus_q3[i]['dir']) + np.pi), barb_increments={'half': 0.5, 'full': 1, 'flag': 5}, length=9, linewidth = 4, alpha=0.6, zorder=6)

            ax.scatter([ax.get_xlim()[0]*0.1 + ax.get_xlim()
                       [1]*0.9]*n_lev, levels_u, c='k', s=15, zorder=6)
            
            ax.plot(clus_values[i]['meanU'], levels_u, c='k', linewidth = 0, marker = '^', markersize=15)

            ax.set_title('Cluster {} ({} %)'.format(
                i+1, round(100*clus_freqs[i], 1)), size = lab_size)
            
            if i == 0:
                ax.set_ylabel('Height (m)', size = lab_size)
                ax.set_yticks(levels_u)
            if i > 0:
                ax.set_yticks([])
            ax.tick_params(labelsize = lab_size)

        
        for i in range(n_clusters):

            ax = axes[1].inset_axes([0.01 + i*(1/n_clusters), 0.05, (1/n_clusters) - 0.01, 0.95])
            
            ax.set_ylim(0, 5)
            ax.set_xlim(minT, maxT)
            
            T = []
            if tower == 1:
                zmax = 4.5
            else:
                zmax = 4
            z = np.linspace(0, zmax, 500)[1:]
            
            aT = temp_fits.where(temp_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).aT
            bT = temp_fits.where(temp_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).bT
            cT = temp_fits.where(temp_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).cT
            aL_T = temp_fits.where(temp_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).aL_T
            aT_T = temp_fits.where(temp_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).aT_T
            Km_T = temp_fits.where(temp_fits.time.isin(
                data.time[data.cluster.values == i]), drop=True).Km_T
            
            for j in tqdm(range(len(aT))):
                best_fit = temp_fits['best_fit_T'][j]
                if best_fit == 'log':
                    T_fit = ta.log_T(
                        z, float(aT[j]), float(bT[j]), float(cT[j])) - 273.15
                    ax.plot(T_fit, z, c_list[i], linewidth=0.2)
                    T.append(T_fit)
                if best_fit == 'prd':
                    T_fit = ta.Prandtl_T(
                        z, float(aL_T[j]), float(aT_T[j]), float(Km_T[j]))
                    ax.plot(T_fit, z, c_list[i], linewidth=0.2)
                    T.append(T_fit)
            T = np.array(T)
            T_med = np.nanquantile(T, 0.5, axis=0)
            T_q1 = np.nanquantile(T, 0.25, axis=0)
            T_q3 = np.nanquantile(T, 0.75, axis=0)
            ax.plot(T_med, z, 'k', linewidth=3)
            ax.plot(T_q1, z, 'k', linewidth=1.5)
            ax.plot(T_q3, z, 'k', linewidth=1.5)
            
            ax.plot(clus_values[i][temperature],levels_T, c='k', linewidth = 0, marker = '^', markersize=15)

            if i == 0:
                ax.set_ylabel('Height (m)', size = lab_size)
                ax.set_yticks(levels_T)
            if i > 0:
                ax.set_yticks([])
            ax.tick_params(labelsize = lab_size)


        plt.tight_layout()

        if save_fig_prandtl_profiles != '':
            plt.savefig(save_fig_prandtl_profiles)
    
        else:
            fig, ax = None, None
            
        # Variables figure
        if len(cluster_variables) <= 3:
            fig, axes = plt.subplots(
                len(cluster_variables), 1, figsize=fig1_size, dpi=100)
        else:
            fig, axes = plt.subplots((len(cluster_variables)+3) //
                                     4, 4, figsize=fig1_size, dpi=100)
    
        for i, v in enumerate(cluster_variables):
            if len(cluster_variables) == 1:
                ax = axes
            elif len(cluster_variables) <= 4:
                l = i
                ax = axes[l]
            else:
                l = (i//4, i % 4)
                ax = axes[l]
            ax.scatter(data.time, data[v][:, 2],
                            c=c_list[data.cluster.astype('int').values])
            ax.set_title(v)
        plt.tight_layout()
    
        if save_fig_variables != '':
            plt.savefig(save_fig_variables)
           
        # Daily cycles figure
        fig, ax = plt.subplots(figsize = (15, 12), dpi = 100)
        ax.set_xlabel('Hour of day (UTC)', size = lab_size)
        ax.set_ylabel('Frequency (%)', size = lab_size)
        ax.set_xlim(0,24)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
        ax.tick_params(labelsize = lab_size)
        for i in range(n_clusters):
            ax.plot(occurences[i][0], occurences[i][1]*100, c = c_list[i], label = 'Cluster {}'.format(
                i+1), linewidth = 5, alpha = 0.8) # rolling average
        ylim = ax.get_ylim()
        ax.plot([4.5]*2, ylim, ':', c='gray', linewidth = 2)
        ax.plot([18.5]*2, ylim, ':', c='gray', linewidth = 2)
        ax.set_ylim(ylim)
        ax.legend(loc = 'upper right', fontsize = lab_size/2)
        plt.tight_layout()
        if save_fig_daily != '':
            plt.savefig(save_fig_daily)
        

    clus_ids = data.cluster.values + 1
    data['cluster'] = data['cluster'] + 1

    return data, clus_ids, clus_values, clus_q1, clus_q3, X


def show_clusters_transitions(clustered_data, cmap = 'YlGn'):
    
    n = len(np.unique(clustered_data.cluster.values))
    
    M = np.zeros((n,n))
    cmap = cmap
    for i in range(len(clustered_data.cluster.values)-1):
        c_0 = clustered_data.cluster.values[i] - 1
        c_1 = clustered_data.cluster.values[i+1] - 1
        M[c_1, c_0] += 1
    M[M==0] = 0.001
    for j in range(n):
        M[:,j] = M[:,j] / np.sum(M[:,j]) * 100
    ticks = ['Cluster ' + str(i) for i in range(1,n+1)]
    fig, ax = plt.subplots(layout = 'constrained', dpi = 200)
    ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
    a = ax.imshow(M, cmap = cmap, norm=mpl.colors.LogNorm(vmin=M[M>0.5].min(), vmax=M.max()))
    threshold = a.norm(M.max())/2.
    ax.set_yticks(np.arange(len(ticks)), labels=ticks)
    ax.set_xticks(np.arange(len(ticks)), labels=ticks)
    starts_light = mpl.colormaps[cmap](0) > mpl.colormaps[cmap](1)
    if starts_light:
        c_l = ('black', 'white')
    else:
        c_l = ('white', 'black')
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            if np.isfinite(M[i,j]):
                
                c = c_l[int(a.norm(M[i, j]) > threshold)]
            else:
                c = 'white'
            text = ax.text(j, i, round(M[i, j], 1),
                           ha="center", va="center", color=c)
    plt.colorbar(a, label = 'Frequency (%)')
    ax.set_xlabel('First element of transition')
    ax.set_ylabel('Second element of transition')
    
    
def clusters_overlap(data1, data2, n, norm = True, xlab = '', ylab = '',  path=''):
    """
    Calculates the overlap and the relative frequencies of two clustered timeseries to see how the 
    clusters match.

    Parameters
    ----------
    data1 : xarray dataset
        Must contain a "cluster" variable.
    data2 : xarray dataset
        Must contain a "cluster" variable.
    n : int
        Number of clusters.
    xlab : str
        Label of the x-axis, coresponds to data1
    ylab : str
        Label of the x-axis, coresponds to data2
    path : str
        Saving path of figure
    
    Returns
    -------
    None.

    """
    clusters_overlap = np.zeros((n+1, n+1))
    d = xr.merge([data1.cluster.rename('cluster_t1'), data2.cluster.rename('cluster_t2')])
    for t in tqdm(d.time.values):
        t1 = d.sel(time = t).cluster_t1
        t2 = d.sel(time = t).cluster_t2
        if np.isnan(t1):
            t1 = n+1
        if np.isnan(t2):
            t2 = n+1
        t1 = int(t1)
        t2 = int(t2)
        clusters_overlap[t2-1, t1-1] += 1
    fig, ax = plt.subplots(figsize = (8,5), dpi = 150)
    c = clusters_overlap.copy()
    rmnan = False
    if np.sum(c[:,-1]) == 0: #remove the nan column
        c = c[:,:-1]
        rmnan = True
    bottom = np.zeros(n+1 - int(rmnan))
    if norm:
        c = c/c.sum(axis = 0)
    for i in range(n):
        if not rmnan:
            ax.bar([str(j) for j in range(1,n+1)] + ['NaN'], c[i], bottom = bottom, color = color_list[i])
        else:
            ax.bar([str(j) for j in range(1,n+1)], c[i], bottom = bottom, color = color_list[i])
        bottom = bottom + c[i]
    if not rmnan:
        ax.bar([str(j) for j in range(1,n+1)] + ['NaN'], c[n], bottom = bottom, color = 'k')
    else:
        ax.bar([str(j) for j in range(1,n+1)], c[n], bottom = bottom, color = 'k')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(-1,n+3 - int(rmnan))
    if norm:
        ax.set_ylim(0,1)
    ax.legend(['Cluster ' + str(j) for j in range(1,n+1)] + ['NaN'])
    if len(path)>0:
        plt.savefig(path)
    return clusters_overlap


def plot_clust_eofs_space(clustered_data, x, c_list = []):
    
    fig, ax = plt.subplots(1,2, figsize = (8,4), dpi = 200)
    ax[0].scatter(x[:,0], x[:,1], color = c_list[clustered_data.cluster-1], s = 20)
    ax[0].set_xlabel('EOF 1')
    ax[0].set_ylabel('EOF 2')
    ax[1].scatter(x[:,2], x[:,1], color = c_list[clustered_data.cluster-1], s = 20)
    ax[1].set_xlabel('EOF 3')
    ax[1].set_ylabel('EOF 2')
    plt.tight_layout()

    
#######################
### MAIN PARAMETERS ###
#######################
path = 'Data/SILVEX2_Silvia2/'
figpath = path + 'Figures/PCA'

color_list = np.array(['#EE5C42', '#EEAD0E', '#6E8B3D', '#008B8B', '#2F4F4F', '#68228B',
       '#9F2626', '#FFDD34', '#416618', '#0F0026'])


# choose which temperture data to use: 'temp' or 'temp_diff'
temperature = 'temp'


start_T1 = datetime.datetime(2025, 6, 23, 11, 41)
end_T1 = datetime.datetime(2025, 7, 23, 11, 5)

start_T2 = datetime.datetime(2025, 6, 23, 11, 41)
end_T2 = datetime.datetime(2025, 7, 23, 11, 5)



#####################
### MAIN FUNCTION ###
#####################

if __name__ == '__main__':

    Data_T1_orig = xr.open_dataset(
        path + 'data/HEFEXII/HEFEX2023_T303_DetrendDR_1min.nc')
    Data_T2_orig = xr.open_dataset(
        path + 'data/HEFEXII/HEFEX2023_T275_DetrendDR_1min.nc')

    snowfox_T1 = pd.read_csv(
        path + 'data/HEFEXII/T1_snowfox_lf.dat', skiprows=1).iloc[2:]
    snowfox_T2 = pd.read_csv(
        path + 'data/HEFEXII/T2_snowfox_lf.dat', skiprows=1).iloc[2:]

    wind_grad_T1 = pd.read_csv(
        path + 'results/tower_analysis/wind_gradients_T303.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'})
    wind_grad_T2 = pd.read_csv(
        path + 'results/tower_analysis/wind_gradients_T275.txt', sep='\t').rename(columns={'Unnamed: 0': 'Timestamp'})
    
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

    Data_T1, wind_grad_T1, turb_max_height_T1, wind_log_T1, temp_log_T1 = ta.clean_data(
        Data_T1_orig, levels=3, snowfox=snowfox_T1, wind_grad=wind_grad_T1, turb_max_height=turb_max_height_T1, temp_grad=temp_grad_T1, wind_log=wind_log_T1, temp_log=temp_log_T1,
        start_time=pd.Timestamp(
            2023, 8, 17, 0), end_time=pd.Timestamp(2023, 9, 26, 0), qc_threshold=0.9, fill_gaps=False)
    Data_T2, wind_grad_T2, turb_max_height_T2, wind_log_T2, temp_log_T2 = ta.clean_data(
        Data_T2_orig, levels=3, snowfox=snowfox_T2, wind_grad=wind_grad_T2, turb_max_height=turb_max_height_T2, temp_grad=temp_grad_T2, wind_log=wind_log_T2, temp_log=temp_log_T2,
        start_time=pd.Timestamp(
            2023, 8, 17, 0), end_time=pd.Timestamp(2023, 9, 26, 0), qc_threshold=0.9, fill_gaps=False)
        
    Data_T1['temp'] = Data_T1['temp'] - 273.15
    Data_T2['temp'] = Data_T2['temp'] - 273.15
    
    best_fits_temp_T1['time'] = pd.to_datetime( best_fits_temp_T1['time'])
    best_fits_temp_T2['time'] = pd.to_datetime( best_fits_temp_T2['time'])
    best_fits_wind_T1['time'] = pd.to_datetime( best_fits_wind_T1['time'])
    best_fits_wind_T2['time'] = pd.to_datetime( best_fits_wind_T2['time'])

    Data_T1['best_fit_u'] = best_fits_wind_T1.best_fit_u
    Data_T2['best_fit_u'] = best_fits_wind_T2.best_fit_u

    Data_T1['best_fit_T'] = best_fits_temp_T1.best_fit_T
    Data_T2['best_fit_T'] = best_fits_temp_T2.best_fit_T

    dic = {'time': slice(datetime.datetime(2023, 8, 22, 0),
                         datetime.datetime(2023, 8, 25, 0))}

    Data_T2['uw_diff'] = Data_T2.uw - Data_T2.uw[:,0]
    #dat_T1, dat_reduc_T1, eofs_T1, pcr, x = pca(Data_T1, n_eigs = 4, no_plot=True)
    #dat_T2, dat_reduc_T2, eofs_T2, pcr, x = pca(Data_T2, n_eigs = 10, no_plot=True)

    # DataT1, clus_T1, clus_values_T1, clus_q1_T1, clus_q3_T1 = clustering(Data_T1, 1, color_list, n_clusters = 3, dim_reduction = 3, no_plot = False,
    #save_fig1 = figpath + 'clustering/4_clusters_T1_16_var.png',
    #save_fig2 = figpath + 'clustering/4_clusters_timeseries_T1_16_var.png'
    #                                                   )

    #New_Data_T1, clus_T1, clus_values_T1, clus_q1_T1, clus_q3_T1 = clustering(variables, Data_T1, 1, color_list, n_clusters=4, dim_reduction=-1, no_plot=False,
                                                                          #save_fig1 = figpath + 'clustering/5_clusters_T2_8_var.png',
                                                                          #save_fig2 = figpath + 'clustering/5_clusters_timeseries_T2_8_var.png'
    #                                                                     )
    
    
    #######################################################
    ###TESTING FOR FINDING APPROPRIATE WAY OF CLUSTERING###
    #######################################################
    
    #%%
    
    ### BAMS : no dimension reduction, gradients are computed by bulk difference between the level 0 (bad), 
    ###      completely independent normalization, direction might not be normalized properly
    
    temperature = 'temp_diff' # which temperature to plot
    clus_variables = [
        'u_diff_level_0',
        'meanU',
        'uw_diff',
        'temp_diff',
        'dir',
        'stddir',  
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=5, dim_reduction=-1, no_plot=True, 
                                                                              save_fig_profiles='Figures/PCA/clusters_bams_SILVEX2_Silvia2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_bams_SILVEX2_Silvia2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_bams_SILVEX2_Silvia2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_bams_SILVEX2_Silvia2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_bams_SILVEX2_Silvia2.png',
                                                                              fig1_size = (30,24),
                                                                              lab_size = 45,
                                                                              bams=True,
                                                                            )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig('Figures/PCA/clusters_transitions_SILVEX2_Silvia2.png')
    
    plt.close('all')
    
    #%%
    
    ### V0 : no dimension reduction, gradients are computed by bulk difference between the level 0 (bad), 
    ###      completely independent normalization, direction might not be normalized properly
    
    temperature = 'temp_diff' # which temperature to plot
    clus_variables = [
        'u_diff_level_0',
        'meanU',
        'uw_diff',
        'temp_diff',
        'dir',
        'stddir',  
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=5, dim_reduction=-1, no_plot=False, dir_circ = False,
                                                                              save_fig_profiles='Figures/PCA/clusters_v0_SILVEX2_Silvia2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v0_SILVEX2_Silvia2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v0_SILVEX2_Silvia2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v0_SILVEX2_Silvia2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v0_SILVEX2_Silvia2.png',
                                                                              fig1_size = (48,30),
                                                                              lab_size = 30,
                                                                            )
    
    plt.close('all')
    #%%
    
    ### V1 : no dimension reduction, gradients are computed by bulk difference between the level 0 (bad), 
    ###      completely independent normalization, direction is probably normalized properly
    
    temperature = 'temp_diff' # which temperature to plot
    clus_variables = [
        'u_diff_level_0',
        'meanU',
        'uw_diff',
        'temp_diff',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=5, dim_reduction=-1, no_plot=False,
                                                                              save_fig_profiles='Figures/PC/clusters_v1_SILVEX2_Silvia2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v1_SILVEX2_Silvia2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v1_SILVEX2_Silvia2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v1_SILVEX2_Silvia2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v1_SILVEX2_Silvia2.png',
                                                                              fig1_size = (48,30),
                                                                              lab_size = 30,
                                                                            )
    plt.close('all')
    #%%
    
    ### V2 : no dimension reduction, gradients are computed relative to previous levels (better), 
    ###      completely independent normalization, direction is probably normalized properly
    
    temperature = 'temp' # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'grad_uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=5, dim_reduction=-1, no_plot=False,
                                                                              save_fig_profiles='Figures/PCA/clusters_v2_SILVEX2_Silvia2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v2_SILVEX2_Silvia2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v2_SILVEX2_Silvia2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v2_SILVEX2_Silvia2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v2_SILVEX2_Silvia2.png',
                                                                              fig1_size = (48,30),
                                                                              lab_size = 30,
                                                                            )
    plt.close('all')
    #%%
    
    ### V3 : dimension reduction with 4 EOFS, gradients are computed relative to previous levels (better),
    ###      completely independent normalization, direction is probably normalized properly

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'grad_uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=5, dim_reduction=4, no_plot=False,
                                                                              save_fig_profiles='Figures/PCA/clusters_v3_SILVEX2_Silvia2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v3_SILVEX2_Silvia2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v3_SILVEX2_Silvia2.png',
                                                                              save_fig_pca='Figures/PCA/eigenvalues_v3_SILVEX2_Silvia2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v3_SILVEX2_Silvia2.png',
                                                                              save_fig_eofs_var='Figures/PCA/eigenvalues_variables_v3_SILVEX2_Silvia2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v3_SILVEX2_Silvia2.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=30,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig('Figures/PCA/clusters_transitions_v3_SILVEX2_Silvia2.png')
    
    plt.close('all')
    #%%
    
    ### V4 : dimension reduction with 4 EOFS, gradients are computed relative to previous levels (better),
    ###      non-independent normalization (each variable is normalized by mean and standard deviation of the profile),
    ###      direction is probably normalized properly

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'grad_uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=5, dim_reduction=4, no_plot=False,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles='Figures/PCA/clusters_v4_SILVEX2_Silvia2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v4_SILVEX2_Silvia2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v4_SILVEX2_Silvia2.png',
                                                                              save_fig_pca='Figures/PCA/eigenvalues_v4_SILVEX2_Silvia2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v4_SILVEX2_Silvia2.png',
                                                                              save_fig_eofs_var='Figures/PCA/eigenvalues_variables_v4_SILVEX2_Silvia2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v4_SILVEX2_Silvia2.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=30,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig('Figures/PCA/clusters_transitions_v4_SILVEX2_Silvia2.png')
    
    plt.close('all')
    #%%
    
    ### V5 : dimension reduction with 5 EOFS, gradients are computed relative to previous levels (better),
    ###      non-independent normalization (each variable is normalized by mean and standard deviation of the profile),
    ###      direction is probably normalized properly

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'grad_uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=5, dim_reduction=5, no_plot=True,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles='Figures/PCA/clusters_v5_SILVEX2_Silvia2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v5_SILVEX2_Silvia2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v5_SILVEX2_Silvia2.png',
                                                                              save_fig_pca='Figures/PCA/eigenvalues_v5_SILVEX2_Silvia2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v5_SILVEX2_Silvia2.png',
                                                                              save_fig_eofs_var='Figures/PCA/eigenvalues_variables_v5_SILVEX2_Silvia2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v5_SILVEX2_Silvia2.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=30,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig('Figures/PCA/clusters_transitions_v5_SILVEX2_Silvia2.png')
    
    plt.close('all')
    #%%
    
    ### V6 : dimension reduction with 5 EOFS, gradients are computed relative to previous levels (better),
    ###      non-independent normalization (each variable is normalized by mean and standard deviation of the profile),
    ###      direction is probably normalized properly
    ###      6 clusters

    p = 'Figures/PCA/clustering_T275/'

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'grad_uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=6, dim_reduction=5, no_plot=True,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles= p + 'clusters_v6_T275.png',
                                                                              save_fig_variables= p + 'clusters_variables_v6_T275.png',
                                                                              save_fig_daily= p + 'daily_clusters_v6_T275.png',
                                                                              save_fig_pca= p + 'eigenvalues_v6_T275.png',
                                                                              save_fig_dendro= p + 'dendro_v6_T275.png',
                                                                              save_fig_eofs_var= p + 'eigenvalues_variables_v6_T275.png',
                                                                              save_fig_prandtl_profiles= p + 'clusters_prandtl_v6_T275.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=50,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig( p + 'clusters_transitions_v6_T275.png')
    
    plt.close('all')
    #%%
    ### V6 for T1
    
    p = 'Figures/PCA/clustering_T303/'
    
    New_Data_T1, clus_T1, clus_values_T1, clus_q1_T1, clus_q3_T1, X1 = clustering(clus_variables, Data_T1, 1, color_list, n_clusters=6, dim_reduction=5, no_plot=False,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles= p + 'clusters_v6_T303.png',
                                                                              save_fig_variables= p + 'clusters_variables_v6_T303.png',
                                                                              save_fig_daily= p + 'daily_clusters_v6_T303.png',
                                                                              save_fig_pca= p + 'eigenvalues_v6_T303.png',
                                                                              save_fig_dendro= p + 'dendro_v6_T303.png',
                                                                              save_fig_eofs_var= p + 'eigenvalues_variables_v6_T303.png',
                                                                              save_fig_prandtl_profiles= p + 'clusters_prandtl_v6_T303.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=50,
                                                                              levels_u=[1,2,3],
                                                                              levels_T=[1,2,3]
                                                                              )
    
    show_clusters_transitions(New_Data_T1)
    plt.savefig( p + 'clusters_transitions_v6_T303.png')
    
    plt.close('all')

     #%%
     ### V6 for T1 and T2 simultaneously
     ### Some information is lost because both towers are not always recording at the same time,
     ### so we won't get the same clusters
     
    p = 'Figures/PCA/'
     
    timestamps = xr.merge([Data_T1.dropna(dim = 'time', subset = clus_variables), Data_T2.dropna(dim = 'time', subset = clus_variables)], join = 'inner', compat = 'override').time
    
    dset2 = Data_T2.sel(time = timestamps).copy()
    for s in list(Data_T1.variables)[2:]:
        dset2[[s+'_T1']] = Data_T1[s]
        
    dset1 = Data_T1.sel(time = timestamps).copy()
    for s in list(Data_T2.variables)[2:]:
        dset1[[s+'_T2']] = Data_T2[s]
    
    New_Data_T1_T2, clus_T1_T2, clus_values_T1_T2, clus_q1_T1_T2, clus_q3_T1_T2, X_T1_T2 = clustering(clus_variables + [s + '_T2' for s in clus_variables], 
                                                                              dset1, 1, color_list, n_clusters=6, dim_reduction=5, no_plot=False,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles= p + 'clusters_v6_T303_two_towers.png',
                                                                              save_fig_variables= p + 'clusters_variables_v6_T303_two_towers.png',
                                                                              save_fig_daily= p + 'daily_clusters_v6_two_towers.png',
                                                                              save_fig_pca= p + 'eigenvalues_v6_T303_two_towers.png',
                                                                              save_fig_dendro= p + 'dendro_v6_two_towers.png',
                                                                              save_fig_eofs_var= p + 'eigenvalues_variables_v6_T303_two_towers.png',
                                                                              save_fig_prandtl_profiles= p + 'clusters_prandtl_v6_T303_two_towers.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=50,
                                                                              levels_u=[1,2,3],
                                                                              levels_T=[1,2,3]
                                                                              )
    
    New_Data_T2_T1, clus_T2_T1, clus_values_T2_T1, clus_q1_T2_T1, clus_q3_T2_T1, X_T2_T1 = clustering(clus_variables + [s + '_T1' for s in clus_variables], 
                                                                              dset2, 2, color_list, n_clusters=6, dim_reduction=5, no_plot=False,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles= p + 'clusters_v6_T275_two_towers.png',
                                                                              save_fig_variables= p + 'clusters_variables_v6_T275_two_towers.png',
                                                                              #save_fig_daily= p + 'daily_clusters_v6_T275_two_towers.png',
                                                                              save_fig_pca= p + 'eigenvalues_v6_T275_two_towers.png',
                                                                              #save_fig_dendro= p + 'dendro_v6_T275_two_towers.png',
                                                                              save_fig_eofs_var= p + 'eigenvalues_variables_v6_T275_two_towers.png',
                                                                              save_fig_prandtl_profiles= p + 'clusters_prandtl_v6_T275_two_towers.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=50,
                                                                              )
    
    show_clusters_transitions(New_Data_T1_T2)
    plt.savefig( p + 'clusters_transitions_v6_two_towers.png')
    
     
    plt.close('all')
    #%%
    
    ### V7 : dimension reduction with 5 EOFS, gradients are computed relative to previous levels (better),
    ###      non-independent normalization (each variable is normalized by mean and standard deviation of the profile),
    ###      direction is probably normalized properly
    ###      7 clusters

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'grad_uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=7, dim_reduction=5, no_plot=False,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles='Figures/PCA/clusters_v7_T2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v7_T2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v7_T2.png',
                                                                              save_fig_pca='Figures/PCA/eigenvalues_v7_T2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v7_T2.png',
                                                                              save_fig_eofs_var='Figures/PCA/eigenvalues_variables_v7_T2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v7_T2.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=30,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig('Figures/PCA/clusters_transitions_v7_T2.png')
    
    plt.close('all')
    #%%
    
    ### V8 : dimension reduction with 5 EOFS, gradients are computed relative to previous levels (better),
    ###      non-independent normalization (each variable is normalized by mean and standard deviation of the profile),
    ###      direction is probably normalized properly
    ###      8 clusters

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'grad_uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=8, dim_reduction=5, no_plot=False,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles='Figures/PCA/clusters_v8_T2.png',
                                                                              save_fig_variables='Figures/PCA/clusters_variables_v8_T2.png',
                                                                              save_fig_daily='Figures/PCA/daily_clusters_v8_T2.png',
                                                                              save_fig_pca='Figures/PCA/eigenvalues_v8_T2.png',
                                                                              save_fig_dendro='Figures/PCA/dendro_v8_T2.png',
                                                                              save_fig_eofs_var='Figures/PCA/eigenvalues_variables_v8_T2.png',
                                                                              save_fig_prandtl_profiles='Figures/PCA/clusters_prandtl_v8_T2.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=30,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig('Figures/PCA/clusters_transitions_v8_T2.png')
    
    plt.close('all')

    #%%
    
    ### V9 : dimension reduction with 5 EOFS, gradients are computed relative to previous levels (better),
    ###      non-independent normalization (each variable is normalized by mean and standard deviation of the profile),
    ###      direction normalized according to circular statistics but that's wrong
    ###      using uw instead of grad uw
    ###      6 clusters

    p = '/home/giordano/Work/ENS2/stage/figures/tower_analysis/clustering/after_18_12_24/'

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'uw',
        'grad_temp',
        'dir',
        'stddir',
    ]
    
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=6, dim_reduction=5, no_plot=False,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles= p + 'clusters_v9_T275.png',
                                                                              save_fig_variables= p + 'clusters_variables_v9_T275.png',
                                                                              save_fig_daily= p + 'daily_clusters_v69T275.png',
                                                                              save_fig_pca= p + 'eigenvalues_v9_T275.png',
                                                                              save_fig_dendro= p + 'dendro_v9_T275.png',
                                                                              save_fig_eofs_var= p + 'eigenvalues_variables_v9_T275.png',
                                                                              save_fig_prandtl_profiles= p + 'clusters_prandtl_v9_T275.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=50,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig( p + 'clusters_transitions_v9_T275.png')
    
    plt.close('all')

    #%%
    
    ### V10 : dimension reduction with 5 EOFS, gradients are computed relative to previous levels (better),
    ###      non-independent normalization (each variable is normalized by mean and standard deviation of the profile),
    ###      direction normalized separating easting and northing components
    ###      using uw instead of grad uw
    ###      6 clusters

    p = 'Figures/PCA/clustering_T275/'

    temperature = 'temp'  # which temperature to plot
    clus_variables = [
        'temp',
        'grad_meanU',
        'meanU',
        'uw',
        'grad_temp',
        'easting',
        'northing',
        'stddir',
    ]
    
    New_Data_T2, clus_T2, clus_values_T2, clus_q1_T2, clus_q3_T2, X = clustering(clus_variables, Data_T2, 2, color_list, n_clusters=6, dim_reduction=5, no_plot=True,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles= p + 'clusters_v10_T275.png',
                                                                              save_fig_variables= p + 'clusters_variables_v10_T275.png',
                                                                              save_fig_daily= p + 'daily_clusters_v10_T275.png',
                                                                              save_fig_pca= p + 'eigenvalues_v10_T275.png',
                                                                              save_fig_dendro= p + 'dendro_v10_T275.png',
                                                                              save_fig_eofs_var= p + 'eigenvalues_variables_v10_T275.png',
                                                                              save_fig_prandtl_profiles= p + 'clusters_prandtl_v10_T275.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=50,
                                                                              )
    
    show_clusters_transitions(New_Data_T2)
    plt.savefig( p + 'clusters_transitions_v10_T275.png')
    
    plt.close('all')

#%%
    ### V10 for T1
    
    p = 'Figures/PCA/clustering_T303/'
    
    New_Data_T1, clus_T1, clus_values_T1, clus_q1_T1, clus_q3_T1, X1 = clustering(clus_variables, Data_T1, 1, color_list, n_clusters=6, dim_reduction=5, no_plot=True,
                                                                              independent_normalization=False,
                                                                              save_fig_profiles= p + 'clusters_v10_T303.png',
                                                                              save_fig_variables= p + 'clusters_variables_v10_T303.png',
                                                                              save_fig_daily= p + 'daily_clusters_v10_T303.png',
                                                                              save_fig_pca= p + 'eigenvalues_v10_T303.png',
                                                                              save_fig_dendro= p + 'dendro_v10_T303.png',
                                                                              save_fig_eofs_var= p + 'eigenvalues_variables_v10_T303.png',
                                                                              save_fig_prandtl_profiles= p + 'clusters_prandtl_v10_T303.png',
                                                                              fig1_size=(
                                                                                  48, 30),
                                                                              lab_size=50,
                                                                              levels_u=[1,2,3],
                                                                              levels_T=[1,2,3]
                                                                              )
    
    show_clusters_transitions(New_Data_T1)
    plt.savefig( p + 'clusters_transitions_v10_T303.png')
    
    plt.close('all')

#%%
    c = clusters_overlap(New_Data_T2, New_Data_T1, 6, xlab = 'Clusters of T275', ylab = 'Clusters of T303', norm = True, path = 'Figures/PCA/clusters_T1_f_T2.png')


#%%
            

    ###Write clusters in a file
    da = xr.Dataset()
    dat = New_Data_T2
    da['time'] = dat.time
    da['cluster'] = dat.cluster
    db = da.to_dataframe()
    db.to_csv('/home/giordano/Work/ENS2/stage/results/tower_analysis/clustering/clusters_T275_bams.txt')
    
    plt.show()
