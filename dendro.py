import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import matplotlib.patches as patches


def cluster_relabel(y: np.ndarray) -> np.ndarray:
    '''Reassign cluster labels such that the biggest cluster is cluster 0, next biggest is cluster 1, etc.
    
    Parameters
    ----------
    y : ndarray
        Array containing class labels as integers from 0 to n-1, where n is the number of clusters
        
    Returns
    -------
    arr[y] : ndarray
        Relabelled array
        
    Example
    -------
    >>> y_labels = np.array([2,2,2,0,1,1,2])
    >>> cluster_relabel(y_labels)
    array([0,0,0,2,1,1,0])
        
    '''
    n_clusters = len(set(y))
    arr = np.zeros(n_clusters + 1, dtype=int)
    arr[np.arange(n_clusters)[np.bincount(y).argsort()]] = np.arange(n_clusters)[::-1]
    return arr[y]

def decompose_linkage(k,Z,n_obs):
    if k<n_obs:
        yield k
    else:
        for i in Z[k-n_obs][:2]:
            yield from decompose_linkage(i,Z,n_obs)

def improved_dendrogram(ax: plt.axis,
                        dat: np.ndarray,
                        n: int,
                        p: int=20,
                        c_list: list = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'],
                        sc: list = [[]]*10,
                        cluster_names: list = None) -> tuple[plt.axis, np.ndarray]:
    
    '''Using shc and ward's method, plot a dendrogram that is more customizable and visually appealing
    
    Parameters
    ----------
    ax : axis
        Axis object to plot on
        
    dat : ndarray
        2-dimensional numpy array to on which to cluster.
        Should be oriented such that dat.cluster = (n_observations,n_variables)
        
    n : int
        Desired number of clusters
        
    p : int
        Number of leaves at bottom of dendrogram
        
    c_list : list
        Colors to assign to each cluster. len(c_list) must be >= n
        
    cluster_names: list
        Names to label under each cluster
        
        
    Returns
    -------
    ax : axis
        Axis object containing dendrogram
    
    clus: ndarray
        Array of cluster labels

    '''
    
    
    ### Correct inputs to reduce errors
    n_obs = dat.shape[0]
    c_list = np.array(c_list)
    
    ### First, cluster as normal. 
    ### We close out of the dendrogram because we're going to plot it ourselves. We just need dend
    Z = shc.linkage(dat, method = 'ward')
    dend = shc.dendrogram(Z, 
            orientation='top', 
            truncate_mode='lastp',
            count_sort='descending',
            p=p)
    plt.cla()
    
    ### Relabel the clusters such that the first one is the one with the most elements
    clus = cluster_relabel(shc.fcluster(Z,n,criterion='maxclust')-1)

    ### Cast lists as arrays for more convenience
    y_c = np.array(dend['dcoord'])
    x_c = np.array(dend['icoord'])
    Z = np.array(Z,dtype=int)
    
    ### For each of the lowest leaves determine to which cluster it belongs
    ### Run next() on generator because we only need one as by definition all obs in one leaf are 
    CC = clus[np.array([next(decompose_linkage(i,Z,n_obs)) for i in dend['leaves']])]

    ### Determine what color belongs between what horizontal extent
    ### e.g. if c_lims[2] = [12.5,22.5], then all observations between these numbers should be c_list[2]
    c_lims = []
    for i in range(n):
        r = (5+10*np.arange(p))[CC==i]
        c_lims.append(np.array([r[0]-5,r[-1]+5]))
    c_lims = np.array(c_lims)
    
    ### Automatically determine a cutoff given number of clusters
    cutoff = np.mean(sorted(y_c[:,1])[::-1][n-2:n])

    
    
    ### Add elements to ax object
    ax.axhline(cutoff,c="k",ls="--")
    for X,Y in zip(x_c,y_c):

        color_l = c_list[:n][(X[0]>=c_lims[:,0]) & (X[0]<c_lims[:,1])][0]
        color_r = c_list[:n][(X[-1]>=c_lims[:,0]) & (X[-1]<c_lims[:,1])][0]
        
        if all(Y<cutoff): # Color when all lines below cutoff
            ax.plot(X,Y,c=color_l)

        elif all(Y>cutoff): # Color when all lines above cutoff
            ax.plot(X,Y,c='k')

        else:
            ax.plot(X[1:3],Y[1:3],c="k") # Horizontal line above cutoff
            if (Y[0]<cutoff) & (Y[1]>cutoff): # Left-side line breaks over cutoff
                ax.plot(X[:2],[Y[0],cutoff],c=color_l)
                ax.plot(X[:2],[cutoff,Y[1]],c="k")
            else:
                ax.plot(X[:2],Y[:2],c="k")

            if (Y[2]>cutoff) & (Y[3]<cutoff): # Right-side line breaks over cutoff
                ax.plot(X[2:],[Y[3],cutoff],c=color_r)
                ax.plot(X[2:],[cutoff,Y[2]],c="k")
            else:
                ax.plot(X[2:],Y[2:],c="k")

    for i in range(len(c_lims)):
        ax.add_patch(patches.Rectangle((c_lims[i][0]+1, 0), c_lims[i][1]-c_lims[i][0]-2, cutoff*0.9, facecolor=c_list[i],alpha=0.1,zorder=-2))
    ### Formatting
    ax.set_xticks([])
    ax.set_ylabel("Intercluster distance")
    for i in range(n):
        if cluster_names:
            ax.text(c_lims.mean(axis=1)[i],-0.2,cluster_names[i] + '\n ({})'.format(sc[i]),ha='center',va='top')
        else:
            ax.text(c_lims.mean(axis=1)[i],-0.2,'C{} \n({})'.format(i+1, sc[i]),ha='center',va='top')
    ax.set_xlim(0,p*10)
    ax.set_ylim(ymin=0)
    
    return ax, clus