'''
Created on 9.2.2015

@author: tohekorh
'''
import numpy as np
from aid.help import find_layers
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def kink_func(xs,*args):
    
    a2,b1,b2,c   =   args
    ys              =   np.zeros(len(xs))
    # print c, b1
    for i, x in enumerate(xs):
        
        if  x <= c: ys[i]   =    b1
        elif c < x: ys[i]   =    a2*x + b2    
    
    return ys 


def get_angle_set(traj, plot = False):
    
    
    layer_indices   =   find_layers(traj[0].positions)[1]
    n               =   len(layer_indices)
    par_set         =   np.empty((len(traj), n), dtype = 'object')

    M               =   int(len(traj)/10)
    angles          =   np.zeros(len(traj))
    dg              =   [.01,.01,.01,.01,10]
    
    for j, atom_conf in enumerate(traj):
        positions   =   atom_conf.positions

        
        
        for i, lay_inds in enumerate(layer_indices):
            xdata    =   positions[lay_inds][:,0]
            zdata    =   positions[lay_inds][:,2]
            
            left_ind    =   np.where(xdata == np.min(xdata))[0]
            
            kink_ids    =   np.where((zdata - zdata[left_ind]) < -.3)[0]
            
            if len(kink_ids) == 0:
                kink    =   np.min(xdata) + (np.max(xdata) - np.min(xdata))*3./4.
            else:
                kink_id =   np.where(xdata == np.min(xdata[kink_ids]))
                kink    =   xdata[kink_id[0]]
            ques        =   [0., zdata[0], 0., kink]
            
            par_set[j,i]    =   curve_fit(kink_func, xdata, zdata, p0=ques, \
                                      factor = 1., diag=dg, epsfcn = .00001)[0]
            angles[j]  +=   np.arctan(par_set[j,i][0])/(2*np.pi)*360/n             
             
        if plot and j%M==0:
            for i in range(n):
                use_idx =   layer_indices[i]
                xdata   =   positions[use_idx][:,0]
                ydata   =   positions[use_idx][:,2]

                x_min   =   np.min(xdata)
                x_max   =   np.max(xdata)
                
                xset    =   [x_min, par_set[j,i][3], x_max]
                
                a2,b1,b2,c   =   par_set[j, i]
                plt.scatter(xdata, ydata, alpha = 0.2, color = 'black')
                plt.plot(xset, kink_func(xset, a2,b1,b2,c), color = 'black')
            
            plt.text(np.min(positions[:,0]) + 5, np.max(positions[:,2]) + 5, 'angle = %.2f' %angles[j], fontsize=15)
            plt.axis('equal')    
            plt.show()
    
    return angles
            