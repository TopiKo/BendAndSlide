'''
Created on 9.2.2015

@author: tohekorh
'''
import numpy as np
from aid.help import find_layers
from scipy.optimize import curve_fit, fmin_l_bfgs_b
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import os
def kink_func(xs,*args):
    
    a2,b1,b2,c   =   args
    ys              =   np.zeros(len(xs))
    # print c, b1
    for i, x in enumerate(xs):
        
        if  x <= c: ys[i]   =    b1
        elif c < x: ys[i]   =    a2*x + b2    
    
    return ys 

def kink_func2(xs, *args):
    
    R, phi, z0, x0  =   args
    ys              =   np.zeros(len(xs))
    
    for i, x in enumerate(xs):
        
        if  x <= x0: 
            ys[i]   =    z0
        elif x0 < x <= x0 + R*np.sin(phi): 
            ys[i]   =    np.sqrt(R**2 - (x - x0)**2) + (z0 - R)    
        elif x0 + R*np.sin(phi) < x: 
            ys[i]   =    -np.tan(phi)*x + z0 - R*(1-np.cos(phi)) \
                                        + np.tan(phi)*(x0 + R*np.sin(phi))    
        
    return ys 

def kink_func3(popsis, *args):
    
    R, phi, z0, x0      =   popsis
    xs,y                =   args[0], args[1]
    ys                  =   np.zeros(len(xs))
    
    for i, x in enumerate(xs):
        
        if  x <= x0: 
            ys[i]   =    z0
        elif x0 < x <= x0 + R*np.sin(phi): 
            ys[i]   =    np.sqrt(R**2 - (x - x0)**2) + (z0 - R)    
        elif x0 + R*np.sin(phi) < x: 
            ys[i]   =    -np.tan(phi)*x + z0 - R*(1-np.cos(phi)) \
                                        + np.tan(phi)*(x0 + R*np.sin(phi))    
    
    Rsqr    =   0.
    for i in range(len(y)):
        Rsqr    +=  (ys[i] - y[i])**2   
        
    return Rsqr 


    
def plot_atoms(ax1, ax2, ax3, traj_init, traj_c, angles_t, angles_av_t, Rads_t, \
               z0s_t, x0s_t, ep, N, edge, z_t, iz):
    
    layer_indices   =   find_layers(traj_init.positions)[1][:-2]
    positions       =   traj_c.positions   
    xdata       =   positions[:,0]
    ydata       =   positions[:,2]
    zsets       =   np.zeros((1000, len(layer_indices)))
    xsets       =   np.zeros((1000, len(layer_indices)))

    n           =   len(layer_indices) 
    angles_av   =   0.
    
    angles  =   angles_t[iz]
    Rads    =   Rads_t[iz]
    z0s     =   z0s_t[iz]
    x0s     =   x0s_t[iz]
    z       =   z_t[iz]
    
    #ax =   plt.gca()
    #plt.figure(1, figsize = (8,6))
    for i in range(n):
        x_min   =   np.min(xdata[layer_indices[i]])
        x_max   =   np.max(xdata[layer_indices[i]])
        
        xsets[:,i]      =   np.linspace(x_min, x_max, 1000)
        
        R, phi, z0, x0  =   Rads[i], angles[i]/360.*2*np.pi, z0s[i], x0s[i] #par_set[j, i]
        
        zsets[:,i]      =   kink_func2(xsets[:,i], R, phi, z0, x0)
        
        angles_av  +=   angles[i]/n   

    
    im  =   [ax1.scatter(xdata, ydata, alpha = 0.2, color = 'black')]
    for i in range(n):
        im.append(ax1.plot(xsets[:,i], zsets[:,i], color = 'black')[0])
    
    
    im.append(ax2.plot(z_t[:iz], angles_av_t[:iz], color = 'black')[0])
    im.append(ax3.plot(z_t[:iz], ep[:iz], '-.', color = 'black', label = 'eP')[0])
    
    im.append(ax2.text(np.max(z_t) - 15, 1.5*np.max(angles_av_t)/7., '- angle', fontsize=15))
    im.append(ax2.text(np.max(z_t) - 15, 2*np.max(angles_av_t)/7., '-. eP', fontsize=15))
    
    im.append(ax2.text(np.max(z_t) - 15, np.max(angles_av_t)/14., 'angle = %.2f Deg' %angles_av, fontsize=15))
    im.append(ax2.text(np.max(z_t) - 15, np.max(angles_av_t)/7., 'z = %.2f Angst' %z, fontsize=15))
    
    
    #
        
    #im     =   ax.scatter(xdata, ydata, alpha = 0.2, color = 'black')
    #ims.append(im)
        
    
    ax1.axis('equal') 
    
    '''
    path_bendF  =   '/space/tohekorh/BendAndSlide/pictures/bend/N=%i/%s/' %(N, edge)   
    
    if os.path.exists(path_bendF):
        plt.savefig(path_bendF + r'fig_h=%i.pdf' %iz)
    else:
        os.makedirs(path_bendF)
        plt.savefig(path_bendF + r'fig_h=%i.pdf' %iz)
    '''
    
    return im
        
def get_angle_set(traj, indent = 'fixTop', round_kink = True):
    
    
            
    if indent == 'fixTop':
        layer_indices   =   find_layers(traj[0].positions)[1][:-2]
    
    n               =   len(layer_indices)
    par_set         =   np.empty((len(traj), n), dtype = 'object')

    #M               =   int(len(traj)/4)
    angles          =   np.zeros((len(traj), len(layer_indices)))
    Rads            =   np.zeros((len(traj), len(layer_indices)))
    z0s             =   np.zeros((len(traj), len(layer_indices)))
    x0s             =   np.zeros((len(traj), len(layer_indices)))
    
    yav             =   np.zeros((len(traj), len(layer_indices)))
    angles_av       =   np.zeros(len(angles))
    
    dg              =   [.01,.01,.01,.01,10]
    
    
    
    for j, atom_conf in enumerate(traj):
    
        positions   =   atom_conf.positions
        
        for i, lay_inds in enumerate(layer_indices):
            xdata    =   positions[lay_inds][:,0]
            ydata    =   positions[lay_inds][:,1]
            zdata    =   positions[lay_inds][:,2]
            
            if j > 0:
                yav[j, i]   =   np.average(ydata) - yav[0, i]
            elif j == 0:
                yav[j, i]   =   np.average(ydata)
            
            left_ind    =   np.where(xdata == np.min(xdata))[0][0]
            
            kink_ids    =   np.where((zdata - zdata[left_ind]) < -.3)[0]
            
            if len(kink_ids) == 0:
                kink    =   np.min(xdata) + (np.max(xdata) - np.min(xdata))*9./10.
            else:
                kink_id =   np.where(xdata == np.min(xdata[kink_ids]))
                kink    =   xdata[kink_id[0]]
            
            if not round_kink:
                ques        =   [0., zdata[left_ind], 0., kink]
                
                par_set[j,i]    =   curve_fit(kink_func, xdata, zdata, p0=ques, \
                                          factor = 1., diag=dg, epsfcn = .00001)[0]
            
                angles[j][i]   +=   -np.arctan(par_set[j,i][0])/(2*np.pi)*360/n             
            
            elif round_kink:
                '''
                ques            =   [20., 0., zdata[left_ind], kink]
                par_set[j,i]    =   curve_fit(kink_func2, xdata, zdata, p0=ques, \
                                          factor = 1., diag=dg, epsfcn = .00001, \
                                          maxfev = 1000*len(xdata))[0]
                '''
                hz              =   zdata[left_ind]
                Rmax            =   (max(xdata) - kink)*1.2
                ques2           =   np.array([Rmax/2., 0., hz, kink])
                bounds          =   [(0, Rmax), (0., np.pi/2.1), (hz-.3, hz+.3), (0., None)]
                par_set[j,i]    =   fmin_l_bfgs_b(kink_func3, ques2, args=(xdata, zdata), \
                                                  approx_grad = True, bounds = bounds)[0]
                
                angles[j,i]     =   par_set[j,i][1]/(2*np.pi)*360
                Rads[j,i]       =   par_set[j,i][0]
                z0s[j,i]        =   par_set[j,i][2]
                x0s[j,i]        =   par_set[j,i][3]
                
        for angle in angles[j]:
            angles_av[j]   +=   angle/len(angles[j])   
        
        '''    
        
        if  j%M==0:
            plot_atoms(traj[0],  atom_conf, angles[j], Rads[j], z0s[j], x0s[j])
        if  j%M==0:
            xdata   =   positions[:,0]
            ydata   =   positions[:,2]

            for i in range(n):
                
                
                x_min   =   np.min(xdata[layer_indices[i]])
                x_max   =   np.max(xdata[layer_indices[i]])
                
                xset    =   [x_min, par_set[j,i][3], x_max]
                xset    =   np.linspace(x_min, x_max, 1000)
                if not round_kink:
                    a2,b1,b2,c      =   par_set[j, i]
                    plt.plot(xset, kink_func(xset, a2,b1,b2,c), color = 'black')
                elif round_kink:
                    R, phi, z0, x0  =   par_set[j, i]
                    print R, phi, z0, x0
                    plt.plot(xset, kink_func2(xset, R, phi, z0, x0), color = 'black')
                    
            plt.scatter(xdata, ydata, alpha = 0.2, color = 'black')
            
                
            plt.text(np.min(positions[:,0]) + 5, np.max(positions[:,2]) + 5, 'angle = %.2f' %angles_av[j], fontsize=15)
            plt.axis('equal')    
            plt.show()
        '''
    plot_data   =   np.concatenate((angles, Rads, x0s, z0s, yav), axis=1)
        
    return angles_av, plot_data 
            