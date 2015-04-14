'''
Created on 26.1.2015

@author: tohekorh
'''

import matplotlib.pyplot as plt
import numpy as np
from aid.help import get_fileName, get_traj_and_ef, find_layers
from aid.help3 import get_shifts, get_streches#, get_shifts2
from find_angles import get_angle_set, plot_atoms, plot_KC_atoms, \
        plot_x_shift, plot_atoms2, plot_atoms3, plot_plotLogAtoms
from ase.io.trajectory import PickleTrajectory
from aid.tilt_adhesion_KC import get_e_adh
import os.path
from ase.visualize import view
from get_KC_per_atom import get_KC
import time

Ns      =   [8] #[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] #,10,11,12,13,14,15,16,17] 
v       =   1.0
edge    =   'arm'
T       =   10
stack   =   'ab'
bond    =   1.39695    

view_fit    =   False
taito       =   False

fig_width_pt    =   650 #200.0 #550 #150

inches_per_pt   =   1.0/72.27                   # Convert pt to inches
golden_mean     =   (np.sqrt(5)-1.0)/2.0           # Aesthetic ratio
fig_width       =   fig_width_pt*inches_per_pt  # width in inches
fig_height      =   fig_width*golden_mean*0.8       # height in inches
fig_size        =   [fig_width,fig_height]

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
path    =   '/space/tohekorh/BendAndSlide/pictures/anims/'

def get_datas(N, edge, taito, v, cont, nimages):
    
    if T == 0:
        indent = 'fixTop'
    elif T == 10:
        if stack == 'ab':
            indent = 'fixTop_T=10'
        elif stack == 'abc':
            indent = 'fixTop_T=10_abc'

    mdfile, mdLogFile, plotlogfile, plotKClog, plotShiftlog, plotIlDistlog, plotStrechlog \
                        =   get_fileName(N, indent, taito, v, edge)[:7]
    cmdfile, cmdLogFile, cplotlogfile, cplotKClog, cplotShiftlog, cplotIlDistlog, cplotStrechlog \
                        =   get_fileName(N, indent, taito, v, edge, cont)[:7]
    
    
    try:
        traj, ef, conc, positions_t  \
                 =   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile)
    except IOError as io:
        print io
        return 
    
    
    if conc:
        plotKClog   =   cplotKClog
        plotlogfile =   cplotlogfile
        plotShiftlog=   cplotShiftlog
        plotIlDistlog=  cplotIlDistlog 
        plotStrechlog=  cplotStrechlog
                    
    if os.path.isfile(plotKClog + '.npy'):
        e_KC_ti     =   np.load(plotKClog + '.npy')
    else:
        print 'no data KC'
        e_KC_ti     =   get_KC(traj)
        np.save(plotKClog, e_KC_ti)
    
    if os.path.isfile(plotlogfile):
        plotLog     =   np.loadtxt(plotlogfile)
    else:
        print 'no plot log'
        angles_av, plotLog      =   get_angle_set(positions_t, indent = 'fixTop', \
                                                  round_kink = True)
        np.savetxt(plotlogfile, plotLog)
    
    view(traj[0])
    if os.path.isfile(plotShiftlog + '.npy'):
        x_shift_t   =   np.load(plotShiftlog + '.npy')
        il_dist_t   =   np.load(plotIlDistlog + '.npy')
    else:
        print 'no shift log'
        x_shift_t, il_dist_t    =   get_shifts(traj, positions_t)
        np.save(plotShiftlog, x_shift_t)
        np.save(plotIlDistlog, il_dist_t)
           
    if os.path.isfile(plotStrechlog + '.npy'):
        strech_t    =   np.load(plotStrechlog + '.npy')
    else:
        print 'no strech log'
        strech_t    =   get_streches(traj, positions_t)
        np.save(plotStrechlog, strech_t)
    
    
    m           =   0
    
    if nimages < len(ef):
        x       =   max(int(len(ef)/nimages), 2)
        m       =   int(len(ef)/x)
        
        efn     =   np.zeros((m, len(ef[0])))
        plotLogn=   np.zeros((m, len(plotLog[0])))
        e_KC_n  =   np.empty(m, dtype = 'object')
        positions_tn    =   np.empty(m, dtype = 'object')
        x_shift_tn      =   np.empty(m, dtype = 'object')
        il_dist_tn      =   np.empty(m, dtype = 'object')
        strech_tn       =   np.empty(m, dtype = 'object')
        
        ntraj   =   []
        
        for i in range(m):
            ntraj.append(traj[i*x]) 
            efn[i,:]        =   ef[i*x] 
            plotLogn[i,:]   =   plotLog[i*x]
            e_KC_n[i]       =   e_KC_ti[i*x]
            positions_tn[i] =   positions_t[i*x]
            x_shift_tn[i]   =   x_shift_t[i*x]
            il_dist_tn[i]   =   il_dist_t[i*x]
            strech_tn[i]    =   strech_t[i*x]
                
            
        ef          =   efn
        traj        =   ntraj
        plotLog     =   plotLogn
        e_KC_ti     =   e_KC_n
        positions_t =   positions_tn
        x_shift_t   =   x_shift_tn
        il_dist_t   =   il_dist_tn
        strech_t    =   strech_tn
        
    nl          =   N - 2
    angles      =   plotLog[:,0:nl]
    Rads        =   plotLog[:,nl:2*nl]
    x0s         =   plotLog[:,2*nl:3*nl] 
    z0s         =   plotLog[:,3*nl:4*nl] 
    yav         =   plotLog[:,4*nl:5*nl] 
    
    angles_av   =   np.zeros(len(angles))
    for k,angle_l in enumerate(angles):
        angles_av[k]    =   np.average(angle_l)
    
    yav_p   =   yav.copy()
    yav_p[0]=   np.zeros(len(yav[0]))
    
    return traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strech_t, angles_av, \
        angles, Rads, x0s, z0s, yav_p

def get_limits(traj, positions_t, e_KC_ti, x_shift_t, z, ep, angles_av, yav_p):
    layer_indices   =   find_layers(traj[0].positions)[1]
        
    xmin        =   np.min(positions_t[0][layer_indices[-1]][:,0])
    ymax        =   np.max(positions_t[0][layer_indices[-1]][:,2])
    xmax        =   np.max(positions_t[0][layer_indices[-1]][:,0])
    ymin        =   np.min(positions_t[-1][layer_indices[0]][:,2])
    
    ymax_plot   =   ymax + (ymax - ymin)/3
    
    limits      =   [xmin, xmax, ymin, ymax_plot, 3.2, 3.6]
    maxE        =   0.
    minE        =   1000.
    maxS        =   0.
    minS        =   1000.
    
    for k in range(len(traj)):
        test_maxE   =   np.max(np.abs(e_KC_ti[k] - e_KC_ti[0]))
        test_minE   =   np.min(np.abs(e_KC_ti[k] - e_KC_ti[0]))
        test_maxS   =   np.max(x_shift_t[k][:,-2])
        test_minS   =   np.min(x_shift_t[k][:,-2])
        
        if maxE < test_maxE:
            maxE    =   test_maxE 
        if test_minE < minE:
            minE    =   test_minE
    
        if maxS < test_maxS:
            maxS    =   test_maxS 
        if test_minS < minS:
            minS    =   test_minS
    
    
    line_limits =   [np.min(z), np.max(z), np.min(ep), np.max(ep), \
                     np.min(angles_av), np.max(angles_av), \
                     np.min(yav_p), np.max(yav_p), \
                     minS, maxS]
    return limits, line_limits, minE, maxE

def plot_KC_and_lines(Ns):
    

    
    for N in Ns:
        
        traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strech_t, \
            angles_av, angles, Rads, x0s, z0s, yav_p \
                    = get_datas(N, edge, taito, v, 'cont_bend', 800)
        
        #t           =   ef[:,0] 
        z           =   ef[:,1] 
        #et          =   ef[:,4] - ef[0,4]
        #ek          =   ef[:,3]
        ep          =   ef[:,2] - ef[0,2]
        
        
        limits, line_limits, minE, maxE     =   get_limits(traj, positions_t, \
                                           e_KC_ti, x_shift_t, z, ep, angles_av, yav_p)
            
        
        for k in range(len(traj)):
            
            path_to_fig     =   path + 'pic_%04d' %k
                
            e_KC            =   e_KC_ti[k] - e_KC_ti[0]
            
            plot_atoms3(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                        N, edge, z, bond, k, minE, maxE, limits, line_limits, e_KC, \
                        x_shift_t[k], il_dist_t[k], path_to_fig)
            
            print k
        
        print 'Now sleep for 10s'
        time.sleep(10)
        
        os.system('mencoder "mf://%spic*.png" -mf type=png:fps=10  \
            -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %svideo_T=%i_N=%i.mpg' %(path, path, T, N)) 
        os.system('rm -f %spic*.png' %path)   
    

def plot_angleOfN():
    
    Ns          =   [4,5,6,8,9,10]
    zs          =   np.linspace(10, 30, 1001) #[10,12,14,16,18,20]
    z0          =   10
    angleOfz    =   np.zeros((len(zs), len(Ns)))
    
    for iN, N in enumerate(Ns):
        traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strechs_t, \
                angles_av, angles, Rads, x0s, z0s, yav_p = get_datas(N, edge, taito, v, 'cont_bend', 1e5)
        limits, _, minE, maxE  =   get_limits(traj, positions_t, e_KC_ti, x_shift_t, \
                        ef[:,1], ef[:,2] - ef[0,2], angles_av, yav_p)
        layer_indices_f =   find_layers(positions_t[0])[1]

        for iz, z0 in enumerate(zs):
                
            z           =   ef[:,1] 
            idx0        =   np.where(np.abs(z - z0) < 1e-5)[0][0]
            angleOfz[iz, iN]    =   angles_av[idx0]
            
            #plot_plotLogAtoms(positions_t[idx0], e_KC_ti[idx0] - e_KC_ti[0], layer_indices_f, \
            #                  angles[idx0], Rads[idx0], z0s[idx0], x0s[idx0], \
            #                  limits, minE, maxE)
            
        plt.plot(zs, angleOfz[:,iN], label = N)
    plt.legend(frameon = False)
    plt.show()
#plot_angleOfN()
plot_KC_and_lines(Ns)
