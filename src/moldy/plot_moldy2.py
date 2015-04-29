'''
Created on 26.1.2015

@author: tohekorh
'''

import matplotlib.pyplot as plt
import numpy as np
from aid.help import get_fileName, get_traj_and_ef, find_layers
from aid.help3 import get_shifts, get_streches, get_forces_traj#, get_shifts2
from find_angles import get_angle_set, plot_atoms, plot_KC_atoms, \
        plot_x_shift, plot_atoms2, plot_atoms3, plot_plotLogAtoms, plot_KC_single, \
        plot_Cor_single, plot_il_single, plot_strech_single, plot_sY_single, plot_aEp_single   
from ase.io.trajectory import PickleTrajectory
from aid.tilt_adhesion_KC import get_e_adh
import os.path
from ase.visualize import view
from get_KC_per_atom import get_KC
from atom_groups import get_ind
import time

Ns      =   [8] #,9,10] #[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] #,10,11,12,13,14,15,16,17] 
v       =   1.0
edge    =   'arm'
T       =   10
stack   =   'abc'
file_t  =   'png' #svg

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
plt.tick_params(labelsize=12)
path    =   '/space/tohekorh/BendAndSlide/pictures/anims/'

def get_av(data, i, x):
    
    data_av     =   np.zeros(data[i].shape)
    x           =   float(x)
    for d in data[i*x :(i+1)*x]:
        data_av +=  d/x
        
#        for index, val in np.ndenumerate(d):
#            data_av[index]  +=   val/x 
         
    return data_av

def get_av_ct(data, i,x):
    
    nmin = 1e8
    for d in data[i*x :(i+1)*x]:
        n   =   len(d)
        if n < nmin: nmin = n
    
    data_av     =   np.zeros((nmin, len(data[0][0])))
    
    for d in data[i*x :(i+1)*x]:
        data_av +=  d[:nmin]/x

    return data_av
    
def get_datas(N, edge, taito, v, cont, stack, nimages):
    
    if T == 0:
        indent = 'fixTop'
    elif T == 10:
        if stack == 'ab':
            indent = 'fixTop_T=10'
        elif stack == 'abc':
            indent = 'fixTop_T=10_abc'

    mdfile, mdLogFile, plotlogfile, plotKClog, \
    plotShiftlog, plotIlDistlog, plotStrechlog, plotForcelog \
                        =   get_fileName(N, indent, taito, v, edge)[:8]
    cmdfile, cmdLogFile, cplotlogfile, cplotKClog, \
    cplotShiftlog, cplotIlDistlog, cplotStrechlog, cplotForcelog \
                        =   get_fileName(N, indent, taito, v, edge, cont)[:8]
    
    
    try:
        traj, ef, conc, positions_t  \
                 =   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile, edge)
    except IOError as io:
        print io
        return [None for _ in range(14)]
    
    
    if conc:
        plotKClog   =   cplotKClog
        plotlogfile =   cplotlogfile
        plotShiftlog=   cplotShiftlog
        plotIlDistlog=  cplotIlDistlog 
        plotStrechlog=  cplotStrechlog
        plotForcelog=   cplotForcelog
                    
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
        z           =   ef[:,1] 
        therm_idx    =   np.min(np.where(0. < z)[0]) - 1
        plotLog      =   get_angle_set(positions_t, therm_idx, indent = 'fixTop', \
                                                  round_kink = True)[1]
        np.savetxt(plotlogfile, plotLog)
    
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
    
    if os.path.isfile(plotForcelog + '.npy'):
        force_t    =   np.load(plotForcelog + '.npy')
    else:
        print 'no force log'
        force_t    =   get_forces_traj(traj, 2)
        np.save(plotForcelog, force_t)
     
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
        force_tn        =   np.empty(m, dtype = 'object')
        
        ntraj   =   []
        
        for i in range(m):
            ntraj.append(traj[i*x]) 
            if T == 0:
                efn[i,:]        =   ef[i*x] 
                plotLogn[i,:]   =   plotLog[i*x]
                e_KC_n[i]       =   e_KC_ti[i*x]
                positions_tn[i] =   positions_t[i*x]
                x_shift_tn[i]   =   x_shift_t[i*x]
                il_dist_tn[i]   =   il_dist_t[i*x]
                strech_tn[i]    =   strech_t[i*x]
                force_tn[i]     =   force_t[i*x]
            else:
                efn[i]          =   get_av(ef, i, x) 
                plotLogn[i]     =   get_av(plotLog, i, x)
                e_KC_n[i]       =   get_av(e_KC_ti, i, x)
                positions_tn[i] =   get_av(positions_t, i, x)
                strech_tn[i]    =   get_av(strech_t, i, x)
                force_tn[i]     =   get_av(force_t, i, x)
                x_shift_tn[i]   =   get_av_ct(x_shift_t, i, x)    
                il_dist_tn[i]   =   get_av_ct(il_dist_t, i,x)
                if i == m - 1:
                    print 'time averaging over ' + str(ef[x,0] - ef[0,0]) + 'fs'        
        
        ef          =   efn
        traj        =   ntraj
        plotLog     =   plotLogn
        e_KC_ti     =   e_KC_n
        positions_t =   positions_tn
        x_shift_t   =   x_shift_tn
        il_dist_t   =   il_dist_tn
        strech_t    =   strech_tn
        force_t     =   force_tn
    
    
    
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
    
    return traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strech_t, force_t, angles_av, \
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

def plot_KC_and_lines(Ns, edge):
    
    def make_fig(k):
        path_to_fig     =   path + 'pic_%04d' %k
        e_KC            =   e_KC_ti[k] - e_KC_ti[0]
            
        plot_atoms3(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                        N, edge, z, bond, k, minE, maxE, limits, line_limits, e_KC, \
                        x_shift_t[k], il_dist_t[k], path_to_fig)
            
    
    for N in Ns:
        
        traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strech_t, _,\
            angles_av, angles, Rads, x0s, z0s, yav_p \
                    = get_datas(N, edge, taito, v, 'cont_bend', stack, 100)
        
        #t           =   ef[:,0] 
        z           =   ef[:,1] 
        #et          =   ef[:,4] - ef[0,4]
        #ek          =   ef[:,3]
        ep          =   ef[:,2] - ef[0,2]
        
        
        limits, line_limits, minE, maxE     =   get_limits(traj, positions_t, \
                                           e_KC_ti, x_shift_t, z, ep, angles_av, yav_p)
            
        
        h_target    =   np.linspace(0, 35, 4)
        k_set   =   np.zeros(len(h_target))
    
        print h_target
        for ih, he in enumerate(h_target):
            k_set[ih]   = (np.abs(z - he)).argmin() 

        plot_KC_single(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                        N, edge, stack, z, bond, k_set, minE, maxE, limits, line_limits, e_KC_ti, \
                        x_shift_t, path + 'N%i_%s_KC_%s.%s' %(N, edge, stack, file_t))

        
        plot_Cor_single(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                        N, edge, stack, z, bond, k_set, minE, maxE, limits, line_limits, \
                        x_shift_t, path + 'N%i_%s_Cor_%s.%s' %(N, edge, stack, file_t))
        
        plot_il_single(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                        N, edge, stack, z, bond, k_set, limits, line_limits, \
                        il_dist_t, path + 'N%i_%s_Ilds_%s.%s' %(N, edge, stack, file_t))
            
        
        plot_strech_single(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                           N, edge, stack, z, bond, k_set, limits, line_limits, \
                           path + 'N%i_%s_strech_%s.%s' %(N, edge, stack, file_t))
            

        
        k = len(positions_t) - 1
        plot_sY_single(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                        N, edge, stack, z, bond, k, minE, maxE, limits, line_limits, e_KC_ti[k] - e_KC_ti[0], \
                        x_shift_t[k], il_dist_t[k], path + 'N%i_%s_shiftY_%s.pdf' %(N, edge, stack))
        
        plot_aEp_single(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, strech_t, ep, \
                        N, edge, stack, z, bond, k, minE, maxE, limits, line_limits, e_KC_ti[k] - e_KC_ti[0], \
                        x_shift_t[k], il_dist_t[k], path + 'N%i_%s_angleEp_%s.pdf' %(N, edge, stack))
        
        '''
        for k in range(len(traj)):
            
            make_fig(k)
            print k
        
        print 'Now sleep for 10s'
        time.sleep(10)
        

              
        os.system('mencoder "mf://%spic*.png" -mf type=png:fps=10  \
            -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %svideo_T=%i_N=%i_%s.mpg' %(path, path, T, N, edge)) 
        os.system('rm -f %spic*.png' %path)   
        '''
        

def plot_zOfAngle():
    M           =   1000
    Ns          =   [9] #[4,5,6,7,8,9,10]  # [4,5,6,7,8,9,10]
    study_angle =   np.sqrt(3)*bond/3.4/(2*np.pi)*360
   
    #_, [ax1, ax2, ax3]  =   plt.subplots(3, figsize = (4,6))
    #fig1, [ax1, ax2]    =   plt.subplots(2, figsize = (4,5))
    #fig3, fax           =   plt.subplots(1, figsize = (4,3))
    #fig2, eax           =   plt.subplots(1, figsize = (6,5))
    
    
    #f_arm_av    =   np.zeros((len(Ns), 3))
    #f_arm_av_abc=   np.zeros((len(Ns), 3))
    #f_zz_av     =   np.zeros((len(Ns), 3))
        
    for iN, N in enumerate(Ns):
        for stack_edge in [['ab','zz']]:
            edge    =   stack_edge[1]
            stack   =   stack_edge[0]
            
            '''
            if edge == 'arm':
                ax  =   ax1 
            elif edge == 'zz':
                ax  =   ax2    
            print N
            '''
            
            traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strechs_t, force_t, \
                angles_av, angles, Rads, x0s, z0s, yav_p \
                                =   get_datas(N, edge, taito, v, 'cont_bend', stack, M)
            
            if traj != None:
                limits, _, minE, maxE   =   get_limits(traj, positions_t, e_KC_ti, x_shift_t, \
                                                       ef[:,1], ef[:,2] - ef[0,2], angles_av, yav_p)
                layer_indices_f         =   find_layers(positions_t[0])[1]
        
                #z           =   ef[:,1] 
                idx         =   (np.abs(angles_av - study_angle)).argmin()
                
                
                plot_plotLogAtoms(positions_t[idx], e_KC_ti[idx] - e_KC_ti[0], layer_indices_f, \
                                  angles[idx], Rads[idx], z0s[idx], x0s[idx], \
                                  limits, minE, maxE)
                        
                

def plot_eOfZ():
    M           =   100
    Ns          =   [4,5,6,7,8,9,10]  # [4,5,6,7,8,9,10]
    
    _, [axarm, axzz]     =   plt.subplots(2, figsize = (4,7))
        
    
    for _, N in enumerate(Ns):
        for stack_edge in [['ab','arm'], ['abc','arm'], ['ab','zz']]:
            edge    =   stack_edge[1]
            stack   =   stack_edge[0]
            print 'N = ' + str(N)
            traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strechs_t, force_t, \
                angles_av, angles, Rads, x0s, z0s, yav_p \
                                =   get_datas(N, edge, taito, v, 'cont_bend', stack, M)
            
            if traj != None:
                limits, _, minE, maxE   =   get_limits(traj, positions_t, e_KC_ti, x_shift_t, \
                                                       ef[:,1], ef[:,2] - ef[0,2], angles_av, yav_p)
                layer_indices           =   find_layers(positions_t[0])[1][:-3]
                z                       =   ef[:,1] 
                therm_idx               =   np.min(np.where(0. < z)[0]) - 1
                e_tot                   =   np.zeros(len(traj) - therm_idx)
                
                
                print len(layer_indices)
                n=0
                for i in range(len(e_tot)):
                    for inds in layer_indices:
                        e_tot[n]   +=   np.sum(e_KC_ti[therm_idx + i][inds])
                    n += 1   
                if edge == 'arm':
                    axarm.plot(angles_av[therm_idx:], (e_tot - e_tot[0])/len(traj[0]), label = 'N=%i, %s %s' %(N, edge, stack))
                elif edge == 'zz':
                    axzz.plot(angles_av[therm_idx:], (e_tot - e_tot[0])/len(traj[0]), label = 'N=%i, %s %s' %(N, edge, stack))
        
    axarm.legend(frameon = False, loc = 2, prop={'size':10})
    axzz.legend(frameon = False, loc = 2, prop={'size':10})
    axzz.plot([40.8, 40.8], [0, .0004], '-.')
    
    plt.show()
                       

def plot_angleOfN():
    
    M           =   1000
    Ns          =   [4,5,6,7,8,9,10]  # [4,5,6,7,8,9,10]
    
    
    '''
    >>> arctan(1.39/3.4)/2/pi*360
    22.235898979967853
    >>> arctan(1.39*2/3.4)/2/pi*360
    39.27104871525512
    >>> arctan(1.39*3/3.4)/2/pi*360
    50.807996618096766
    
    zz
    >>> arctan(sqrt(3)*1.39/3.4)/(2*pi)*360
    35.302429313135221
    >>> arctan(2*sqrt(3)*1.39/3.4)/(2*pi)*360
    54.77363209792076
    '''
    #_, [ax1, ax2, ax3]  =   plt.subplots(3, figsize = (4,6))
    fig1, [ax1, ax2]    =   plt.subplots(2, figsize = (4,5))
    fig3, fax           =   plt.subplots(1, figsize = (4,3))
    fig2, eax           =   plt.subplots(1, figsize = (6,5))
    
    
    f_arm_av    =   np.zeros((len(Ns), 3))
    f_arm_av_abc=   np.zeros((len(Ns), 3))
    f_zz_av     =   np.zeros((len(Ns), 3))
        
    for iN, N in enumerate(Ns):
        for stack_edge in [['ab','arm'], ['abc','arm'], ['ab','zz']]:
            edge    =   stack_edge[1]
            stack   =   stack_edge[0]
            
            
            if edge == 'arm':
                ax  =   ax1 
            elif edge == 'zz':
                ax  =   ax2    
            print N
            
            
            traj, ef, e_KC_ti, positions_t, x_shift_t, il_dist_t, strechs_t, force_t, \
                angles_av, angles, Rads, x0s, z0s, yav_p \
                                =   get_datas(N, edge, taito, v, 'cont_bend', stack, M)
            
            if traj != None:
                limits, _, minE, maxE   =   get_limits(traj, positions_t, e_KC_ti, x_shift_t, \
                                                       ef[:,1], ef[:,2] - ef[0,2], angles_av, yav_p)
                layer_indices_f         =   find_layers(positions_t[0])[1]
        
                rend    =   get_ind(traj[0].positions.copy(), 'rend', \
                                    traj[0].get_chemical_symbols(), 2, edge)
                z       =   ef[:,1] 
                
                idx0    =   (np.abs(z - 20)).argmin()   
                    
                #zplot   =   np.zeros(len(z))
                f       =   np.zeros((len(z), 3))
                f_20    =   np.zeros((idx0, 3))
                #angleOfz=   np.zeros(len(z))
                #view(traj[0])
                
                therm_idx   =   np.min(np.where(0. < z)[0])
                
                for iz in range(len(z)):
                    
                    
                    for re in rend:
                        f[iz]          +=   force_t[iz][re]/len(rend) #[sf1, sf2, sf3]    
                        if iz < idx0:
                            f_20[iz]   +=   force_t[iz][re]/len(rend) 
                    
                    
                    '''
                    if iz% 10 == 0:
                        plot_plotLogAtoms(positions_t[idx0], e_KC_ti[idx0] - e_KC_ti[0], layer_indices_f, \
                                      angles[idx0], Rads[idx0], z0s[idx0], x0s[idx0], \
                                      limits, minE, maxE)
                        print Rads[iz]
                    '''
                
                if stack == 'abc':
                    color = 'red'
                else:
                    color = 'black'
                
                if edge == 'arm':
                    if stack == 'ab':
                        ec = 'blue'
                    else:
                        ec = 'yellow'
                else:    
                    if stack == 'ab':
                        ec = 'red'
                    else:
                        ec = 'green'
                    
                ax.plot(z, angles_av, label = 'N=%i, %s' %(N, stack), lw = 1.*N/Ns[-1], c = color)
                eax.plot(z[therm_idx:], (ef[therm_idx:,2] - ef[therm_idx, 2])/len(traj[0]), \
                         label = 'N=%i, %s %s' %(N, edge, stack), lw = 1.5*N/Ns[-1], c = ec)
            
                if edge == 'arm' and stack == 'ab':
                    #f_arm_av[iN]    =   [np.average(f[:,0]), np.average(f[:,2])]
                    f_arm_av[iN]    =   [np.average(f_20[:,0]), np.average(f_20[:,1]), np.average(f_20[:,2])]

                elif edge == 'zz' and stack == 'ab':
                    #f_zz_av[iN]     =   [np.average(f[:,0]), np.average(f[:,2])]
                    f_zz_av[iN]     =   [np.average(f_20[:,0]), np.average(f_20[:,1]), np.average(f_20[:,2])]
                elif edge == 'arm' and stack == 'abc':
                    #f_arm_av_abc[iN]     =   [np.average(f[:,0]), np.average(f[:,2])]
                    f_arm_av_abc[iN]     =   [np.average(f_20[:,0]), np.average(f_20[:,1]), np.average(f_20[:,2])]
            else: 
                print 'no data for ' + str(stack_edge) + str(N)
        
        
        ax1.plot([0,45], [22.23, 22.23], '--', c = 'black', lw = .5)
        ax1.plot([0,45], [39.27, 39.27], '--', c = 'black', lw = .5)
        ax1.plot([0,45], [50.80, 50.80], '--', c = 'black', lw = .5)
        #ax1.plot([0,45], [23., 23.], '--', c = 'black', lw = .5)
        ax2.plot([0,45], [35.30, 35.30], '--', c = 'black', lw = .5)
        ax2.plot([0,45], [54.77, 54.77], '--', c = 'black', lw = .5)
        #ax2.plot([0,45], [40.8, 40.8], '--', c = 'black', lw = .5)
        
        
        
    ax1.legend(frameon = False, loc = 4, prop={'size':10})
    ax2.legend(frameon = False, loc = 4, prop={'size':10})
    eax.legend(frameon = False, loc = 2, prop={'size':12})
    
    ax1.set_title('Edge = arm')
    ax1.set_xlabel('dist z')
    ax1.set_ylabel('angle fit deg')
    ax2.set_ylabel('angle fit deg')
    
    eax.set_xlabel('dist z')
    eax.set_ylabel('Pot E per atom eV')
    eax.set_title('Potential energies')
    eax.set_xlim([0,10])
    
    ax1.set_xlim([0,45])
    ax2.set_xlim([0,45])
    ax2.set_title('Edge = zz')
    
    
    
    
    fax.set_title('Forces average Y')
    
    
    bar_width = 0.2
    
    opacity =   .71
    Ns      =   np.array(Ns)
    
    NsStr   =   []
    for n in Ns:
        NsStr.append(str(n))
    
    
    
    rects1  =   fax.bar(Ns, f_arm_av[:,2], bar_width,
                        alpha=opacity,
                        color='b',
                        label='arm')
    
    rects2  =   fax.bar(Ns + bar_width, f_zz_av[:,2], bar_width,
                        alpha=opacity,
                        color='r',
                        label='zz')
    
    rects3  =   fax.bar(Ns + 2*bar_width, f_arm_av_abc[:,2], bar_width,
                        alpha=opacity/1.2,
                        color='b',
                        label='arm_abc')
    
    
    fax.set_xticks(Ns + 3/2.*bar_width)
    fax.set_xticklabels(NsStr)
    
    
    fax.set_xlabel('N number of layers')
    fax.set_ylabel('force eV/angst')
    fax.legend(loc = 2, frameon = False, prop={'size':12})
    
    
    plt.tight_layout()
    fig1.savefig(path + 'angles.pdf')
    plt.show()

#plot_angleOfN()
plot_zOfAngle()
#plot_eOfZ()
#for edge in ['arm']:
#    plot_KC_and_lines(Ns, edge)
#plot_KC_and_lines(Ns, edge)