'''
Created on 26.1.2015

@author: tohekorh
'''

import matplotlib.pyplot as plt
import numpy as np
from aid.help import get_fileName, get_traj_and_ef, find_layers
from aid.help3 import get_shifts#, get_shifts2
from find_angles import get_angle_set, plot_atoms, plot_KC_atoms, \
        plot_x_shift, plot_atoms2, plot_atoms3
from ase.io.trajectory import PickleTrajectory
from aid.tilt_adhesion_KC import get_e_adh
import os.path
from ase.visualize import view
from get_KC_per_atom import get_KC
import time

Ns      =   [8] #[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] #,10,11,12,13,14,15,16,17] 
v       =   1.0
edge    =   'arm'
T       =   0 #10
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


def analyze_corrugation(Ns):
    
    for N in Ns:
        mdfile, mdLogFile,      =   get_fileName(N, 'fixTop', v, edge)[:2]
        cmdfile, cmdLogFile     =   get_fileName(N, 'fixTop', v, edge, 'cont_bend')[:2]
        
        
        traj, ef, conc=   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile)[:3]
    
        m           =   0
        nimages     =   100
        
        if nimages < len(ef):
            x       =   max(int(len(ef)/nimages), 2)
            m       =   int(len(ef)/x)
            efn     =   np.zeros((m, len(ef[0])))
            ntraj   =   []
            
            for i in range(m):
                ntraj.append(traj[i*x]) 
                efn[i,:]        =   ef[i*x] 
            
            ef      =   efn
            traj    =   ntraj


        x_shift =   get_shifts(traj)
        
        for k in range(len(traj)):
            fig2, ax        =   plt.subplots(1, figsize = (8,5))
            plot_x_shift(ax, x_shift[k], edge, bond)
        
            
            
def plot_KCi(Ns):
    
    for N in Ns:
        mdfile, mdLogFile, plotlogfile, plotKClog       =   get_fileName(N, 'fixTop', v, edge)[:4]
        cmdfile, cmdLogFile, cplotlogfile, cplotKClog   =   get_fileName(N, 'fixTop', v, edge, 'cont_bend')[:4]
        
        path    =   '/space/tohekorh/BendAndSlide/pictures/anims/'
        
        traj, ef, conc, positions_t=   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile)
    
        m           =   0
        nimages     =   10000
        
        if nimages < len(ef):
            x       =   max(int(len(ef)/nimages), 2)
            m       =   int(len(ef)/x)
            efn     =   np.zeros((m, len(ef[0])))
            ntraj   =   []
            
            for i in range(m):
                ntraj.append(traj[i*x]) 
                efn[i,:]        =   ef[i*x] 
            
            ef      =   efn
            traj    =   ntraj
        
        
        if conc:
            plotKClog   =   cplotKClog
        
        if os.path.isfile(plotKClog + '.npy'):
            e_KC_ti     =   np.load(plotKClog + '.npy')
        else:
            print 'Dataa ei loydy'
            e_KC_ti     =   get_KC(traj)
            np.save(plotKClog, e_KC_ti)
        
        
        
        layer_indices   =   find_layers(traj[0].positions)[1]
        xmin        =   np.min(traj[0].positions[layer_indices[-1]][:,0])
        ymax        =   np.max(traj[0].positions[layer_indices[-1]][:,2])
        xmax        =   np.max(traj[0].positions[layer_indices[-1]][:,0])
        ymin        =   np.min(traj[-1].positions[layer_indices[0]][:,2])
        
        limits      =   [xmin, xmax, ymin, ymax]
        maxE        =   0.
        for k in range(len(traj)):
            test_maxE   =   np.max(np.abs(e_KC_ti[k] - e_KC_ti[0]))
            
            if maxE < test_maxE:
                maxE    =   test_maxE 
        
        print maxE
        
            
        
        for k, atoms in enumerate(traj):
            
            
            path_to_fig     =   path + 'pic_%04d' %k
                
            plot_KC_atoms(atoms, e_KC_ti[k] - e_KC_ti[0], \
                          layer_indices, maxE, limits, path_to_fig)
            print np.sum(e_KC_ti[k]) - np.sum(e_KC_ti[0])
        
        
        os.system('mencoder "mf://%spic*.png" -mf type=png:fps=10  \
            -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %svideo_N=%i.mpg' %(path, path, N)) 
        os.system('rm -f %spic*.png' %path)
    

def plot_KC_and_lines(Ns, T):
    
    if T == 0:
        indent = 'fixTop'
    elif T == 10:
        if stack == 'ab':
            indent = 'fixTop_T=10'
        elif stack == 'abc':
            indent = 'fixTop_T=10_abc'

    
    for N in Ns:
        
        mdfile, mdLogFile, plotlogfile, plotKClog, plotShiftlog, plotIlDistlog \
                            =   get_fileName(N, indent, taito, v, edge)[:6]
        cmdfile, cmdLogFile, cplotlogfile, cplotKClog, cplotShiftlog, cplotIlDistlog \
                            =   get_fileName(N, indent, taito, v, edge, 'cont_bend')[:6]
        
        path    =   '/space/tohekorh/BendAndSlide/pictures/anims/'
        
        traj, ef, conc, positions_t  \
                     =   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile)
    
        
        
        if conc:
            plotKClog   =   cplotKClog
            plotlogfile =   cplotlogfile
            plotShiftlog=   cplotShiftlog
            plotIlDistlog=  cplotIlDistlog 
               
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
        
        
        if os.path.isfile(plotShiftlog + '.npy'):
            x_shift_t   =   np.load(plotShiftlog + '.npy')
            il_dist_t   =   np.load(plotIlDistlog + '.npy')
        else:
            print 'no shift log'
            x_shift_t, il_dist_t    =   get_shifts(traj, positions_t)
            np.save(plotShiftlog, x_shift_t)
            np.save(plotIlDistlog, il_dist_t)
               
        
        m           =   0
        nimages     =   400
        
        if nimages < len(ef):
            x       =   max(int(len(ef)/nimages), 2)
            m       =   int(len(ef)/x)
            
            efn     =   np.zeros((m, len(ef[0])))
            plotLogn=   np.zeros((m, len(plotLog[0])))
            e_KC_n  =   np.empty(m, dtype = 'object')
            positions_tn    =   np.empty(m, dtype = 'object')
            x_shift_tn      =   np.empty(m, dtype = 'object')
            il_dist_tn      =   np.empty(m, dtype = 'object')
            
            ntraj   =   []
            
            for i in range(m):
                ntraj.append(traj[i*x]) 
                efn[i,:]        =   ef[i*x] 
                plotLogn[i,:]   =   plotLog[i*x]
                e_KC_n[i]       =   e_KC_ti[i*x]
                positions_tn[i] =   positions_t[i*x]
                x_shift_tn[i]   =   x_shift_t[i*x]
                il_dist_tn[i]   =   il_dist_t[i*x]
                
                
            ef      =   efn
            traj    =   ntraj
            plotLog =   plotLogn
            e_KC_ti =   e_KC_n
            positions_t =   positions_tn
            x_shift_t   =   x_shift_tn
            il_dist_t   =   il_dist_tn
 
 
        
        #x_shift, il_dist_t  =   get_shifts(traj, positions_t)
        
        #ia_sep              =   get_il_separation(pair_table, positions_t)
        
        
        
        t           =   ef[:,0] 
        z           =   ef[:,1] 
        et          =   ef[:,4] - ef[0,4]
        ek          =   ef[:,3]
        ep          =   ef[:,2] - ef[0,2]
        
            
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
            
        
        for k, atoms in enumerate(traj):
            
            path_to_fig     =   path + 'pic_%04d' %k
                
            e_KC            =   e_KC_ti[k] - e_KC_ti[0]
            
            plot_atoms3(positions_t, angles, angles_av, Rads, z0s, x0s, yav_p, ep, \
                        N, edge, z, bond, k, minE, maxE, limits, line_limits, e_KC, \
                        x_shift_t[k], il_dist_t[k], path_to_fig)
            
            print k
        
        print 'Now sleep for 10s'
        time.sleep(10)
        
        #mencoder "mf://pic*.png" -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o video_N=.mpg
        os.system('mencoder "mf://%spic*.png" -mf type=png:fps=10  \
            -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %svideo_T=%i_N=%i.mpg' %(path, path, T, N)) 
        os.system('rm -f %spic*.png' %path)   
    

def plot_lines(Ns):
    collect_ep      =   np.empty(len(Ns), dtype = 'object')
    collect_ek      =   np.empty(len(Ns), dtype = 'object')
    collect_et      =   np.empty(len(Ns), dtype = 'object')
    collect_z       =   np.empty(len(Ns), dtype = 'object')
    collect_etear   =   np.empty(len(Ns), dtype = 'object')
    collect_angles  =   np.empty(len(Ns), dtype = 'object')
    
        
     
    for j, N in enumerate(Ns):
        
        mdfile, mdLogFile, plotlogfile   =   get_fileName(N, 'fixTop', v, edge)[:3]
        cmdfile, cmdLogFile, cplotlogfile=   get_fileName(N, 'fixTop', v, edge, 'cont_bend')[:3]
        
        
        traj,ef, conc, positions_t \
                =   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile)
        
        if conc:
            plotlogfile =   cplotlogfile
    
        
        #traj        =   PickleTrajectory(mdfile, 'r')
        #ef          =   np.loadtxt(mdLogFile)
        
        natoms      =   len(traj[0])
        width       =   traj[0].get_cell()[1,1]
        m           =   0
        
        
        if os.path.isfile(plotlogfile):
            plotLog     =   np.loadtxt(plotlogfile)
        else:
            print 'Dataa ei loydy'
            angles_av, plotLog      =   get_angle_set(positions_t, indent = 'fixTop', \
                                                      round_kink = True)
            np.savetxt(plotlogfile, plotLog)
        
        
        
        if 1e3 < len(ef):
            x       =   max(int(len(ef)/1e3), 2)
            m       =   int(len(ef)/x)
            efn     =   np.zeros((m, len(ef[0])))
            ntraj   =   []
            plotLogS=   np.zeros((m, len(plotLog[0])))
            
            for i in range(m):
                ntraj.append(traj[i*x]) 
                efn[i,:]        =   ef[i*x] 
                plotLogS[i,:]   =   plotLog[i*x] 
            
            ef      =   efn
            traj    =   ntraj
            plotLog =   plotLogS
    
        t           =   ef[:,0] 
        z           =   ef[:,1] 
        et          =   ef[:,4] - ef[0,4]
        ek          =   ef[:,3]
        ep          =   ef[:,2] - ef[0,2]
        
            
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
        
        
        # Animation
        '''
        path_anim    =      '/space/tohekorh/BendAndSlide/pictures/anims/' 
        if not os.path.isfile(path_anim + 'N=%i_%s.mp4' %(N, edge)):
            
            print plt.rcParams['animation.codec']
            #FFwriter = animation.FFMpegWriter()
            fig2, axs   =   plt.subplots(4, 1, figsize = (6,12))
            axs         =   np.append(axs, axs[1].twinx())
            print axs
            #fig2     =     plt.figure()
            ims      =     []
            for k, traj_s in enumerate(traj):
                if k%(len(traj)/10) == 0:
                #if k%1 == 0:
                    print z[k]
                    ims.append(plot_atoms(axs, traj[0], traj_s, \
                                          angles, angles_av, Rads, z0s, x0s, yav_p, ep, \
                                          N, edge, z, bond, k))
            
            im_ani = animation.ArtistAnimation(fig2, ims, interval=100, repeat_delay=3000, blit=True)
            
            axs[1].set_xlabel('z')
            axs[1].set_ylabel('Angle Deg')
            axs[2].set_xlabel('z')
            axs[2].set_ylabel('shift in y')
            axs[3].set_xlabel('z')
            axs[3].set_ylabel('R*phi diff')
            axs[4].set_ylabel('PotE eV')
            #im_ani.save(path_anim, writer=FFwriter)
            plt.show()   
        '''
        fig2, axs   =   plt.subplots(4, 1, figsize = (6,12))
        axs         =   np.append(axs, axs[1].twinx())
        
        
        axs[0].set_title('N=%i, edge = %s' %(N, edge))        
        plot_atoms(axs, traj[0], traj[-1], \
                   angles, angles_av, Rads, z0s, x0s, yav_p, ep, \
                   N, edge, z, bond, len(z) - 1)
        axs[1].set_xlabel('z')
        axs[1].set_ylabel('Angle Deg')
        axs[2].set_xlabel('z')
        axs[2].set_ylabel('shift in y')
        axs[3].set_xlabel('z')
        axs[3].set_ylabel('R*phi diff')
        axs[4].set_ylabel('PotE eV')
        
        plt.savefig(r'/space/tohekorh/BendAndSlide/pictures/fig1_N=%i_%s.svg' %(N, edge))
        plt.show()
        
        # PLOT individual
        f, (ax1, ax2)   =   plt.subplots(1, 2, figsize = fig_size)
        legp, leg       =   [], []
        c               =   ['b', 'g', 'r']
        
        
        ax1.set_title(r'Energies, N=%i' %N)
        ax1.plot(z, et, '-',  linewidth = 2, label = r'etot', color = 'black')
        ax1.plot(z, ek, '--', linewidth = 1, label = r'ekin', color = 'black') 
        ax1.plot(z, ep, '-',  linewidth = 1, label = r'epot', color = 'black')
        ax1.set_ylabel(r'E ev')
        
        ax1.legend(loc = 2, frameon = False)
        
        ax2.set_title(r'Potential energy ev')
        
        e_tear      =   np.zeros(len(z))
        A_per_atom  =   np.sqrt(3)/2.*bond**2.
        
        for i in range(len(z)):
            E_adh_dens  =   get_e_adh(angles_av[i])/A_per_atom
            angle_rad   =   angles_av[i]/360*2*np.pi
            e_tear[i]   =   E_adh_dens*width*1./np.sin(angle_rad)*z[i]
            
        ax2.plot(z, ep,     '-',  linewidth = 1, label = r'epot', color = 'black')
        ax2.plot(z, e_tear, '--', linewidth = 1, label = r'tear', color = 'black')
        ax2.set_ylabel(r'E_p ev')
        ax2.legend(loc = 3, frameon = False)
        
        
        ax3 = ax2.twinx()
        ax3.plot(z, angles_av, '-.', linewidth = 1, label = r'angle', color = 'black')
        ax3.set_ylabel(r'Kink angle Deg')
        
        ax2.set_xlabel(r'Bend heigh Angstrom')
        ax3.legend(loc = 2, frameon = False)
        
        plt.savefig(r'/space/tohekorh/BendAndSlide/pictures/fig2_N=%i_%s.svg' %(N, edge))
        #plt.show()
        
        #
        collect_ep[j]       =   ep/natoms
        collect_ek[j]       =   ek/natoms
        collect_et[j]       =   et/natoms
        collect_z[j]        =   z
        collect_etear[j]    =   e_tear/natoms
        collect_angles[j]   =   angles_av
        #
    
    fal, ax1a   =   plt.subplots(1, figsize = fig_size)
    ax1a.set_title(r'Pot Energies')
    ax1a.set_xlabel(r'Bend heigh Angstrom')
    ax1a.set_ylabel(r'Ep/atoms eV')
           
    for i, n in enumerate(Ns):
        ep          =   collect_ep[i]
        ek          =   collect_ek[i]
        et          =   collect_et[i]
        z           =   collect_z[i]
        etear       =   collect_etear[i]
        angles_av   =   collect_angles[i]    
        
        #ax1a.plot(z, et, '-',  linewidth = 2, label = r'etot N = %i' %n, color = 'black')
        #ax1a.plot(z, ek, '--', linewidth = 1, label = r'ekin N = %i' %n, color = 'black') 
        ax1a.plot(z, ep, '-',  linewidth = 1, label = r'epot N = %i' %n)
    
    ax1a.legend(loc = 4, frameon = False)
    
    plt.show()

#analyze_corrugation([6])
#plot_KCi([12]) #,7,8,9,10,11,12,13,14,15,16,17])
#plot_lines([8])
plot_KC_and_lines(Ns, T)
'''

# TEST
f, ax   =   plt.subplots(1, figsize = fig_size)
ax.set_title(r'Pot Energies')
ax.set_xlabel(r'Bend heigh Angstrom')
ax.set_ylabel(r'Ep/atoms eV')



path_test_old   =   '/space/tohekorh/BendAndSlide/files/taito/test/md_N=9_v=1.00.log'
path_test_new   =   '/space/tohekorh/BendAndSlide/files/taito/fixTop/N=9_v=1/arm/md_N=9_v=1.00_arm.log'


ef_o            =   np.loadtxt(path_test_old)
ef_n            =   np.loadtxt(path_test_new)

ep_o            =   ef_o[:,2][:1000] - ef_o[0,2]
z_o             =   ef_o[:,1][:1000]
ep_n            =   ef_n[:,2][:1000] - ef_n[0,2]
z_n             =   ef_n[:,1][:1000]


    
(ep_o - ep_n)/(ef_n[:,2][:1000] - ef_n[0,2])

ax.plot(z_o, ep_o, label = 'old')
ax.plot(z_n, ep_n, label = 'new')
ax.legend()
plt.show()
'''
