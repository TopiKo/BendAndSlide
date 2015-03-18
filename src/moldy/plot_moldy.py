'''
Created on 26.1.2015

@author: tohekorh
'''

import matplotlib.pyplot as plt
import numpy as np
from aid.help import get_fileName
from find_angles import get_angle_set, plot_atoms
from ase.io.trajectory import PickleTrajectory
from aid.tilt_adhesion_KC import get_e_adh
import os.path
import matplotlib.animation as animation

Ns      =   [12] #[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] #,10,11,12,13,14,15,16,17] 
v       =   1.0
edge    =   'arm'
bond    =   1.39695    

view_fit    =   False

fig_width_pt    =   650 #200.0 #550 #150

inches_per_pt   =   1.0/72.27                   # Convert pt to inches
golden_mean     =   (np.sqrt(5)-1.0)/2.0           # Aesthetic ratio
fig_width       =   fig_width_pt*inches_per_pt  # width in inches
fig_height      =   fig_width*golden_mean*0.8       # height in inches
fig_size        =   [fig_width,fig_height]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'


collect_ep      =   np.empty(len(Ns), dtype = 'object')
collect_ek      =   np.empty(len(Ns), dtype = 'object')
collect_et      =   np.empty(len(Ns), dtype = 'object')
collect_z       =   np.empty(len(Ns), dtype = 'object')
collect_etear   =   np.empty(len(Ns), dtype = 'object')
collect_angles  =   np.empty(len(Ns), dtype = 'object')

    
 
for j, N in enumerate(Ns):
    
    mdfile, mdLogFile, plotlogfile   =   get_fileName(N, 'fixTop', v, edge)[:3]
    
    traj        =   PickleTrajectory(mdfile, 'r')
    natoms      =   len(traj[0])
    width       =   traj[0].get_cell()[1,1]
    ef          =   np.loadtxt(mdLogFile)
    m           =   0
    
    
    if os.path.isfile(plotlogfile):
        plotLog     =   np.loadtxt(plotlogfile)
    else:
        print 'Dataa ei loydy'
        raise
        angles_av, plotLog      =   get_angle_set(traj, plot = False, indent = 'fixTop', \
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
    
    
    # Animation
    path_anim    =      '/space/tohekorh/BendAndSlide/pictures/anims/' 
    if not os.path.isfile(path_anim + 'N=%i_%s.mp4' %(N, edge)):
        
        print plt.rcParams['animation.codec']
        #FFwriter = animation.FFMpegWriter()
        fig2, (ax1, ax2)   =   plt.subplots(2, 1, figsize = (6,8))
        ax3     =   ax2.twinx()
        #fig2     =     plt.figure()
        ims      =     []
        for k, traj_s in enumerate(traj):
            #if k%(len(traj)/10) == 0:
            if k%1 == 0:
                print z[k]
                ims.append(plot_atoms(ax1, ax2, ax3, traj[0], traj_s, \
                                      angles, angles_av, Rads, z0s, x0s, ep, \
                                      N, edge, z, k))
        
        im_ani = animation.ArtistAnimation(fig2, ims, interval=100, repeat_delay=3000, blit=True)
        
        ax2.set_xlabel('z')
        ax2.set_ylabel('Angle Deg')
        ax3.set_ylabel('PotE eV')
        #im_ani.save(path_anim, writer=FFwriter)
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
    
    plt.savefig(r'/space/tohekorh/BendAndSlide/pictures/fig_N=%i_%s.svg' %(N, edge))
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
