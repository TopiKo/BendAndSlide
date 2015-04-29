'''
Created on 7.4.2015

@author: tohekorh
'''

import numpy as np
from get_KC_per_atom import get_KC
from aid.help import get_fileName, get_traj_and_ef
from aid.help3 import get_shifts, get_streches, get_forces_traj
from find_angles import get_angle_set

import os
import sys

N, v, edge, T, ncores   =   int(sys.argv[1]), float(sys.argv[2]), \
                            sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
#N, v, edge, T, ncores   =   5, 1, 'arm', 0, 2

taito           =   False

if T == 0:
    indent = 'fixTop'
elif T == 10:
    indent = 'fixTop_T=10'

if ncores > 1:
    calc_forces = True
else:
    calc_forces = False

def makeData(cont_id, calc_forces):
    
    mdfile, mdLogFile, plotlogfile, plotKClog, plotShiftlog, \
    plotIlDistlog, plotStrechlog, plotForcelog \
                        =   get_fileName(N, indent, taito, v, edge)[:8]
    cmdfile, cmdLogFile, cplotlogfile, cplotKClog, \
    cplotShiftlog, cplotIlDistlog, cplotStrechlog, cplotForcelog \
                        =   get_fileName(N, indent, taito, v, edge, cont_id)[:8]
    
    
    traj, ef, conc, positions_t  \
                    =   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile)
    
    if conc:
        plotKClog   =   cplotKClog
        plotlogfile =   cplotlogfile
        plotShiftlog=   cplotShiftlog
        plotIlDistlog=  cplotIlDistlog 
        plotStrechlog=  cplotStrechlog 
        plotForcelog=   cplotForcelog   
    
    if not calc_forces:       
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
            therm_idx   =   np.min(np.where(0. < z)[0]) - 1
            plotLog     =   get_angle_set(positions_t, therm_idx, indent = 'fixTop', \
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
            strech_t    =   get_streches(traj, positions_t)
            np.save(plotStrechlog, strech_t)
            
    else:
        if os.path.isfile(plotForcelog + '.npy'):
            force_t    =   np.load(plotForcelog + '.npy')
        else:
            print 'no force log'
            force_t    =   get_forces_traj(traj, ncores)
            np.save(plotForcelog, force_t)


#'cont_release'
makeData('cont_bend', calc_forces)
    
