'''
Created on 7.4.2015

@author: tohekorh
'''

import numpy as np
from get_KC_per_atom import get_KC
from aid.help import get_fileName, get_traj_and_ef
from aid.help3 import get_shifts, get_streches
from find_angles import get_angle_set
import os
import sys

N, v, edge, T   =   int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
taito           =   False

if T == 0:
    indent = 'fixTop'
elif T == 10:
    indent = 'fixTop_T=10'




def makeData(cont_id):
    
    mdfile, mdLogFile, plotlogfile, plotKClog, plotShiftlog, plotIlDistlog, plotStrechlog \
                        =   get_fileName(N, indent, taito, v, edge)[:7]
    cmdfile, cmdLogFile, cplotlogfile, cplotKClog, cplotShiftlog, cplotIlDistlog, cplotStrechlog \
                        =   get_fileName(N, indent, taito, v, edge, cont_id)[:7]
    
    
    traj, ef, conc, positions_t  \
                    =   get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile)
    
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
        plotLog      =   get_angle_set(positions_t, indent = 'fixTop', \
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


#'cont_release'
makeData('cont_bend')
    
