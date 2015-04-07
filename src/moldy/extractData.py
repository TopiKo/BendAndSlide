'''
Created on 7.4.2015

@author: tohekorh
'''

import numpy as np
from get_KC_per_atom import get_KC
from aid.help import get_fileName, get_traj_and_ef
from aid.help3 import get_shifts
from find_angles import get_angle_set
import os
import sys

N, v, edge, T   =   int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
taito           =   False

if T == 0:
    indent = 'fixTop'
elif T == 10:
    indent = 'fixTop_T=10'
    
mdfile, mdLogFile, plotlogfile, plotKClog, plotShiftlog, plotIlDistlog \
                    =   get_fileName(N, indent, taito, v, edge)[:6]
cmdfile, cmdLogFile, cplotlogfile, cplotKClog, cplotShiftlog, cplotIlDistlog \
                    =   get_fileName(N, indent, taito, v, edge, 'cont_bend')[:6]



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

if os.path.isfile(plotlogfile):
    x_shift_t   =   np.loadtxt(plotShiftlog)
    il_dist_t   =   np.loadtxt(plotIlDistlog)
else:
    print 'no shift log'
    x_shift_t, il_dist_t    =   get_shifts(traj, positions_t)
    np.savetxt(plotShiftlog, x_shift_t)
    np.savetxt(plotIlDistlog, il_dist_t)
    
