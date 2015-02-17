'''
Created on 12.2.2015

@author: tohekorh
'''
from ase.io.trajectory import PickleTrajectory
import numpy as np
from ase.units import *


print 0.1*fs

#20.00 0.020000 -2505.585201408766 0.003948638987 -2505.581252769779 
#20.00 0.020000 -2505.585201408766 0.008183026989 -2505.577018381777 

'''
mdfile      =   '/space/tohekorh/BendAndSlide/files/test_N=6_L=8/md_N=6_v=1.00_test.traj'
mdlogfile   =   '/space/tohekorh/BendAndSlide/files/test_N=6_L=8/md_N=6_v=1.00_test.log'

mdfile_t    =   '/space/tohekorh/BendAndSlide/files/test_N=6_L=8/md_N=6_v=1.00_taito.traj'
mdlogfile_t =   '/space/tohekorh/BendAndSlide/files/test_N=6_L=8/md_N=6_v=1.00_taito.log'
'''

path        =   '/space/tohekorh/BendAndSlide/files/test_lammps/N=5/'

mdfile      =   '/space/tohekorh/BendAndSlide/files/test_lammps/N=5/md.traj'
mdlogfile   =   '/space/tohekorh/BendAndSlide/files/test_lammps/N=5/md.log'

mdfile_t    =   '/space/tohekorh/BendAndSlide/files/test_lammps/N=5/md_taito.traj'
mdlogfile_t =   '/space/tohekorh/BendAndSlide/files/test_lammps/N=5/md_taito.log'


traj        =   PickleTrajectory(mdfile, 'r')
ef          =   np.loadtxt(mdlogfile)

traj_t      =   PickleTrajectory(mdfile_t, 'r')
ef_t        =   np.loadtxt(mdlogfile_t)


dt          =   ef[1,0] - ef[0,0]
dt_t        =   ef[1,0] - ef[0,0]

if dt_t != dt: raise


n           =   min(len(traj), len(traj_t)) - 3 

va          =   np.empty(n-1, dtype = 'object')
va_t        =   np.empty(n-1, dtype = 'object')

for i in range(n):
    if i + 1 <= n - 1:
        print 'hii'
        va[i]   =   (traj[i].positions - traj[i + 1].positions)/dt
        va_t[i] =   (traj_t[i].positions - traj_t[i + 1].positions)/dt_t
        
        r_va    =   np.loadtxt(path + 'v_i=%i' %(i*10))
        r_va_t  =   np.loadtxt(path + 'v_i=%i_taito' %(i*10))

        
        Ek          =   0.
        Ek_t        =   0.
        r_Ek        =   0.
        r_Ek_t      =   0.
        

#        print va[i] - r_va
#        print va_t[i] - r_va_t
        
        for v in va[i]:
            #print   np.linalg.norm(v*1e5)
            Ek     +=   .5*12.0*np.linalg.norm(v)**2*1.66e-27*6.24e18*(1e-10/1e-15)**2    #v = [Angst/fs] 

        for v in va_t[i]:
            #print   np.linalg.norm(v*1e5)
            Ek_t   +=   .5*12.0*np.linalg.norm(v)**2*1.66e-27*6.24e18*(1e-10/1e-15)**2    #v = [Angst/fs] 
        
        for v in r_va:
            v_APfs  =   v*0.0982269353
            r_Ek   +=   .5*12.0*np.linalg.norm(v_APfs)**2*1.66e-27*6.24e18*(1e-10/1e-15)**2    #v = [Angst/fs] 

        for v in r_va_t:
            v_APfs  =   v*0.0982269353
            r_Ek_t +=   .5*12.0*np.linalg.norm(v_APfs)**2*1.66e-27*6.24e18*(1e-10/1e-15)**2    #v = [Angst/fs] 

        mx_r, mx_r_traj, mx_traj = 0., 0., 0.
        
        for j in range(len(r_va)):
            dev =   np.linalg.norm(r_va[j]- r_va_t[j])*0.0982269353 
            if mx_r < dev:  mx_r = dev   
            
        for j in range(len(r_va)):
            dev = np.linalg.norm(va_t[i][j] - r_va_t[j]*0.0982269353 )
            if mx_r_traj < dev: mx_r_traj   =   dev 
        
        for j in range(len(va[i])):
            dev     =   np.linalg.norm(va[i][j] - va_t[i][j])
            if mx_traj < dev: mx_traj = dev
        
        print mx_r, mx_r_traj, mx_traj      

        if ef[i,1] != ef_t[i,1]:    raise    
        
        print ef[i,1], Ek, Ek_t, r_Ek, r_Ek_t, ef[i,3], ef_t[i,3]

