'''
Created on 24.9.2014

@author: tohekorh
'''


import numpy as np
from ase.calculators.lammpsrun import LAMMPS
from aid.help import make_graphene_slab, get_fileName
from aid.KC_potential_constraint import KC_potential
from aid.LJ_potential_constraint import add_adhesion
from ase.visualize import view
import time
import matplotlib.pyplot as plt
from ase.io.trajectory import PickleTrajectory
# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 

    


def time_test(width, length, N):
    
    params              =   {'bond':bond, 'a':a, 'h':h}
    
    # GRAPHENE SLAB
    #mdfile_read         =   get_fileName(N, 'tear_E_rebo+KC')[0]  
    #traj        =   PickleTrajectory(mdfile_read, 'r')
    #atoms       =   traj[0]

    
    atoms               =   make_graphene_slab(a,h,width,length, N, passivate = True)[3]
    
    
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   10
    params['chemical_symbols']  \
                =   atoms.get_chemical_symbols()
    
    # FIX
    constraints =   []
    
    add_adh     =   add_adhesion(params)
    add_kc      =   KC_potential(params)

    constraints.append(add_adh)
    constraints.append(add_kc)
    # END FIX
    
    # CALCULATOR LAMMPS 
    parameters = {'pair_style':'rebo',
                  'pair_coeff':['* * CH.airebo C H'],
                  'mass'      :['* 12.0'],
                  'units'     :'metal', 
                  'boundary'  :'f p f'}
    
    #atoms       =   traj[-1]
    
    calc        =   LAMMPS(parameters=parameters) 
    atoms.set_calculator(calc)
    # END CALCULATOR
    
    # RELAX
    atoms.set_constraint(constraints)
    #view(atoms)
    
    start_e     =   time.time()
    atoms.get_potential_energy()
    end_e       =   time.time()

    
    start_f     =   time.time()
    atoms.get_forces()
    end_f       =   time.time()
    
    
    time_e      =   end_e - start_e
    time_f      =   end_f - start_f    
    
    print 'total time:'
    print 'forces   = %.2fs, (=%.2f)' %(time_f, time_f*100/(time_f + time_e)) + '%'
    print 'energy   = %.2fs, (=%.2f)' %(time_e, time_e*100/(time_f + time_e)) + '%'    

    return len(atoms), time_e, time_f
    

m           =   12
natoms, time_e, time_f  =   np.zeros(m), np.zeros(m), np.zeros(m)

for i, n in enumerate(range(4, 4 + m)):
    width   =   1
    length  =   8*n
    natoms[i], time_e[i], time_f[i]  =   time_test(width, length, 7)

plt.plot(natoms, time_f, '-o', label = 'forces')
plt.plot(natoms, time_e, '-o', label = 'energy')
plt.ylabel('time [s]')
plt.xlabel('# of atoms')
plt.title('Time taken to calculate forces, rebo_KC')
plt.legend(loc = 2)

plt.show()
    

