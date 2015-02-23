'''
Created on 24.9.2014

@author: tohekorh
'''


import numpy as np
from ase.calculators.lammpsrun import LAMMPS
from aid.help import make_graphene_slab
from aid.KC_potential_constraint import KC_potential
from aid.KC_parallell import KC_potential_p
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
    atoms               =   make_graphene_slab(a,h,width,length, N, \
                                        edge_type = 'arm', h_pass = True)[3]
    
    
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   10
    params['chemical_symbols']  \
                =   atoms.get_chemical_symbols()
    
    # FIX 
    constraints1 =   []
    constraints2 =   []
            
    add_adh     =   add_adhesion(params)
    add_kc      =   KC_potential(params)
    add_kc_p    =   KC_potential_p(params)
    

    constraints1.append(add_kc)
    constraints2.append(add_kc_p)
    constraints1.append(add_adh)
    constraints2.append(add_adh)

    # END FIX
    
    # CALCULATOR LAMMPS 
    parameters = {'pair_style':'rebo',
                  'pair_coeff':['* * CH.airebo C H'],
                  'mass'      :['1 12.0', '2 1.0'],
                  'units'     :'metal', 
                  'boundary'  :'f p f'}
    
    
    #atoms       =   traj[-1]
    
    calc        =   LAMMPS(parameters=parameters) 
    atoms.set_calculator(calc)
    # END CALCULATOR
   
    # CALCULATE NOT PARALLEL
    atoms.set_constraint(constraints1)
    view(atoms)
    
    start_f     =   time.time()
    forces1 =   atoms.get_forces()
    end_f       =   time.time()
    
    
    start_e     =   time.time()
    e1      =   atoms.get_potential_energy()
    end_e       =   time.time()

    time_e      =   end_e - start_e
    time_f      =   end_f - start_f    
    print 'total time:'
    print 'forces   = %.2fs, (=%.2f)' %(time_f, time_f*100/(time_f + time_e)) + '%'
    print 'energy   = %.2fs, (=%.2f)' %(time_e, time_e*100/(time_f + time_e)) + '%'    

    # CALCULATE PARALLEL
    atoms.set_constraint(constraints2)

    start_f     =   time.time()
    forces2 =   atoms.get_forces()
    end_f       =   time.time()
    
    
    start_e     =   time.time()
    e2          =   atoms.get_potential_energy()
    end_e       =   time.time()
    
    
    time_e_p    =   end_e - start_e
    time_f_p    =   end_f - start_f    
    
    print 'total time:'
    print 'forces   = %.2fs, (=%.2f)' %(time_f_p, time_f_p*100/(time_f_p + time_e_p)) + '%'
    print 'energy   = %.2fs, (=%.2f)' %(time_e_p, time_e_p*100/(time_f_p + time_e_p)) + '%'    

    
    # COMPARE    
    forces_diff =   forces1 - forces2
    energy_diff =   e1 - e2
    
    
    
    for i, f in enumerate(forces_diff):
        if np.linalg.norm(f) > 1e-6: 
            print i
            print np.linalg.norm(f) 
            print 'KATASTROFI!'
            raise
    
    print 'ediff = %.10f' %energy_diff
    
    
    
    return len(atoms), time_e, time_f, time_e_p, time_f_p
    

m           =   12
natoms, time_e, time_f, time_e_p, time_f_p \
            =   np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)

for i, n in enumerate(range(4, 4 + m)):
    width   =   1
    length  =   8*n
    natoms[i], time_e[i], time_f[i], time_e_p[i], time_f_p[i]  =   time_test(width, length, 7)

plt.plot(natoms, time_f, '-D', c = 'black', label = 'forces')
plt.plot(natoms, time_e, '-o', c = 'black', label = 'energy')

plt.plot(natoms, time_f_p, '--D', c = 'black', label = 'forces_p')
plt.plot(natoms, time_e_p, '--o', c = 'black', label = 'energy_p')

plt.ylabel('time [s]')
plt.xlabel('# of atoms')
plt.title('Time taken to calculate forces, rebo_KC + PARALLEL')
plt.legend(loc = 2)

plt.show()
