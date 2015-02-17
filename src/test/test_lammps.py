'''
Created on 24.9.2014

@author: tohekorh
'''
# This class creates the deforming structure.
# *************************************


from ase import units
import numpy as np
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS
from ase.io.trajectory import PickleTrajectory
from aid.help import make_graphene_slab, saveAndPrint
from atom_groups import get_mask, get_ind
from ase.md.langevin import Langevin
#from aid.my_constraint import add_adhesion, KC_potential
from aid.KC_potential_constraint import KC_potential
from aid.LJ_potential_constraint import add_adhesion
#from aid.help import find_layers
from ase.visualize import view 


# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 
width       =   1
M           =   1000
dt          =   2
dz          =   1e-3
T           =   0

def run_moldy():
    
    KC = False
    
    # DEFINE FILES
    mdfile, mdlogfile   =   'md.traj', 'md.log'   
    
    atoms               =   make_graphene_slab(a,h,width,4,2, passivate = True)[3]
    
   
    params              =   {'bond':bond, 'a':a, 'h':h}
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   14
    params['chemical_symbols']  =   atoms.get_chemical_symbols()
    
    constraints =   []
    add_kc      =   KC_potential(params)
    left        =   get_ind(atoms.positions.copy(), 'left', 1, bond)
    rend        =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), 0)
    
    
    '''
    # FIX
    
    #zset, layer_inds     =   find_layers(atoms.positions.copy())
    
    
    add_adh     =   add_adhesion(params)
    add_kc      =   KC_potential(params)

    
    #constraints.append(add_adh)
    
    # END FIX
    '''
    
    fix_left    =   FixAtoms(indices = left)
    
    for ind in rend:
        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
        constraints.append(fix_deform)
    
    
    constraints.append(fix_left)
    if KC: constraints.append(add_kc)
    
    # CALCULATOR LAMMPS 
    parameters = {'pair_style':'rebo',
                  'pair_coeff':['* * CH.airebo C H'],
                  'mass'      :['1 12.0', '2 1.0'],
                  'units'     :'metal', 
                  'boundary'  :'f p f'}
    
    calc = LAMMPS(parameters=parameters) 
    atoms.set_calculator(calc)
    # END CALCULATOR
    
    
    
    atoms.set_constraint(constraints)
    
    #view(atoms)
    
    
    # TRAJECTORY
    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    
    data    =   np.zeros((M, 6))
    
    # DYNAMICS
    dyn     =   Langevin(atoms, dt*units.fs, T*units.kB, 0.002)
    n       =   0
    header  =   '#t [fs], d [Angstrom], epot_tot [eV], ekin_tot [eV], etot_tot [eV] \n'
    log_f   =   open(mdlogfile, 'w')
    log_f.write(header)            
    log_f.close()

    
    for i in range(0, M):
        #for ind in rend:
        atoms[3].position[2] -= dz 
        dyn.run(1)

        va      =   atoms.get_velocities()
        vamax   =   np.linalg.norm(va[3]*0.0982269353)
        #print vamax
        epot, ekin  =   saveAndPrint(atoms, traj, False)[:2]
        data[n]     =   [i*dt, i*dz, epot, ekin, epot + ekin, vamax]
        
         
        log_f   =   open(mdlogfile, 'a')
        stringi =   ''
        for k,d in enumerate(data[n]):
            if k == 0:           
                stringi += '%.2f ' %d
            elif k == 1:
                stringi += '%.6f ' %d
            else:
                stringi += '%.12f ' %d
        log_f.write(stringi +  '\n')
        log_f.close()
        np.savetxt(mdlogfile, data, header = header)  

        n += 1
            
        if i%(int(M/100)) == 0: print 'ready = %.1f' %(i/(int(M/100))) + '%' 
    


run_moldy()
#for N in range(3, 15):    
#    run_moldy(N, True)   
    
