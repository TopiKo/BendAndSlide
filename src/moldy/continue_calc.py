'''
Created on 27.1.2015

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
from aid.help import make_graphene_slab, saveAndPrint, get_fileName
from atom_groups import get_mask, get_ind
from ase.md.langevin import Langevin
#from aid.my_constraint import add_adhesion, KC_potential
from aid.KC_potential_constraint import KC_potential
from aid.LJ_potential_constraint import add_adhesion
from aid.help import find_layers
from ase.visualize import view 

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 
dt          =   2               # units: fs
length      =   2*12            # slab length has to be integer*2
width       =   1               # slab width
fix_top     =   0               # 

# SIMULATION PARAMS
M           =   int(1e4)        # number of moldy steps
d           =   3*h             # maximum separation
dz          =   d/M
dt          =   2               # fs
T           =   0.              # temperature
interval    =   10              # interval for writing stuff down
    


def run_moldy(N, save = False):
    
    params      =   {'bond':bond, 'a':a, 'h':h}
    
    # DEFINE FILES
    mdfile_read         =   get_fileName(N, 'tear_E_rebo+KC')[0]  
    mdfile, mdlogfile   =   get_fileName(N, 'tear_E_rebo+KC', 'continue_release')[:2]    

    print mdfile, mdlogfile
    # GRAPHENE SLAB
    traj        =   PickleTrajectory(mdfile_read, 'r')
    
    atoms       =   traj[0]
    
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   10
    params['chemical_symbols']  =   atoms.get_chemical_symbols()
    
    # FIX
    constraints =   []
    
    #zset, layer_inds     =   find_layers(atoms.positions.copy())
    left        =   get_mask(atoms.positions.copy(), 'left', 2, bond)[0]
    rend        =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), fix_top)
    
    fix_left    =   FixAtoms(mask = left)
    
    add_adh     =   add_adhesion(params)
    add_kc      =   KC_potential(params)

    
    for ind in rend:
        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
        #constraints.append(fix_deform)
    
    constraints.append(fix_left)
    constraints.append(add_adh)
    constraints.append(add_kc)
    # END FIX
    
    # CALCULATOR LAMMPS 
    parameters = {'pair_style':'rebo',
                  'pair_coeff':['* * CH.airebo C H'],
                  'mass'      :['* 12.0'],
                  'units'     :'metal', 
                  'boundary'  :'f p f'}
    
    calc = LAMMPS(parameters=parameters) 
    
    atoms   =   traj[-1]
    atoms.set_calculator(calc)
    # END CALCULATOR
    
    
    
    # TRAJECTORY
    if save:    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    else:       traj    =   None
    
    data = np.zeros((M/interval, 5))
    
    view(atoms)
    
    
    # FIX AFTER RELAXATION
    atoms.set_constraint(constraints)
    
    # DYNAMICS
    dyn     =   Langevin(atoms, dt*units.fs, T*units.kB, 0.002)
    n       =   0
    print 'Start the dynamics for N = %i' %N
    
    for i in range(0, M):
        #for ind in rend:
        #    atoms[ind].position[2] -= dz 
        dyn.run(1)
        
        if i%interval == 0:
            epot, ekin = saveAndPrint(atoms, traj, False)[:2]
            data[n] = [i*dt, i*dz, epot, ekin, epot + ekin]
            n += 1
            
        if i%(int(M/100)) == 0: print 'ready = %.1f' %(i/(int(M/100))) + '%' 
    
    if save:
        header = 't [fs], d [Angstrom], epot/Atom [eV], ekin/Atom [eV], etot/Atom [eV]'
        np.savetxt(mdlogfile, data, header = header)  


run_moldy(5, True)
#for N in range(3, 15):    
#    run_moldy(N, True)   
    