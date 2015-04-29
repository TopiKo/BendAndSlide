'''
Created on 27.1.2015

@author: tohekorh
'''

# This class creates the deforming structure.
# *************************************


from ase import units, Atoms
import numpy as np
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.lammpsrun import LAMMPS
from ase.io.trajectory import PickleTrajectory
from aid.help import saveAndPrint, get_fileName
from atom_groups import get_ind
from ase.md.langevin import Langevin
#from aid.my_constraint import add_adhesion, KC_potential
from ase.optimize import BFGS
#from aid.KC_potential_constraint import KC_potential
from aid.KC_parallel import KC_potential_p
from ase.visualize import view 
import sys

#N, v, edge, ncores   =   int(sys.argv[1]), float(sys.argv[2]), sys.argv[4], \
#                                int(sys.argv[8])

N, v, edge, ncores   =   8, 1., 'arm', 2

taito       =   False
  

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.37
dt          =   2               # units: fs
fixtop      =   2               #

# SIMULATION PARAMS
dt          =   2               # fs
dz          =   dt*v/1000.      # d/M  
fric        =   0.002
delta_h     =   2.


tau         =   10./fric
T           =   0.              # temperature
interval    =   10              # interval for writing stuff down


def remove_atoms(atoms, indices):
        
    atoms_n =   Atoms()
    atoms_n.set_cell(atoms.get_cell())
    atoms_n.set_pbc(atoms.get_pbc())
    for ia, a in enumerate(atoms):
        if ia not in indices: 
            atoms_n +=  a
    
    return atoms_n

def run_moldy(N, save = False):
    
    cont_type           =   'cont_release'
    params              =   {'bond':bond, 'a':a, 'h':h}
    
    # DEFINE FILES
#    mdfile_read         =   get_fileName(N, 'tear_E_rebo+KC_v', v, edge)[0]  
#    mdfile, mdlogfile   =   get_fileName(N, 'tear_E_rebo+KC_v', v, edge, cont_type)[:2]    
    mdfile_read         =   get_fileName(N, 'fixTop', taito, v, edge)[0]  
    mdfile, mdlogfile   =   get_fileName(N, 'fixTop', taito, v, edge, cont_type)[:2]    
    
    
    
    # GRAPHENE SLAB
    traj_a      =   PickleTrajectory(mdfile_read, 'r')
    
    
    
    atoms       =   traj_a[0]
    
    top_rm_inds =   get_ind(atoms.positions.copy(), 'top', 2, [])
    atoms       =   remove_atoms(atoms, top_rm_inds) 
    fixtop      =   0
        
    params['ncores']    =   ncores
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   10
    params['chemical_symbols']  =   atoms.get_chemical_symbols()
    
    # FIX
    #constraints         =   []
    #constraints_init    =   []
    
    left        =   get_ind(atoms.positions.copy(), 'left', 2, bond)
    rend        =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), fixtop, edge)


    heights     =   np.zeros(len(traj_a))
    for i, atoms in enumerate(traj_a):
        heights[i]  =   traj_a[0][rend].positions[0][2] - atoms[rend].positions[0][2] 
    
    h_scale     =   heights[-1] - heights[0]
    h_target    =   np.linspace(delta_h, h_scale, int(h_scale/delta_h))
    begin_set   =   np.zeros(len(h_target))
    
    for ih, he in enumerate(h_target):
        begin_set[ih]   = (np.abs(heights - he)).argmin()   
    
    
    fix_left    =   FixAtoms(indices = left)
    add_kc      =   KC_potential_p(params)

    
    constraints =   []
    constraints_def =   []

    constraints.append(add_kc)
    constraints.append(fix_left)
    
    for ind in rend:
        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
        constraints_def.append(fix_deform)
    
    
    constraints_simul   =   constraints[:]
    for const in constraints_def:
        constraints.append(const)
    constraints_init    =   constraints
    
    # use initial atoms to obtain fixes
    header  =   '#t [fs], d [Angstrom], epot_tot [eV], ekin_tot [eV] \n'
    log_f   =   open(mdlogfile, 'w')
    log_f.write(header)            
    log_f.close()
    
    if save:    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    else:       traj    =   None
    stable          =   False
    
    for n_begin in begin_set:
        n_begin     =   int(n_begin)
        atoms       =   traj_a[n_begin]
        atoms       =   remove_atoms(atoms, top_rm_inds) 
            
        atoms.set_constraint(constraints_init)
        # END FIX
        
        # CALCULATOR LAMMPS 
        parameters = {'pair_style':'rebo',
                      'pair_coeff':['* * CH.airebo C H'],
                      'mass'      :['1 12.0', '2 1.0'],
                      'units'     :'metal', 
                      'boundary'  :'f p f'}
        
        calc    =   LAMMPS(parameters=parameters) 
        
        atoms.set_calculator(calc)
        # END CALCULATOR
        
        
        # TRAJECTORY
        dyn_r   =   BFGS(atoms)
        dyn_r.run(fmax=0.05)
        
        del atoms.constraints
        atoms.set_constraint(constraints_simul)
            
        
        # DYNAMICS
        dyn         =   Langevin(atoms, dt*units.fs, T*units.kB, fric)
        i           =   0
        finished    =  False
        h_init      =  atoms[rend].positions[0][2] 
        
        while not finished:
            
            dyn.run(1)
            
            hw          =   atoms[rend].positions[0][2] - h_init
            if i%interval == 0:
                
                epot, ekin  =   saveAndPrint(atoms, traj, False)[:2]
                
                data        =   [i*dt, hw, epot, ekin, epot + ekin]
                
                
                if save:
                    log_f   =   open(mdlogfile, 'a')
                    stringi =   ''
                    for k,d in enumerate(data):
                        if k == 0:           
                            stringi += '%.2f ' %d
                        elif k == 1:
                            stringi += '%.6f ' %d
                        else:
                            stringi += '%.12f ' %d
                    
                    log_f.write(stringi +  '\n')
                    log_f.close()
            
            
            if 2e3 < i:
                stable      =   True
                finished    =   True
            
            if delta_h/3.    <   abs(hw):
                finished    =   True 
                    
            print n_begin, i, hw
            
            i += 1
        
        if stable:
            print 'found staple position at n_begin = %i' %n_begin
            exit()



run_moldy(N, True)
    