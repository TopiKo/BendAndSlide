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
from aid.help import make_graphene_slab, saveAndPrint, get_fileName
from atom_groups import get_ind
from ase.md.langevin import Langevin
from aid.KC_potential_constraint import KC_potential
from aid.KC_parallel import KC_potential_p
from ase.visualize import view 
import sys

#N, v, M, edge, ncores   =   int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])

N, v, M, edge, ncores   =   3, 1.,  10000, 'zz', 2 

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.3705 
dt          =   2               # units: fs
length      =   4*14 #16            # slab length has to be integer*2
width       =   1               # slab width
fixtop      =   2               #

# SIMULATION PARAMS
dt          =   2               # fs
dz          =   dt*v/1000.      # d/M  
fric        =   0.002

tau         =   10./fric
T           =   0.       # ~10K       # temperature
interval    =   10              # interval for writing stuff down
    

def run_moldy(N, save = False):
    
    # 
    params      =   {'bond':bond, 'a':a, 'h':h}
    
    # DEFINE FILES
    mdfile, mdlogfile, mdrelax = get_fileName(N, 'tear_E_rebo+KC_v', v, edge)  
    
    # GRAPHENE SLAB
    atoms               =   make_graphene_slab(a,h,width,length,N, \
                                               edge_type = edge, h_pass = True)[3]
    view(atoms)
    params['ncores']    =   ncores
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   10
    params['chemical_symbols']  =   atoms.get_chemical_symbols()
    
    # FIX
    constraints =   []
    
    left        =   get_ind(atoms.positions.copy(), 'left', 2, bond)
    top         =   get_ind(atoms.positions.copy(), 'top', fixtop - 1, left)
    rend_t      =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), 1)
    rend        =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), fixtop)
    
    
    fix_left    =   FixAtoms(indices = left)
    fix_top     =   FixAtoms(indices = top)
    
    add_kc      =   KC_potential_p(params)

    
    for ind in rend:
        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
        constraints.append(fix_deform)
    
    for ind in rend_t:
        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
        constraints.append(fix_deform)
    
    constraints.append(fix_left)
    constraints.append(fix_top)
    constraints.append(add_kc)
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
    
    #view(atoms)
    
    # TRAJECTORY
    if save:    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    else:       traj    =   None
    
    #data    =   np.zeros((M/interval, 5))
    
    # RELAX
    atoms.set_constraint(add_kc)
    dyn     =   BFGS(atoms, trajectory = mdrelax)
    dyn.run(fmax=0.05)
    
    # FIX AFTER RELAXATION
    atoms.set_constraint(constraints)
    
    # DYNAMICS
    dyn     =   Langevin(atoms, dt*units.fs, T*units.kB, fric)
    n       =   0
    header  =   '#t [fs], d [Angstrom], epot_tot [eV], ekin_tot [eV], etot_tot [eV] \n'
    log_f   =   open(mdlogfile, 'w')
    log_f.write(header)            
    log_f.close()

    print 'Start the dynamics for N = %i' %N
    
    for i in range(0, M):
        
        if T == 0:
            for ind in rend:
                atoms[ind].position[2] -= dz 
        elif T != 0:
            if tau < i*dt:
                for ind in rend:
                    atoms[ind].position[2] -= dz 
            
        dyn.run(1)
        
        if i%interval == 0:

            epot, ekin  =   saveAndPrint(atoms, traj, False)[:2]
            data        =   [i*dt, i*dz, epot, ekin, epot + ekin]
            
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
                
                if T != 0 and i*dt == tau:
                    log_f.write('# Thermalization complete. ' +  '\n')
                log_f.write(stringi +  '\n')
                log_f.close()
                  

            n += 1
        
        if 1e2 <= M:    
            if i%(int(M/100)) == 0: print 'ready = %.1f' %(i/(int(M/100))) + '%' 
    


run_moldy(N, True)   
    
