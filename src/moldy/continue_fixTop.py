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
from ase.io.trajectory import PickleTrajectory
from aid.help import saveAndPrint, get_fileName
from atom_groups import get_ind
from ase.md.langevin import Langevin
#from aid.my_constraint import add_adhesion, KC_potential
from ase.optimize import BFGS
#from aid.KC_potential_constraint import KC_potential
from aid.KC_parallel import KC_potential_p
from ase.visualize import view 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as mbd
import sys

#N, v, M, edge, release, ncores   =   int(sys.argv[1]), float(sys.argv[2]), \
#                                    int(sys.argv[3]), sys.argv[4], \
#                                    sys.argv[5] in ['True', 'true', 1], int(sys.argv[6]) 

N, v, M, edge, release, ncores   =   8, 1., 10000, 'arm', True, 2

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

tau         =   10./fric
T           =   0.              # temperature
interval    =   10              # interval for writing stuff down

n_begin     =   500

def run_moldy(N, save = False):
    
    if release:
        cont_type = 'cont_release'
    elif not release:
        cont_type = 'cont_bend'

    params              =   {'bond':bond, 'a':a, 'h':h}
    
    # DEFINE FILES
#    mdfile_read         =   get_fileName(N, 'tear_E_rebo+KC_v', v, edge)[0]  
#    mdfile, mdlogfile   =   get_fileName(N, 'tear_E_rebo+KC_v', v, edge, cont_type)[:2]    
    mdfile_read         =   get_fileName(N, 'fixTop', taito, v, edge)[0]  
    mdfile, mdlogfile   =   get_fileName(N, 'fixTop', taito, v, edge, cont_type)[:2]    
    mdrelax             =   get_fileName(N, 'fixTop', taito, v, edge, cont_type)[-1]    


    # GRAPHENE SLAB
    traj        =   PickleTrajectory(mdfile_read, 'r')
    atoms       =   traj[0]
    
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
    top         =   get_ind(atoms.positions.copy(), 'top', fixtop - 1, left)
    rend        =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), fixtop)
    
    # use initial atoms to obtain fixes
    if not release:
        atoms   =   traj[-1]
    elif release:
        atoms   =   traj[n_begin]
        
    if not release:
        cell_h  =   atoms.get_cell()[2,2]
        zmax    =   np.max(atoms.positions[top,2])
        atoms.translate([0.,0., cell_h - zmax - 10])
    
    fix_left    =   FixAtoms(indices = left)
    fix_top     =   FixAtoms(indices = top)
    
    add_kc      =   KC_potential_p(params)

    
    constraints =   []
    constraints_def =   []

    constraints.append(add_kc)
    constraints.append(fix_left)
    constraints.append(fix_top)
    
    for ind in rend:
        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
        constraints_def.append(fix_deform)
    
    if release:
        constraints_simul   =   constraints[:]
        for const in constraints_def:
            constraints.append(const)
        constraints_init    =   constraints
        
    elif not release:
        for const in constraints_def:
            constraints.append(const)
        constraints_simul   =   constraints
        
   
    atoms.set_constraint(constraints_init)
    
    #if not release:
    #    for ind in rend:
    #        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
    #        constraints.append(fix_deform)
    #if release and T != 0:
    #    for ind in rend:
    #        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
    #        constraints_init.append(fix_deform)
    
    
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
    if save:    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    else:       traj    =   None
    
    view(atoms)
    
    # FIX 
    #if T != 0:
    #    constraints_therm   =   []
    #    for const in constraints:
    #        constraints_therm.append(const)
    #    for const in constraints_init:
    #        constraints_therm.append(const)
    
    #    atoms.set_constraint(constraints_therm)
    #else: atoms.set_constraint(constraints)
        
    
    
    
    if release:
        dyn     =   BFGS(atoms, trajectory = mdrelax)
        dyn.run(fmax=0.05)
        if T == 0:
            del atoms.constraints
            atoms.set_constraint(constraints_simul)
        
        
    if T != 0:
        # put initial MaxwellBoltzmann velocity distribution
        mbd(atoms, T*units.kB)
        
    
    # DYNAMICS
    dyn     =   Langevin(atoms, dt*units.fs, T*units.kB, fric)
    n       =   0
    
    header  =   '#t [fs], d [Angstrom], epot_tot [eV], ekin_tot [eV] \n'
    log_f   =   open(mdlogfile, 'w')
    log_f.write(header)            
    log_f.close()
    
    print 'Start the dynamics for N = %i' %N
    
    for i in range(0, M):
        

        if release:
            if T != 0.:
                if i*dt == tau:
                    # ensure empty constraints
                    del atoms.constraints
                    # add the desired constraints
                    atoms.set_constraint(constraints_simul)
            dyn.run(1)
        
        elif not release:
            if T != 0.:
                if tau <= i*dt:
                    for ind in rend:
                        atoms[ind].position[2] -= dz 
                
            elif T == 0:
                for ind in rend:
                    atoms[ind].position[2] -= dz 
            
            dyn.run(1)
                
        
        if i%interval == 0:
            
            epot, ekin  =   saveAndPrint(atoms, traj, False)[:2]
            
            if T != 0:
                if tau < i*dt:  hw   =   i*dz - tau*v
                else: hw    =   0
            else:   hw      = i*dz
            
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
                  

            n += 1
            
        
        if save and T != 0 and i*dt == tau:
            log_f   =   open(mdlogfile, 'a')
            log_f.write('# Thermalization complete. ' +  '\n')
            log_f.close()
    
        
        if 1e2 <= M:    
            if i%(int(M/100)) == 0: print 'ready = %.1f' %(i/(int(M/100))) + '%' 


run_moldy(N, True)
    