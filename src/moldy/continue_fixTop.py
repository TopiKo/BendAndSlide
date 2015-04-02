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
from aid.KC_potential_constraint import KC_potential
from aid.KC_parallel import KC_potential_p
from ase.visualize import view 
import sys

N, v, M, edge, release, ncores   =   int(sys.argv[1]), float(sys.argv[2]), \
                                    int(sys.argv[3]), sys.argv[4], \
                                    sys.argv[5] in ['True', 'true', 1], int(sys.argv[6]) 

#N, v, M, edge, release, ncores   =   15, 1., 10000, 'arm', False, 2
print sys.argv[5]
print bool(sys.argv[5])
  

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

n_begin     =   -1

def run_moldy(N, save = False):
    
    if release:
        cont_type = 'cont_release'
    elif not release:
        cont_type = 'cont_bend'

    
    print release, cont_type    
    params              =   {'bond':bond, 'a':a, 'h':h}
    
    # DEFINE FILES
#    mdfile_read         =   get_fileName(N, 'tear_E_rebo+KC_v', v, edge)[0]  
#    mdfile, mdlogfile   =   get_fileName(N, 'tear_E_rebo+KC_v', v, edge, cont_type)[:2]    
    mdfile_read         =   get_fileName(N, 'fixTop', v, edge)[0]  
    mdfile, mdlogfile   =   get_fileName(N, 'fixTop', v, edge, cont_type)[:2]    


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
    constraints         =   []
    constraints_init    =   []
    
    left        =   get_ind(atoms.positions.copy(), 'left', 2, bond)
    top         =   get_ind(atoms.positions.copy(), 'top', fixtop - 1, left)
    rend        =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), fixtop)
    
    # use initial atoms to obtain fixes
    if not release:
        atoms       =   traj[-1]
    elif release:
        atoms       =   traj[n_begin]
        
    cell_h      =   atoms.get_cell()[2,2]
    zmax        =   np.max(atoms.positions[top,2])
    
    if not release:
        atoms.translate([0.,0., cell_h - zmax - 10])
    
    fix_left    =   FixAtoms(indices = left)
    fix_top     =   FixAtoms(indices = top)
    
    add_kc      =   KC_potential_p(params)

    
    if not release:
        for ind in rend:
            fix_deform  =   FixedPlane(ind, (0., 0., 1.))
            constraints.append(fix_deform)
    if release and T != 0:
        for ind in rend:
            fix_deform  =   FixedPlane(ind, (0., 0., 1.))
            constraints_init.append(fix_deform)
    
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
    
    
    
    # TRAJECTORY
    if save:    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    else:       traj    =   None
    
    view(atoms)
    
    # FIX 
    if T != 0:
        constraints_therm   =   []
        for const in constraints:
            constraints_therm.append(const)
        for const in constraints_init:
            constraints_therm.append(const)
    
    atoms.set_constraint(constraints_therm)
    
    # DYNAMICS
    dyn     =   Langevin(atoms, dt*units.fs, T*units.kB, fric)
    n       =   0
    
    header  =   '#t [fs], d [Angstrom], epot_tot [eV], ekin_tot [eV] \n'
    log_f   =   open(mdlogfile, 'w')
    log_f.write(header)            
    log_f.close()
    
    print 'Start the dynamics for N = %i' %N
    
    
    
    for i in range(0, M):
        

        if not release:
            if T != 0.:
                if i*dt < tau:
                    dyn.run(1)
                elif tau <= i*dt:
                    for ind in rend:
                        atoms[ind].position[2] -= dz 
                    dyn.run(1)
            elif T == 0:
                for ind in rend:
                    atoms[ind].position[2] -= dz 
                dyn.run(1)
        elif release:
            if T != 0.:
                if i*dt == tau:
                    # ensure empty constraints
                    del atoms.constraints
                    # add the desired constraints
                    atoms.set_constraint(constraints)
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
    