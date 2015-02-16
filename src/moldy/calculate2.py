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
from atom_groups import get_mask, get_ind
from ase.md.langevin import Langevin
#from aid.my_constraint import add_adhesion, KC_potential
from aid.KC_potential_constraint import KC_potential
from aid.LJ_potential_constraint import add_adhesion
#from aid.help import find_layers
from ase.visualize import view 
import sys

#N, v, M     =   int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])

N,v,M   =   5, 1., 10000

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 
dt          =   2               # units: fs
length      =   4*8            # slab length has to be integer*2
width       =   1               # slab width
fix_top     =   0               # 

# SIMULATION PARAMS
#M           =   int(1e4)        # number of moldy steps
dt          =   2               # fs
#v           =   1.#/33            # Angst/ps

d           =   6*h     # M*dt*v      # maximum separation
dz          =   dt*v/1000    # d/M  

T           =   0.              # temperature
interval    =   10              # interval for writing stuff down
    


def run_moldy(N, save = False):
    
    params      =   {'bond':bond, 'a':a, 'h':h}
    
    # DEFINE FILES
    mdfile, mdlogfile, mdrelax = get_fileName(N, 'tear_E_rebo+KC_v', v)  
    
    
    # GRAPHENE SLAB
    atoms               =   make_graphene_slab(a,h,width,length,N, passivate = True)[3]
    
    
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   14
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
        constraints.append(fix_deform)
    
    constraints.append(fix_left)
    constraints.append(add_adh)
    constraints.append(add_kc)
    # END FIX
    
    # CALCULATOR LAMMPS 
    parameters = {'pair_style':'rebo',
                  'pair_coeff':['* * CH.airebo C H'],
                  'mass'      :['1 12.0', '2 1.0'],
                  'units'     :'metal', 
                  'boundary'  :'f p f'}
    
    calc = LAMMPS(parameters=parameters) 
    atoms.set_calculator(calc)
    # END CALCULATOR
    
    
    
    # TRAJECTORY
    if save:    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    else:       traj    =   None
    
    data = np.zeros((M/interval, 7))
    
    #view(atoms)
    
    # RELAX
    atoms.set_constraint([add_kc, add_adh])
    dyn = BFGS(atoms, trajectory = mdrelax)
    dyn.run(fmax=0.05)
    
    # FIX AFTER RELAXATION
    atoms.set_constraint(constraints)
    
    # DYNAMICS
    dyn     =   Langevin(atoms, dt*units.fs, T*units.kB, 0.002)
    n       =   0
    header  =   '#t [fs], d [Angstrom], epot_tot [eV], ekin_tot [eV], vmax, pmax, pmax/vmax \n'
    log_f   =   open(mdlogfile, 'w')
    log_f.write(header)            
    log_f.close()

    print 'Start the dynamics for N = %i' %N
    
    for i in range(0, M):
        for ind in rend:
            atoms[ind].position[2] -= dz 
        dyn.run(1)
        
        if i%interval == 0:
            vmax = 0
            pmax = 0
            
            pa  =   atoms.arrays.get('momenta')
            va  =   atoms.get_velocities()
            
            for kk in range(len(va)):
                vv  =   va[kk]
                p   =   pa[kk]
                if np.linalg.norm(vv) > vmax: 
                    vmax  =  np.linalg.norm(vv) 
                    pmax  =  np.linalg.norm(p) 

            np.savetxt('/space/tohekorh/BendAndSlide/files/v_i=%i' %i, va)
            epot, ekin = saveAndPrint(atoms, traj, False)[:2]
            
            data[n] = [i*dt, i*dz, epot, ekin, vmax, pmax, pmax/vmax]
            
            if save:
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
                #np.savetxt(mdlogfile, data, header = header)  

            n += 1
            
        if i%(int(M/100)) == 0: print 'ready = %.1f' %(i/(int(M/100))) + '%' 
    


run_moldy(N, True)
#for N in range(3, 15):    
#    run_moldy(N, True)   
    
