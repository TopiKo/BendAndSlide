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
from aid.my_constraint import add_adhesion
from aid.help import find_layers
from ase.visualize import view 

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 
dt          =   2               #units: fs
eps_dict    =   {1:[2.82, ''], 2:[45.44, '_Topi']}
idx_epsCC   =   2 
length      =   2*20            # slab length has to be integer*2
width       =   1               # slab width

# SIMULATION PARAMS
M           =   int(1e4)        # number of moldy steps
d           =   3*h            # maximum separation
dz          =   d/M
dt          =   2               # fs
T           =   0.              # temperature
interval    =   10              # interval for writing stuff down
epsCC, eps_str \
            = eps_dict[idx_epsCC]  
    

params      =   {'bond':bond, 'a':a, 'h':h}

def runMoldy(N, save = False):
    
    # DEFINE FILES
    mdfile, mdlogfile = get_fileName(N, epsCC, 'tear_E')  
    
    # GRAPHENE SLAB
    atoms       =   make_graphene_slab(a,h,width,length,N, passivate = True)[3]
    
    # FIX
    constraints =   []
    
    zset        =   find_layers(atoms.positions.copy())[0]
    left        =   get_mask(atoms.positions.copy(), 'left', 2, bond)
    rend        =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols())
    
    fix_left    =   FixAtoms(mask = left)
    
    add_bend    =   add_adhesion(params, np.max(zset))
    
    for ind in rend:
        fix_deform  =   FixedPlane(ind, (0., 0., 1.))
        constraints.append(fix_deform)
    
    constraints.append(fix_left)
    constraints.append(add_bend)
    # END FIX
    
    # CALCULATOR LAMMPS 
    parameters = {'pair_style':'airebo 3.0',
                  'pair_coeff':['* * CH.airebo_Topi C H'],
                  'mass'      :['* 12.0'],
                  'units'     :'metal', 
                  'boundary'  :'f p f'}
    
    calc = LAMMPS(parameters=parameters) 
    atoms.set_calculator(calc)
    # END CALCULATOR
    
    
    
    # TRAJECTORY
    if save:    traj    =   PickleTrajectory(mdfile, 'w', atoms)
    else:       traj    =   None
    
    data = np.zeros((M/interval, 5))
    
    
    # RELAX
    atoms.set_constraint(add_bend)
    dyn = BFGS(atoms)
    dyn.run(fmax=0.05)
    
    # FIX AFTER RELAXATION
    atoms.set_constraint(constraints)
    
    # DYNAMICS
    dyn = Langevin(atoms, dt*units.fs, T*units.kB, 0.002)
    n = 0
    print 'Start the dynamics for N = %i' %N
    
    for i in range(0, M):
        for ind in rend:
            atoms[ind].position[2] -= dz 
        dyn.run(1)
        
        if i%interval == 0:
            epot, ekin = saveAndPrint(atoms, traj, False)[:2]
            data[n] = [i*dt, i*dz, epot, ekin, epot + ekin]
            n += 1
            
        if i%(int(M/100)) == 0: print 'ready = %.1f' %(i/(int(M/100))) + '%' 
    
    if save:
        header = 't [fs], d [Angstrom], epot/Atom [eV], ekin/Atom [eV], etot/Atom [eV]'
        np.savetxt(mdlogfile, data, header = header)  

#adh_pot    =   np.loadtxt('adh_potential.gz')   
#print adh_pot
#runMoldy(3)
for N in range(3, 15):    
    runMoldy(N, True)   
    
