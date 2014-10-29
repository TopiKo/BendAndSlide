'''
Created on 8.10.2014

@author: tohekorh
'''
from help import make_graphene_slab
import numpy as np
from moldy.atom_groups import get_mask
from ase.constraints import FixAtoms
import matplotlib.pyplot as plt
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS
#from scipy.optimize import minimize

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 

length      =   2*4             # slab length has to be integer*2
width       =   1               # slab width
N           =   2


def find_adhesion_potential(params):
    
    bond    =   params['bond']
    a       =   np.sqrt(3)*bond # 2.462
    h       =   params['h']
    CperArea=   (a**2*np.sqrt(3)/4)**(-1)
    
    atoms   =   make_graphene_slab(a,h,width,length,N, (True, True, False))[3]
    
    # FIX
    constraints =   []
    top         =   get_mask(atoms.positions.copy(), 'top', 1, h)
    
    bottom      =   np.logical_not(top)
    fix_bot     =   FixAtoms(mask = bottom)
    constraints.append(fix_bot)
    # END FIX
    
    # DEF CALC AND RELAX
    parameters = {'pair_style':'airebo 3.0',
                  'pair_coeff':['* * CH.airebo C'],
                  'mass'      :['* 12.01'],
                  'units'     :'metal', 
                  'boundary'  :'p p f'}
    
    calc    =   LAMMPS(parameters=parameters) #, files=['lammps.data'])
    atoms.set_calculator(calc)
    
    dyn     =   BFGS(atoms)
    dyn.run(fmax=0.05)
    # SLAB IS RELAXED
    
    atoms.set_constraint(constraints)
    zmax    =   np.amax(atoms.positions[bottom][:,2])
    
    natoms  =   0
    for i in range(len(top)):
        if top[i]: natoms += 1
    
    def get_epot(z):
        
        new_pos     =   atoms.positions.copy()
        for iat in range(len(atoms)):
            if top[iat]:
                new_pos[iat][2] = z
        
        atoms.positions =   new_pos
        return atoms.get_potential_energy()/natoms  

    
    def lj(z):
        
        ecc     =   0.002843732471143
        sigmacc =   3.4
        return 2./5*np.pi*CperArea*ecc*(2*(sigmacc**6/z**5)**2 - 5*(sigmacc**3/z**2)**2), \
            8.*ecc*CperArea*np.pi*(sigmacc**12/z**11 - sigmacc**6/z**5)
    
    # Start to move the top layer in z direction
    zrange  =   np.linspace(h - .7, h + 8, 100)
    adh_pot =   np.zeros((len(zrange), 2))
    
    for i, z in enumerate(zrange):
        adh_pot[i]  =   [z, get_epot(zmax + z)]
    
    adh_pot[:,1]    =   adh_pot[:,1] - np.min(adh_pot[:,1])   
    hmin            =   adh_pot[np.where(adh_pot[:,1] == np.min(adh_pot[:,1]))[0][0], 0]
    
    np.savetxt('adh_potential.gz', adh_pot, fmt='%.12f')
        
    fig         =   plt.figure()
    ax          =   fig.add_subplot(111)
    
    ax.plot(adh_pot[:,0], adh_pot[:,1], label = 'lamps')
    ax.plot(zrange, lj(zrange)[0] - np.min(lj(zrange)[0]), label = 'lj')
    ax.plot(zrange, lj(zrange)[1], label = 'lj')
    
    ax.scatter(hmin, 0)

    ax.set_title('Adhesion energy')
    ax.set_xlabel('height, Angst')
    ax.set_ylabel('Pot. E, eV')
    plt.legend(frameon = False)
    plt.savefig('Adhesion_energy.svg')
    plt.show()
    
params      =   {'bond':bond, 'a':a, 'h':h}  
find_adhesion_potential(params) 
    