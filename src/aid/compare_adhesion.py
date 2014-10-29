'''
Created on 27.10.2014

@author: tohekorh
'''
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
from my_constraint import add_adhesion, LJ_potential, KC_potential
#from scipy.optimize import minimize
from ase.visualize import view
from aid.help import find_layers
# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 

length      =   2*2             # slab length has to be integer*2
width       =   1               # slab width
N           =   2

path        =   '/space/tohekorh/BendAndSlide/test_KC/'

def find_adhesion_potential(params):
    
    bond    =   params['bond']
    a       =   np.sqrt(3)*bond # 2.462
    h       =   params['h']
    
    CperArea=   (a**2*np.sqrt(3)/4)**(-1)
    
    atoms_init              =   make_graphene_slab(a,h,width,length,N, (True, True, False))[3]

    params['positions']     =   atoms_init.positions
    params['cell']          =   atoms_init.get_cell().diagonal()
    params['pbc']           =   atoms_init.get_pbc()

    hmax                    =   (params['cell'][2] - h)/2.1
    
    # FIX
    top             =   get_mask(atoms_init.positions.copy(), 'top', 1, h)
    
    #top_h       =   atoms_init.positions[:,2].max()
    #layer_inds  =   find_layers(atoms_init.positions.copy())[1]
    #add_adh     =   add_adhesion(params, layer_inds, top_h)
    #add_adh     =   add_adhesion(params, layer_inds, top_h)

    bottom      =   np.logical_not(top)
    fix_bot     =   FixAtoms(mask = bottom)
    add_adh_e   =   LJ_potential(params)
    add_KC      =   KC_potential(params)
    constraint_fb_kc_e  = [fix_bot, add_KC]
    constraint_fb_lj_e  = [fix_bot, add_adh_e]
    constraint_fb       = [fix_bot]
    #constraint_sets.append([fix_bot, add_adh])

    
    # END FIX
    
    # DEF CALC AND RELAX
    params_rebo = {'pair_style':'rebo',
                  'pair_coeff':['* * CH.airebo C'],
                  'mass'      :['* 12.01'],
                  'units'     :'metal', 
                  'boundary'  :'p p f'}    
    params_airebo = {'pair_style':'airebo 3.0',
                  'pair_coeff':['* * CH.airebo C'],
                  'mass'      :['* 12.01'],
                  'units'     :'metal', 
                  'boundary'  :'p p f'}    

    fig         =   plt.figure()
    ax          =   fig.add_subplot(111)
        
    constraint_param_sets   = [[constraint_fb_kc_e, params_rebo, 'rebo_KC'],
                               [constraint_fb_lj_e, params_rebo, 'rebo_lj'], 
                               [constraint_fb, params_airebo, 'airebo']]
    
    for const_params in constraint_param_sets: 
        
        constraints =   const_params[0]
        parameters  =   const_params[1]
        indent      =   const_params[2]
        
        atoms       =   atoms_init.copy()

        calc        =   LAMMPS(parameters=parameters) #, files=['lammps.data'])
        atoms.set_calculator(calc)
        atoms.set_constraint(constraints)
        
        dyn         =   BFGS(atoms)
        dyn.run(fmax=0.05)
        
        # SLAB IS RELAXED
        view(atoms)
        
        #print constraints
        
        
        zmax        =   np.amax(atoms.positions[bottom][:,2])
    
        natoms  =   0
        for i in range(len(top)):
            if top[i]: natoms += 1
        
        def get_epot(z):
            
            new_pos     =   atoms.positions.copy()
            for iat in range(len(atoms)):
                if top[iat]:
                    new_pos[iat][2] = z
            
            atoms.positions =   new_pos
            
            e   = atoms.get_potential_energy()/natoms
            print z - zmax, e 
            return e  
    
        
        def lj(z):
            
            ecc     =   0.002843732471143
            sigmacc =   3.4
            return 2./5*np.pi*CperArea*ecc*(2*(sigmacc**6/z**5)**2 - 5*(sigmacc**3/z**2)**2), \
                8.*ecc*CperArea*np.pi*(sigmacc**12/z**11 - sigmacc**6/z**5)
        
        
        # Start to move the top layer in z direction
        zrange  =   np.linspace(h - .7, h + hmax, 1000)
        adh_pot =   np.zeros((len(zrange), 2))
        
        for i, z in enumerate(zrange):
            #view(atoms)
            adh_pot[i]  =   [z, get_epot(zmax + z)]
        
        adh_pot[:,1]    =   adh_pot[:,1] - np.min(adh_pot[:,1])   
        hmin            =   adh_pot[np.where(adh_pot[:,1] == np.min(adh_pot[:,1]))[0][0], 0]
            
        np.savetxt(path + 'adhesion_data_%s.data' %(indent), adh_pot)
        ax.plot(adh_pot[:,0], adh_pot[:,1], label = indent)
        ax.scatter(hmin, 0)
    
    ax.plot(zrange, lj(zrange)[0] - np.min(lj(zrange)[0]), label = 'lj')
        
    
    ax.set_title('Adhesion energy, per atom')
    ax.set_xlabel('height, Angst')
    ax.set_ylabel('Pot. E, eV')
    plt.legend(frameon = False)
    plt.savefig(path + 'Adhesion_energy_aireb_reboLj.svg')
    plt.show()
    
params      =   {'bond':bond, 'a':a, 'h':h}  
find_adhesion_potential(params) 
    