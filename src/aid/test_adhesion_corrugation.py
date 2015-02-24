'''
Created on 27.10.2014

@author: tohekorh
'''
'''
Created on 8.10.2014

@author: tohekorh
'''
from ase.constraints import FixAtoms, FixedLine
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS
from ase.io.trajectory import PickleTrajectory
from ase.visualize import view

from moldy.atom_groups import get_mask
from LJ_potential_constraint import LJ_potential
from KC_potential_constraint import KC_potential
from KC_parallel import KC_potential_p

from help import make_graphene_slab, get_save_atoms, find_layers

import matplotlib.pyplot as plt
import numpy as np

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 

length      =   2*1             # slab length has to be integer*2
width       =   1               # slab width
N           =   2

path        =   '/space/tohekorh/BendAndSlide/test_KC/'


def get_adhesion_energy(atoms, hmax, bottom, top, indent, m):
    
    zmax        =   np.amax(atoms.positions[bottom][:,2])
    natoms      =   0

    for i in range(len(top)):
        if top[i]: natoms += 1
    
    def get_epot(z):
        
        new_pos     =   atoms.positions.copy()
        for iat in range(len(atoms)):
            if top[iat]:
                new_pos[iat][2] = z
        
        atoms.positions =   new_pos
        
        e   = atoms.get_potential_energy()/natoms
        return e  

    # Start to move the top layer in z direction
    zrange  =   np.linspace(h - .7, h + hmax, m)
    adh_pot =   np.zeros((len(zrange), 2))

    traj    =   PickleTrajectory(path + \
                'trajectories/adhesion_trajectory_%s.traj' %(indent), "w", atoms)
    
    # Here we lift the top layer:
    for i, z in enumerate(zrange):
        traj.write(get_save_atoms(atoms))
        adh_pot[i]  =   [z, get_epot(zmax + z)]
        
    np.savetxt(path + 'datas/adhesion_data_%s.data' %(indent), adh_pot)
    return adh_pot


def get_optimal_h(atoms, bottom, top, natoms, dyn = False):
    
    # This find the optimal h - when the top is sliding:
    
    if not dyn:
        from scipy.optimize import fmin
        pos_init    =   atoms.positions.copy()
        
        zmax        =   np.amax(atoms.positions[bottom][:,2])
        
        def get_epot(z):
            
            new_pos     =   pos_init
            for iat in range(len(atoms)):
                if top[iat]:
                    new_pos[iat][2] = z + zmax
            
            atoms.positions =   new_pos
            e   = atoms.get_potential_energy()/natoms
            print z, e
            return e  
        
        hmin    =   fmin(get_epot, 3.34)
        emin    =   get_epot(hmin)
        
        atoms.positions = pos_init
        print 'optimal height= %.2f and e=%.2f' %(hmin, emin) 
        return emin, hmin
    else:
        dyn         =   BFGS(atoms)
        dyn.run(fmax=0.03)
        e           =   atoms.get_potential_energy()/natoms
        layers      =   find_layers(atoms.positions)[0]
        hmin        =   layers[1] - layers[0]
        return e, hmin
        
def get_corrugation_energy(atoms, constraints, bond, bottom, top, indent, m):
    
    xset        =   np.linspace(0., 3*bond, m) 
    atoms_init  =   atoms.copy()
    
    natoms      =   0
    for i in range(len(top)):
        if top[i]: 
            natoms += 1
            fix_l   = FixedLine(i, [0., 0., 1.])
            constraints.append(fix_l)
        
    atoms.set_constraint(constraints)
    
    def get_epot(x):
        
        new_pos     =   atoms_init.positions.copy()
        for iat in range(len(atoms)):
            if top[iat]:
                new_pos[iat][0] += x 
        
        atoms.positions =   new_pos
        e, hmin         =   get_optimal_h(atoms, bottom, top, natoms, False)
        #print  x, e        
        
        return e, hmin  

    # Start to move the top layer in x direction
    corr_pot            =   np.zeros((len(xset), 3))
    traj                =   PickleTrajectory(path + \
                            'trajectories/corrugation_trajectory_%s.traj' %(indent), "w", atoms)
     
    for i, x in enumerate(xset):
        traj.write(get_save_atoms(atoms))
        e, hmin         =   get_epot(x)
        corr_pot[i]     =   [x, hmin, e]
    
    np.savetxt(path + 'datas/corrugation_data_%s.data' %(indent), corr_pot)
    return corr_pot

def plot_adhesion():
    fig         =   plt.figure()
    ax          =   fig.add_subplot(111)
    datas       =   {'rebo_KC':     np.loadtxt(path + 'datas/adhesion_data_rebo_KC.data'),
                     'rebo_KC_iaS': np.loadtxt(path + 'datas/adhesion_data_rebo_KC_iaS.data'),
                     'rebo_KC_p':   np.loadtxt(path + 'datas/adhesion_data_rebo_KC_p.data'),
                     'rebo_KC_iaS_p':np.loadtxt(path+ 'datas/adhesion_data_rebo_KC_iaS_p.data'),
                     'rebo_lj':     np.loadtxt(path + 'datas/adhesion_data_rebo_lj.data'),
                     'airebo':      np.loadtxt(path + 'datas/adhesion_data_airebo.data')}
    
    
    for indent in datas:
        
        adh_pot         =   datas[indent]
        adh_pot[:,1]    =   adh_pot[:,1] - np.min(adh_pot[:,1])   
        hmin            =   adh_pot[np.where(adh_pot[:,1] == np.min(adh_pot[:,1]))[0][0], 0]
        ax.plot(adh_pot[:,0], adh_pot[:,1], label = indent)
        ax.scatter(hmin, 0)
    
    ax.set_title('Adhesion energy, per atom')
    ax.set_xlabel('height, Angst')
    ax.set_ylabel('Pot. E, eV')
    plt.legend(frameon = False)
    #plt.savefig(path + 'pictures/Adhesion_energy.svg')
    plt.show()    

def plot_corrugation():
    fig         =   plt.figure()
    ax          =   fig.add_subplot(111)
    datas       =   {'rebo_KC':     np.loadtxt(path + 'datas/corrugation_data_rebo_KC.data'),
                     'rebo_KC_iaS': np.loadtxt(path + 'datas/corrugation_data_rebo_KC_iaS.data'),
                     'rebo_KC_p':   np.loadtxt(path + 'datas/corrugation_data_rebo_KC_p.data'),
                     'rebo_KC_iaS_p':np.loadtxt(path+ 'datas/corrugation_data_rebo_KC_iaS_p.data'),
                     'rebo_lj':     np.loadtxt(path + 'datas/corrugation_data_rebo_lj.data'),
                     'airebo':      np.loadtxt(path + 'datas/corrugation_data_airebo.data')}
    
    for indent in datas:
        
        corr_pot         =   datas[indent]
        corr_pot[:,2]    =   corr_pot[:,2] - corr_pot[0,2]   
        ax.plot(corr_pot[:,0], corr_pot[:,2], '-o', label = indent)
        
        # Separation(x)
        #xmin             =   corr_pot[np.where(corr_pot[:,2] == np.min(corr_pot[:,2]))[0][0], 0]
        #ax2 = ax.twinx()
        #ax2.plot(corr_pot[:,0], corr_pot[:,1], '-D', color = 'red',label = 'hmin')
        #ax.scatter(xmin, 0)
    
    ax.set_title('Corrugation energy, per atom')
    ax.set_xlabel('x, Angst')
    ax.set_ylabel('Pot. E, eV')
    plt.legend(frameon = False)
    #plt.savefig(path + 'pictures/corrugation_energy.svg')
    plt.show()    
    
def get_adhesion_LJ(zset, CperArea):
    
    ecc     =   0.002843732471143
    sigmacc =   3.4
    adh     =   np.zeros((len(zset), 2))

    for i, z in enumerate(zset):
        adh[i,0]  =   z
        adh[i,1]  =   2./5*np.pi*CperArea*ecc*(2*(sigmacc**6/z**5)**2 - 5*(sigmacc**3/z**2)**2)
        #, \
        #8.*ecc*CperArea*np.pi*(sigmacc**12/z**11 - sigmacc**6/z**5)
    return adh
     
        
def corrugationAndAdhesion(params):
    
    bond        =   params['bond']
    a           =   np.sqrt(3)*bond # 2.462
    h           =   params['h']
    acc         =   params['acc']
    width       =   params['width']
    length      =   params['length'] 
       
    # Carbon atoms per unit area
    CperArea    =   (a**2*np.sqrt(3)/4)**(-1)
    
    # Initial atoms graphite:
    #atoms_init              =   make_graphene_slab(a,h,width,length,N, (True, True, False))[3]
    atoms_init              =   make_graphene_slab(a,h, width,length, N, \
                                                   edge_type = 'arm', h_pass = False)[3]
    
    
    atoms_init.set_pbc((True, True, False))
    atoms_init.set_cell([3*bond, a, 15])
    atoms_init.center()
    
    view(atoms_init)
    # Save something:
    params['ncores']        =   2
    params['positions']     =   atoms_init.positions
    params['cell']          =   atoms_init.get_cell().diagonal()
    params['pbc']           =   atoms_init.get_pbc()
    
    # This is the interaction distance in Angstrom:
    params['chemical_symbols']  =   atoms_init.get_chemical_symbols()
    
    # This controls how far the upper layer is pulled:
    hmax                    =   (params['cell'][2] - h)/2.1
    
    # FIX
    # Certain fixes are imposed. Note the KC and LJ - potential are as constraints
    # added to the atom system. The bottom layer is held fixed:
    top                 =   get_mask(atoms_init.positions.copy(), 'top', 1, h)
    
    bottom              =   np.logical_not(top)
    fix_bot             =   FixAtoms(mask = bottom)
    
    # The constraints for LJ nad KC - potentials:
    params['ia_dist']       =   14
    add_KC              =   KC_potential(params)
    add_KC_p            =   KC_potential_p(params)
    params['ia_dist']   =   10.
    add_KC_iaS          =   KC_potential(params)
    add_KC_iaS_p        =   KC_potential_p(params)
    
    params['ia_dist']   =   14.
    add_LJ              =   LJ_potential(params)
    
    # Different constraint sets for different calculations:
    constraint_fb_kc_e  =   [fix_bot, add_KC]
    constraint_fb_kc_e_iaS  =   [fix_bot, add_KC_iaS]
    constraint_fb_kc_e_p  =   [fix_bot, add_KC_p]
    constraint_fb_kc_e_iaS_p  =   [fix_bot, add_KC_iaS_p]
    
    constraint_fb_lj_e  =   [fix_bot, add_LJ]
    constraint_fb       =   [fix_bot]
    # END FIX
    
    
    # DEF CALC AND RELAX
    # Define proper calculator rebo when LJ or KC and airebe else:
    # Note! The potential files (CH.airebo etc.) have to be set as Enviromental variables
    # in order for lammps to find them!
    params_rebo         =   {'pair_style':'rebo',
                             'pair_coeff':['* * CH.airebo C'],
                             'mass'      :['* 12.01'],
                             'units'     :'metal', 
                             'boundary'  :'p p f'}    
    params_airebo       =   {'pair_style':'airebo 3.0',
                             'pair_coeff':['* * CH.airebo C'],
                             'mass'      :['* 12.01'],
                             'units'     :'metal', 
                             'boundary'  :'p p f'}    

    
    constraint_param_sets   = [['rebo_KC', constraint_fb_kc_e, params_rebo],
                               ['rebo_KC_iaS', constraint_fb_kc_e_iaS, params_rebo],
                               ['rebo_KC_p', constraint_fb_kc_e_p, params_rebo],
                               ['rebo_KC_iaS_p', constraint_fb_kc_e_iaS_p, params_rebo],
                               ['rebo_lj', constraint_fb_lj_e, params_rebo], 
                               ['airebo',  constraint_fb, params_airebo]] 
    
    data_adh    =   {}
    data_corr   =   {}
    
    # Loop over three different methods:
    for const_params in constraint_param_sets: 
        
        # Name for the method:
        indent          =   const_params[0]
        print indent
        
        if indent != 'LJ':
            constraints =   const_params[1]
            parameters  =   const_params[2]
        
            atoms       =   atoms_init.copy()
            init_posits =   atoms.positions.copy()
            
            # Caculator:
            calc        =   LAMMPS(parameters=parameters) #, files=['lammps.data'])
            atoms.set_calculator(calc)
            atoms.set_constraint(constraints)
            
            '''
            # Old tests:
            new_pos     =   atoms_init.positions.copy()
            for iat in range(len(atoms)):
                if top[iat]:
                    new_pos[iat][2] += new_pos[iat][0]*(-0.3) 
                    new_pos[iat][0] += bond/2 
            atoms.positions =   new_pos
            '''
        
            #view(atoms)
            dyn         =   BFGS(atoms, trajectory=path + '/bfgs_bilayer.traj')
            dyn.run(fmax=0.05)
            
            
            # SLAB IS RELAXED
            #
            
            print 'Adhesion'
            
            
            data_adh[indent]    =   get_adhesion_energy(atoms, hmax, \
                                                        bottom, top, indent, acc)
            
            print 'run_moldy'
            
            atoms.positions     =   init_posits
            print 'Corrugation'
            
            data_corr[indent]   =   get_corrugation_energy(atoms, constraints, \
                                                           bond, bottom, top, indent, acc)
            
            
        else:
            data_adh['LJ']      =   get_adhesion_LJ(data_adh['rebo_KC'][:,0], CperArea) 
    
    #PLOT and save
    plot_adhesion()
    plot_corrugation()
    
       
    


params  =   {'bond':bond, 'a':a, 'h':h, 'acc':196, 'width': width, 'length':length}  
    
corrugationAndAdhesion(params) 
    
