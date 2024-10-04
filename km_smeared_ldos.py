# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:56:18 2023

@author: kmilton
"""

# Program to plot smeared ldos plots
# Thomas Durrant (thomas.durrant.14@ucl.ac.uk
# Last modified 24/02/23

import time
import json
#from tkinter import E
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpi4py import MPI
import os
from os.path import exists
from glob import glob
from collections import defaultdict
import ase.io
from ase.io.cube import read_cube_data
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
#sys.path.append('/mnt/lustre/a2fs-work2/work/e05/e05/ucapklm/scripts/ldos_band_align/combination_tom_k/account_vacuum_energy/re_attempt_fermi_removed')
#import ldos_align_funcs as laf


t_start = time.time()

# MPI initialisation (Parallel libary)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    print('Using {0} processors'.format(size))

# Unit conversion factors
hart2eV = 27.21138602
bohr = 5.2917721092e-11
ang2Bohr = 1e-10/bohr
bohr2Ang = 1.0/ang2Bohr



def get_band_edges(pdos_file, vac):

    eigenvals, ldos, fermi = load_pdos_file(pdos_file, vac)

    homo = -np.inf
    lumo = np.inf

    ref=eigenvals-fermi

    for eng in ref:
        if eng <= 0.5 and eng > homo:
            homo = eng+fermi
        if eng > 0.5 and eng < lumo:
            lumo = eng+fermi

    return homo, lumo


def load_pdos_file(pdos_file, vac):

    # Determine number of shells in file
    pdos = open(pdos_file, 'r')
    header = pdos.readline()
    E_fermi = float(header.split()[-2])
    E_fermi=(E_fermi*hart2eV)-vac
    header = pdos.readline()
    if 'f' in header:
        N_shell = 4
    elif 'd' in header:
        N_shell = 3
    elif 'p' in header:
        N_shell = 2
    elif 's' in header:
        N_shell = 1
    else:
        raise Exception('Header line of ldos file {0} not understood'.format(pdos_file))
    pdos.close()

    # Load in the ldos file 
    if N_shell == 1:
        i, eigenvalue, occ, s = np.genfromtxt(pdos_file, unpack=True, skip_header=2, dtype=None)
        ldos = s
    if N_shell == 2:
        i, eigenvalue, occ, s, p = np.genfromtxt(pdos_file, unpack=True, skip_header=2, dtype=None)
        ldos = s + p
    if N_shell == 3:
        i, eigenvalue, occ, s, p, d = np.genfromtxt(pdos_file, unpack=True, skip_header=2, dtype=None)
        ldos = s + p + d
    if N_shell == 4:
        i, eigenvalue, occ, s, p, d, f = np.genfromtxt(pdos_file, unpack=True, skip_header=2, dtype=None)
        ldos = s + p + d + f

    eigenvalue=(eigenvalue*hart2eV)-vac

    return eigenvalue, ldos, E_fermi

def assign_grids(structure_file, NE, Nz, min_eng, max_eng):
    if rank == 0:

        # Load in structure information
        N_atoms, symbols, x, y, z = read_atom_positions(structure_file)
       
        # Physical extent of atoms
        '''z_max, z_min = np.amax(x)+5.0, np.amin(x)-5.0 - the old amount'''
        
        z_max, z_min = np.amax(y), np.amin(y)
        
        # Asign the pdos grid
        engs = np.linspace(min_eng, max_eng, NE, endpoint=True)
        zs = np.linspace(z_min, z_max, Nz, endpoint=True)
    
        return engs, zs, z_max, z_min

    else:
        return None, None, None, None

def read_atom_positions(structure_file):
    if rank == 0:

        # Load in structure information
        symbols, x, y, z = np.genfromtxt(structure_file, unpack=True, skip_header=2, dtype=None)
        symbols = np.array(symbols, dtype=str)
        N_atoms = len(symbols)
        
        return N_atoms, symbols, x, y, z

    else:
        return None, None, None, None, None



def assign_band_edge(engs, zs, smeared_pdos, band_tol, E_homo, E_lumo):
    if rank == 0:
        Nz = len(zs)
    
        local_lumo = np.ones([Nz])*np.inf
        local_homo = np.ones([Nz])*-np.inf
        
        for i, energy in enumerate(engs):
            for j, z_pos in enumerate(zs):
                if (energy <= E_homo) and (smeared_pdos[j,i] > band_tol) and (energy > local_homo[j]):
                    local_homo[j] = energy
                if (energy >= E_lumo) and (smeared_pdos[j,i] > band_tol) and (energy < local_lumo[j]):
                    local_lumo[j] = energy
    
        return local_lumo, local_homo
    else:
        return None, None
    

def get_vacuum_potential(hartree_file):
    '''Use of Tom's electrostatic potential script, condensed into this function.
    Currently using y as the axis perpendicular to the slab i.e. the vacuum region.
    To change to the z axiz, just change the v_elec_y[0] to v_elec_z[0]'''


    hart2eV = 27.21138505

    pot_file = hartree_file
    v_elec, atoms = read_cube_data(pot_file)

    # Unpack the objects we read from file
    latt_a = atoms.cell[0,0]   # Cell vector A (Angstrom)
    latt_b = atoms.cell[1,1]   # Cell vector B (Angstrom)
    latt_c = atoms.cell[2,2]   # Cell vector C (Angstrom)
    Nx, Ny, Nz = v_elec.shape  # Number of grid points in electrostatic potential
    v_elec = v_elec*hart2eV    # Convert electrostatic potential from atomic units to volts
    y = np.linspace(0.0, latt_b, Ny)
    z = np.linspace(0.0, latt_c, Nz)
    v_elec_z = np.mean(v_elec, axis=(0,1))
    v_elec_y = np.mean(v_elec, axis=(0,2))
    vac_pot=v_elec_y[0]
    return vac_pot
 




def get_files(ldos_name,pdos_name,xyz_file):
    """
    This function is for searching through directories and finding the relevant ldos, pdos and structural files.
    Assumed that you are running this script from a master directory, with snapshots sorted in directories.
    Match allows us to make sure that only using directories where both pdos and ldos present

    ldos_name: string of ldos file name, should be a guarenteed ldos file name e.g. ldos_list1-1.pdos
    pdos_name: string of pdos file name, same principle as above e.g. pdos_k1-1.pdos
    xyz_file: accompanying xyz file for structural info

    """
    c_d=os.getcwd()
    pdos_dir=c_d+'/*/'+pdos_name
    ldos_dir=c_d+'/*/'+ldos_name
    struc_dir=c_d+'/*/'+xyz_file

    pdos_g=glob(pdos_dir,recursive=True)
    ldos_g=glob(ldos_dir,recursive=True)
    struc_g=glob(struc_dir, recursive=True)

    pdos_files=[glob(s) for s in pdos_g]
    ldos_file=[glob(s) for s in ldos_g]
    struc_file=[glob(s) for s in struc_g]

    l=[]
    for i in ldos_file:
        a=os.path.dirname(i[0])
        l.append(a)
    p=[]
    for i in pdos_files:
        b=os.path.dirname(i[0])
        p.append(b)


    match=set(l) & set(p)
    match=list(match)

    return match


def averaging(prop):
    mean=np.mean(prop,axis=0)
    std=np.std(prop,axis=0)
    return mean, std


def get_properties(match,hartree_file):

    # Init remaining values

    vac,N_atoms,engs,zs,z_min,z_max,N_atoms,symbols,x,y,z,E_homo,E_lumo,ldos,eigen_val_l,pdos,eigen_val_p=None,None, None, None,None, None, None,None, None, None,None, None, None,None, None, None,None
    

    #empty lists of all values we will be getting

    tot_engs=[]
    tot_zs=[]
    tot_z_min=[]
    tot_z_max=[]
    tot_x=[]
    tot_y=[]
    tot_z=[]
    tot_E_homo=[]
    tot_E_lumo=[]
    tot_pdos=[]
    tot_eigen_val_p=[]
    tot_eigen_val_l=[]
    tot_vacuum=[]
    av_fermi=[]

    d_ldos={}
    tot_ldos=defaultdict(list)
    check=''
    total=len(match)
    
    for j,i in enumerate(match):
        #getting to the correct directory
        os.chdir(i)
        current_directory = os.getcwd()
        last_part = os.path.basename(current_directory)
        #print(last_part)
        print(j,'/',total)

        #these lists are what we will either be producing or have aready made and want to reload, this helps reduce time when fine tuning smearing params
        prop_str=['engs.npy','zs.npy','z_min.npy','z_max.npy','N_atoms.npy','symbols.npy','x.npy','y.npy','z.npy','E_homo.npy','E_lumo.npy','ldos_f.npy','ldos_ev.npy','pdos_f.npy','pdos_ev.npy','all_ldos_attempt.json','vac.npy','loop']#added nonsense to make it loop through the ldos files
        all_props=[engs,zs,z_min,z_max,N_atoms,symbols,x,y,z,E_homo,E_lumo,ldos,eigen_val_l,pdos,eigen_val_p,vac]

        #loading values, if you want to redo values, can add an additional value to prop_str list
        if all([os.path.isfile(f) for f in prop_str]):
            print('from files')
            
            tot_engs.append(np.load(prop_str[0]))
            tot_zs.append(np.load(prop_str[1]))
            tot_z_min.append(np.load(prop_str[2]))
            tot_z_max.append(np.load(prop_str[3]))
            tot_x.append(np.load(prop_str[6]))
            tot_y.append(np.load(prop_str[7]))
            tot_z.append(np.load(prop_str[8]))
            tot_E_homo.append(np.load(prop_str[9]))
            tot_E_lumo.append(np.load(prop_str[10]))
            tot_pdos.append(np.load(prop_str[13]))
            tot_eigen_val_p.append(np.load(prop_str[14]))
            tot_eigen_val_l.append([np.load(prop_str[12])])
            #tot_vacuum.append(np.load(prop_str[15]))
            if exists('../all_ldos.json') == False: #double check that the total_ldos has been made
                print('no total ldos files')
                check='scratch'
                with open ('all_ldos_attempt.json','r') as fi:
                    d_ldos=json.load(fi)
                for key,value in d_ldos.items():
                    if key in tot_ldos:
                        tot_ldos[key].append(value)
                    else:
                        tot_ldos[key]=[value]


        else:
            print('making from scratch')
            check='scratch' #this is a marker to use later, allows us to either save or load all_ldos file

            #using previous functions to get values
            vac=get_vacuum_potential(hartree_file)
            engs, zs, z_max, z_min = assign_grids(structure_file, NE, Nz, min_eng, max_eng)
            N_atoms, symbols, x, y, z = read_atom_positions(structure_file)
            E_homo, E_lumo =  get_band_edges(pdos_file, vac)
            eigen_val_p, pdos, fermi = load_pdos_file(pdos_file, vac)
            eigen_val_l=[]
            

            #making the ldos files into a decent format, dictionary allows for us to look at the ldos of each atom

            for j in range(0,N_atoms):
                ldos_file=ldos_files.format(j+1)
                ev_l,lds, fermi=load_pdos_file(ldos_file, vac)
                d_ldos["atom{0}".format(j+1)]=lds.tolist()
            eigen_val_l.append([ev_l])
            for key,value in d_ldos.items():
                if key in tot_ldos:
                    tot_ldos[key].append(value)
                else:
                    tot_ldos[key]=[value]

            #print(eigen_val_l)
            #print(E_lumo)

            av_fermi.append(fermi)

            eigen_val_l=eigen_val_l#-vac #ALREADY SUBTRACTED THE VACUUM IN THE load_pdos_file function!!!!!
            eigen_val_p=eigen_val_p#-vac
            E_homo=E_homo#-vac
            E_lumo=E_lumo#-vac
            E_homo=E_homo#-fermi
            E_lumo=E_lumo#-fermi
            #print(eigen_val_l)
            #print(E_lumo)


            all_props=[engs,zs,z_min,z_max,N_atoms,symbols,x,y,z,E_homo,E_lumo,pdos,eigen_val_p, vac]
            prop_str=['engs','zs','z_min','z_max','N_atoms','symbols','x','y','z','E_homo','E_lumo','pdos_f','pdos_ev','vac']

            for k in range(len(all_props)):
                title=str(prop_str[k])+'.npy'
                np.save(title,all_props[k])

            outfile=open('all_ldos_attempt.json','w')
            json.dump(d_ldos, outfile)
            outfile.close()

            tot_engs.append(engs)
            tot_zs.append(zs)
            tot_z_min.append(z_min)
            tot_z_max.append(z_max)
            tot_x.append(x)
            tot_y.append(y)
            tot_z.append(z)
            tot_E_homo.append(E_homo)
            tot_E_lumo.append(E_lumo)
            tot_pdos.append(pdos)
            tot_eigen_val_p.append(eigen_val_p)
            tot_eigen_val_l.append(eigen_val_l)
            tot_vacuum.append(vac)

        N_atoms=np.load('N_atoms.npy')
        symbols=np.load('symbols.npy')


    os.chdir(os.getcwd()+'/../')
    if check=='scratch':
        out=open('all_ldos.json','w') 
        json.dump(tot_ldos,out)
        out.close()
    elif os.path.isfile('all_ldos.json') == False: #edit KM 18.09.23 to add in elif statement to make it foolproof as not working on a new example
        out=open('all_ldos.json','w')
        json.dump(tot_ldos,out)
        out.close()
    else:
        f=open('all_ldos.json','r')
        tot_ldos=json.load(f)
        f.close()


    #formatting data, allows for easier averaging


    #tot_props=[tot_engs,tot_zs,tot_z_min,tot_z_max,tot_x,tot_y,tot_z,tot_E_homo,tot_E_lumo,tot_evl, tot_eigen_val_p, tot_pdos]

    tot_engs=np.asarray(tot_engs)
    tot_zs=np.asarray(tot_zs)
    tot_z_min=np.asarray(tot_z_min)
    tot_z_max=np.asarray(tot_z_max)
    tot_x=np.asarray(tot_x)
    tot_y=np.asarray(tot_y)
    tot_z=np.asarray(tot_z)
    tot_E_homo=np.asarray(tot_E_homo)
    tot_E_lumo=np.asarray(tot_E_lumo)
    tot_eigen_val_l=np.asarray(tot_eigen_val_l)
    tot_eigen_val_p=np.asarray(tot_eigen_val_p)
    tot_pdos=np.asarray(tot_pdos)
    av_fermi=np.asarray(av_fermi)

    tot_evl=[]
    for x in tot_eigen_val_l:
        #print(x[0][0].shape)
        tot_evl.append(x[0][0])


    ldos_av=[]
    ldos_std=[]
    for k,v in tot_ldos.items():
        v=np.asarray(v)
        v_mean,v_std=averaging(v)
        ldos_av.append(v_mean)
        ldos_std.append(v_std)

    ldos_mean=np.asarray(ldos_av)
    ldos_std=np.asarray(ldos_std)



    engs_mean,engs_std=averaging(tot_engs)
    zs_mean, zs_std=averaging(tot_zs)
    z_min_mean, z_min_std=averaging(tot_z_min)
    z_max_mean,z_max_std=averaging(tot_z_max)
    x_mean, x_std=averaging(tot_x)
    y_mean,y_std=averaging(tot_y)
    z_mean, z_std=averaging(tot_z)
    E_homo_mean, E_homo_std = averaging(tot_E_homo)
    E_lumo_mean,E_lumo_std = averaging(tot_E_lumo)

    ev_p_mean,ev_p_std=averaging(tot_eigen_val_p)
    pdos_mean,pdos_std=averaging(tot_pdos)

    mean_ev_l,std_ev_l=averaging(tot_evl)

    y_std_mean=np.mean(y_std)

    ev_std_mean=np.mean(std_ev_l)

    mean_fermi,std_fermi=averaging(av_fermi)

    #ldos_mean=ldos_mean-mean_fermi
    
    
    return N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l, E_homo_mean,E_lumo_mean,z_min_mean,z_max_mean,symbols, mean_fermi, std_fermi


def km_smeared_ldos( N_atoms, engs, zs, x, y, z, sigma_E, sigma_z,av_pdos,av_ldos,ldos_ev):

    # Syncronise task across threads
    N_atoms = comm.bcast(N_atoms)
    engs = comm.bcast(engs)
    zs = comm.bcast(zs)
    x = comm.bcast(x)
    y = comm.bcast(y)
    z = comm.bcast(z)
    sigma_E = comm.bcast(sigma_E)
    sigma_z = comm.bcast(sigma_z)

    # Assign the required grid
    NE = len(engs)
    Nz = len(zs)
    smeared_pdos = np.zeros([NE,Nz])

    # Read in pdos file to discover energy range
    atom_ldos=av_pdos #pdos
    eigenvalue=ldos_ev
    N_levels = len(eigenvalue)




    for i in range(0, N_atoms):
        #print(i)
        if (i%size) == rank:
            atom_ldos=av_ldos[i]
            atom_z = y[i]

            for level in range(0, N_levels):
                engs_mesh, zs_mesh = np.meshgrid(engs, zs)
                component = 1.0/(sigma_E*sigma_z*(2*np.pi))*np.exp(-0.5*(((((zs_mesh-atom_z)**2)/sigma_z**2) + ((engs_mesh-eigenvalue[level])**2)/(sigma_E**2))))*atom_ldos[level] #i think this is just the smearing of the ldos
                smeared_pdos += component # Seems faster to do this in two steps?

    if rank == 0:
        print('\n waiting for other processors')
    t1 = time.time()

    #Collect the results
    print('Rank', rank, 'PDOS', type(smeared_pdos))
    smeared_pdos = comm.reduce(smeared_pdos)

    if rank == 0:
        print('all done :)')
        print('Reduce complete')

    return smeared_pdos



def km_band_edge(engs, zs, smeared_pdos, band_tol, E_homo, E_lumo, mean_fermi):
    #smeared_pdos=(smeared_pdos-smeared_pdos.min())/(smeared_pdos.max()-smeared_pdos.min())#KM added 03/20/2024
    if rank == 0:
        Nz = len(zs)

        print(engs)

        local_lumo = np.ones([Nz])*np.inf
        local_homo = np.ones([Nz])*-np.inf

        for i, energy in enumerate(engs):
            for j, z_pos in enumerate(zs):
                #print('energy type:',type(energy))
                #print('E_homo type:',type(E_homo))
                #print('smeared pdos type:',type(smeared_pdos))
                #print('band tol type:',type(band_tol))
                #print('local homo type:',type(local_homo[j]))
                

                #if (energy <= E_homo) and (smeared_pdos[j,i] > band_tol) and (energy > local_homo[j]):
                #    local_homo[j] = energy
                #if (energy >= E_lumo) and (smeared_pdos[j,i] > band_tol) and (energy < local_lumo[j]):
                #    local_lumo[j] = energy

                if (energy <= mean_fermi+0.5) and (smeared_pdos[j,i] > band_tol) and (energy > local_homo[j]): 
                    local_homo[j] = energy
                if (energy >= mean_fermi+0.5) and (smeared_pdos[j,i] > band_tol) and (energy < local_lumo[j]):
                    local_lumo[j] = energy
        print(smeared_pdos[j,i])
        return local_lumo, local_homo
    else:
        return None, None





def smearing_param_input(N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l):

    if os.path.exists('smeared_pdos.npy') == True:
        while True:
            user_input='y'
            #user_input = input("do you want to use previously calculated ldos with default parameters? \n Please enter 'y' or 'n': \n")
            if user_input == 'y':
                smeared_pdos=np.load('smeared_pdos.npy')
                break
            elif user_input == 'n':
                while True:
                    try:
                        E_smear=float(input('please enter your new energy smearing parameter\n'))
                        z_smear=float(input('please enter your new position smearing parameter\n'))
                        break
                    except ValueError:
                        print('Invalid input. Please enter a number!')
                smeared_pdos=km_smeared_ldos(N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, E_smear, z_smear,pdos_mean,ldos_mean,mean_ev_l)
                np.save('smeared_pdos.npy',smeared_pdos)
                break
            else:
                print("Invalid input. Please try again.")
        else:
            print('please only enter y or n.')
    else:
        smeared_pdos=km_smeared_ldos(N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l)
        np.save('smeared_pdos.npy',smeared_pdos)
    return smeared_pdos


def new_smearing_param_input(N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l):


    smeared_pdos=km_smeared_ldos(N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l)
    np.save('smeared_pdos.npy',smeared_pdos)
    return smeared_pdos






######################
### Key parameters ###
######################

system=input('what system are you looking at? full_inter, svac, sio2_h2o, h2o? \n')

if system == 'svac':
    structure_file = 'xyz.xyz'                        # Read in geometry information from here
    pdos_file = 'inter_charge-ALPHA_k1-1.pdos'             # Ordinary pdos for eigenvalues
    ldos_files = 'inter_charge-ALPHA_list{0}-1.pdos'       # Path for the individual pdos files
    single_ldos_file='inter_charge-ALPHA_list1-1.pdos'
    hartree_file='inter_charge-v_hartree-1_0.cube'

#if system == 'svac':
#    structure_file = 'xyz.xyz' 
#    pdos_file='full_dip-ALPHA_k1-1.pdos'#for dipole corrected full interface
#    ldos_files='full_dip-ALPHA_list{0}-1.pdos' #for dipole corrected full interface
#    single_ldos_file='full_dip-ALPHA_list1-1.pdos' #for dipole corrected full interface
#    hartree_file='full_dip-v_hartree-1_0.cube'

elif system =='sio2_h2o':
    structure_file = 'xyz.xyz' 
    pdos_file='watersio2-ALPHA_k1-1.pdos'#for sio2/h2o
    ldos_files='watersio2-ALPHA_list{0}-1.pdos' #sio2/h2o
    single_ldos_file='watersio2-ALPHA_list1-1.pdos' #sio2/h2o
    hartree_file='watersio2-v_hartree-1_0.cube'

elif system =='full_inter':
    structure_file = 'xyz.xyz' 
    pdos_file='full_dip-ALPHA_k1-1.pdos'#for dipole corrected full interface
    ldos_files='full_dip-ALPHA_list{0}-1.pdos' #for dipole corrected full interface
    single_ldos_file='full_dip-ALPHA_list1-1.pdos' #for dipole corrected full interface
    hartree_file='full_dip-v_hartree-1_0.cube'

elif system =='h2o':
    structure_file = 'h2o.xyz' 
    pdos_file='H2O-ALPHA_k1-1.pdos'#for dipole corrected full interface
    ldos_files='H2O-ALPHA_list{0}-1.pdos' #for dipole corrected full interface
    single_ldos_file='H2O-ALPHA_list1-1.pdos' #for dipole corrected full interface
    hartree_file='H2O-v_hartree-1_0.cube'

else:
    print('Input is not entered correctly - please try again!')

b_tol=input('What band_tolerance would you like? Suggested range is 0.05, 0.025,0.01 \n')

Nz = 100                                          # No. of grid points in z
NE = 100                                          # No. of grid points in E
min_eng, max_eng = -11.0, 2.5                     # Energy range of plot required
if system == 'svac':
    min_eng, max_eng = -15.0, 3 
sigma_E = 0.1                                    # Smearing in energy space (eV)
sigma_z = 0.5                                     # Smearing in z (Angstrom)
band_tol = float(b_tol)                                    # Tolerance for assigning local band edge, may require some fiddling
b_tol=float(b_tol)
decimal_part = str(b_tol)[-2:]
figure_name = decimal_part+'_band_tol_dipole_correction_retry_fermi'                              # Prefix for output files 

# Color dictionary for plotting atomic positions
atom_dict = {'Si':'y', 'O':'r', 'W':'g', 'Ob':'r', 'Hb':'k', 'Ht':'k', 'S':'y', 'Hw':'k', 'Ot':'r', 'Ow':'r'}


layers=input('How many layers are you looking at? 1,2, or 3? \n')
silica_min=37.5
silica_max=47.5

if layers=='1':
    ws2_min=25
    ws2_max=33
elif layers =='2':
    ws2_min=25
    ws2_max=31
elif layers =='3':
    ws2_min=20
    ws2_max=26



####################
### Running Code ###
####################


match=get_files(single_ldos_file,pdos_file,structure_file)

#print(match)
#print('number of snapshots:',len(match))

#match=['/mnt/lustre/a2fs-work2/work/e05/e05/ucapklm/interface/small/water/ab_initio/1_layer/confined_water/NVT/400K/energy_sampling/PBE0-TC-LRC/8000', '/mnt/lustre/a2fs-work2/work/e05/e05/ucapklm/interface/small/water/ab_initio/1_layer/confined_water/NVT/400K/energy_sampling/PBE0-TC-LRC/13000']


N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l, E_homo_mean,E_lumo_mean,z_min_mean,z_max_mean,symbols, mean_fermi, std_fermi=get_properties(match,hartree_file)

                
print('mean fermi:', mean_fermi)
#smeared_pdos=smearing_param_input(N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l)

#KM: CHANGED THIS 02/09/24
smeared_pdos=km_smeared_ldos(N_atoms, engs_mean, zs_mean, x_mean, y_mean, z_mean, sigma_E, y_std_mean,pdos_mean,ldos_mean,mean_ev_l)
print('E_homo:')
print(E_homo_mean)
print('E LUMO:')
print(E_lumo_mean)
print('Fermi energy level:')
print(mean_fermi)

#print('average ldos:')
#print(ldos_mean)
#print('average ldos eigen value:')
#print(mean_ev_l)

local_lumo,local_homo=km_band_edge(engs_mean,zs_mean,smeared_pdos,band_tol,E_homo_mean,E_lumo_mean, mean_fermi)

if system == 'svac':
    local_lumo,local_homo=km_band_edge(engs_mean,zs_mean,smeared_pdos,band_tol,-9,-9.8, -9)

with open('local_homo.txt','w') as f:
    for x_val, y_val in zip(zs_mean, local_homo):
        # Write x_val and y_val to the file, separated by a space
        f.write(f"{x_val} {y_val}\n")


with open('local_lumo.txt','w') as f:
    for x_val, y_val in zip(zs_mean, local_lumo):
        # Write x_val and y_val to the file, separated by a space
        f.write(f"{x_val} {y_val}\n")


#print('smeared pdos:')
#print(smeared_pdos)
#print(type(smeared_pdos))
#print(smeared_pdos.shape)
#print('example of smeared_pdos[0,0]:')
#print(smeared_pdos[0,0])

if rank == 0:
    # Plotting commands should only be run on the master process (rank=0)

    # Make a composite plot of the structure and ldos
    #fig=plt.figure() #new
    #ax1=plt.subplots() #new
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=(10, 5))
    plt.ylim(z_min_mean, z_max_mean)
    ax1.set_ylabel('z ($\mathrm{\AA}$)')
    ax1.set_xlabel('x ($\mathrm{\AA}$)')
    ax2.set_xlabel('Energy (eV)')

    for atom in range(0, N_atoms):
        color = None
        # Assign a color to each atom based on symbol, default to black
        try:
            atom_color = atom_dict[symbols[atom]]
        except Exception:
            atom_color = 'k'
        ax1.plot(x_mean[atom], y_mean[atom], marker='o', color=atom_color) # old

    #ax1.set_ylim(z_min_mean, z_max_mean)
    #ax2.set_ylim(z_min_mean, z_max_mean)
    #ax1.set_aspect('equal', adjustable='box', anchor=(0, 0.5))
    #ax2.set_aspect('equal', adjustable='box', anchor=(0, 0.5))
    #ax2=plt.subplots(sharey=ax1)
    img = ax2.imshow(smeared_pdos, extent=(min_eng, max_eng, z_min_mean, z_max_mean),
               interpolation='bicubic',aspect='auto', cmap='Greys',
               vmin=0.0,vmax=15,origin='lower')
    
    #ax2.yaxis.set_visible(False)
    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax2)

    # Append axes to the right of the main axes, with a fixed width
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create the color bar in the new axes
    #cbar = plt.colorbar(img, cax=cax, orientation='vertical')
    #plt.colorbar(img, cax=cax, orientation='vertical')

    #plt.colorbar(img, orientation='vertical')

    #plt.subplots_adjust(wspace=0, hspace=0)

    ax2.plot(local_homo, zs_mean, color='r')
    ax2.plot(local_lumo, zs_mean, color='r')

    lh=local_homo
    ll=local_lumo
    #plt.axvline(x=mean_fermi,linestyle='--',color='blue')
    ax1.set_ylim(z_min_mean, z_max_mean)
    ax2.set_ylim(z_min_mean, z_max_mean)
    #ax1.set_box_aspect(1)
    #ax2.set_box_aspect(1)
    ax1.set_aspect('equal', adjustable='box', anchor=(0,0))
    ax2.set_aspect('equal', adjustable='box', anchor=(0,0))#prev 0,0.5
    

    plt.savefig('{0}_smeared_ldos.png'.format(figure_name),dpi=400)
    plt.show()
    plt.clf()


    # Plot the local band edges individually
    plt.plot(zs_mean, local_homo, label='Local HOMO',color='black')
    plt.plot(zs_mean, local_lumo, label='Local LUMO',color='blue')

    def linear_func(x, a, b):
        return a * x + b
    
    # Identify valid (non-NaN and non-inf) indices
    local_homo_in = ~np.isnan(local_homo) & ~np.isinf(local_homo)
    local_lumo_in = ~np.isnan(local_lumo) & ~np.isinf(local_lumo)

    # Filter the data
    zs_mean_homo = zs_mean[local_homo_in]
    local_homo = local_homo[local_homo_in]
    zs_mean_lumo = zs_mean[local_lumo_in]
    local_lumo=local_lumo[local_lumo_in]

    
    x1, y1 = zs_mean_homo[zs_mean_homo < ws2_max], local_homo[zs_mean_homo < ws2_max]
    x2, y2 = zs_mean_homo[zs_mean_homo >= silica_min], local_homo[zs_mean_homo >= silica_min]


    x3, y3 = zs_mean_lumo[zs_mean_lumo < ws2_max], local_lumo[zs_mean_lumo < ws2_max]
    x4, y4 = zs_mean_lumo[zs_mean_lumo >= silica_min], local_lumo[zs_mean_lumo >= silica_min]

    #popt1, _ = curve_fit(linear_func, x1, y1)
    #popt2, _ = curve_fit(linear_func, x2, y2)

    #y1_fit = linear_func(x1, *popt1)
    #y2_fit = linear_func(x2, *popt2)

    mean1 = np.mean(y1)
    mean2 = np.mean(y2)


    #popt3, _ = curve_fit(linear_func, x3, y3)
    #popt4, _ = curve_fit(linear_func, x4, y4)

    #y3_fit = linear_func(x3, *popt3)
    #y4_fit = linear_func(x4, *popt4)

    mean3 = np.mean(y3)
    mean4 = np.mean(y4)


    #plt.plot(x1, y1, label=f'Fit SiO2 HOMO (mean={mean1:.2f})', color='red',linestyle='--')
    #plt.plot(x2, y2, label=f'Fit WS2 HOMO (mean={mean2:.2f})', color='red',linestyle='--')
    #plt.plot(x3, y3, label=f'Fit SiO2 LUMO (mean={mean3:.2f})', color='orange',linestyle='--')
    #plt.plot(x4, y4, label=f'Fit WS2 LUMO (mean={mean4:.2f})', color='orange',linestyle='--')
    plt.axhline(y=mean1,label=f'WS2 HOMO (mean={mean1:.2f})', color='red',linestyle='--',alpha=0.5)
    plt.axhline(y=mean2,label=f'SiO2 HOMO (mean={mean2:.2f})', color='red',linestyle='--',alpha=0.5)
    plt.axhline(y=mean3,label=f'WS2 LUMO (mean={mean3:.2f})', color='orange',linestyle='--',alpha=0.5)
    plt.axhline(y=mean4,label=f'SiO2 LUMO (mean={mean4:.2f})', color='orange',linestyle='--',alpha=0.5)
    plt.axvline(x=ws2_max, color='black', linestyle=':')
    plt.axvline(x=silica_min, color='black', linestyle=':')


    plt.legend(loc='best')
    plt.xlabel('z ($\mathrm{\AA}$)')
    plt.ylabel('energy (eV)')

    plt.savefig('{0}_local_bands.png'.format(figure_name))
    #plt.show()
    plt.clf()
    if system !='h2o':
        homomax1 = np.max(y1)
        homomax2 = np.max(y2)


        #popt3, _ = curve_fit(linear_func, x3, y3)
        #popt4, _ = curve_fit(linear_func, x4, y4)

        #y3_fit = linear_func(x3, *popt3)
        #y4_fit = linear_func(x4, *popt4)

        lumomin3 = np.min(y3)
        lumomin4 = np.min(y4)

    plt.figure()


    # Plot the local band edges individually
    plt.plot(zs_mean_homo, local_homo, label='Local HOMO',color='black')
    plt.plot(zs_mean_lumo, local_lumo, label='Local LUMO',color='blue')

    if system !='h2o':
        plt.axhline(y=homomax1,label=f'WS2 HOMO ({homomax1:.2f})', color='red',linestyle='--',alpha=0.5)
        plt.axhline(y=homomax2,label=f'SiO2 HOMO ({homomax2:.2f})', color='red',linestyle='--',alpha=0.5)
        plt.axhline(y=lumomin3,label=f'WS2 LUMO ({lumomin3:.2f})', color='orange',linestyle='--',alpha=0.5)
        plt.axhline(y=lumomin4,label=f'SiO2 LUMO ({lumomin4:.2f})', color='orange',linestyle='--',alpha=0.5)
    plt.axvline(x=ws2_max, color='black', linestyle=':')
    plt.axvline(x=silica_min, color='black', linestyle=':')
    plt.legend(loc='best')
    plt.xlabel('z ($\mathrm{\AA}$)')
    plt.ylabel('energy (eV)')

    plt.savefig('{0}_align_min_max.png'.format(figure_name))


    t_final = time.time()
    print('Total time', t_final-t_start)


if rank == 0:
    # Plotting commands should only be run on the master process (rank=0)

    # Make a composite plot of the structure and ldos
    #fig=plt.figure() #new
    #ax1=plt.subplots() #new
    plt.figure(figsize=(6,5))

    plt.xlabel('z ($\mathrm{\AA}$)',fontsize=20)
    plt.ylabel('Energy (eV)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    
    new_smear=smeared_pdos.T

    z_min=z_min_mean-z_min_mean
    z_max_mean-=z_min_mean
    print(z_min)
    print(z_max_mean)

    img = plt.imshow(new_smear, extent=(z_min, z_max_mean, min_eng, max_eng),
            interpolation='bicubic', cmap='Greys',
            vmin=0.0,origin='lower')
    #plt.grid()
    
    #ax2.yaxis.set_visible(False)
    if layers =='1':
        shrink=0.75
    else:
        shrink=0.5


    #plt.colorbar(img, orientation='vertical',shrink=shrink)

    #plt.subplots_adjust(wspace=0, hspace=0)

    #plt.plot(zs_mean, lh, color='r')
    #plt.plot(zs_mean, ll, color='r')
    zs_mean_homo-=z_min_mean
    zs_mean_lumo-=z_min_mean
    print(zs_mean_homo[0])
    plt.plot(zs_mean_homo, local_homo, color='r')#,label='Local HOMO',color='black')
    plt.plot(zs_mean_lumo, local_lumo, color='r')#label='Local LUMO',color='blue')
    ws2_max-=z_min_mean
    silica_min-=z_min_mean
    plt.axvline(x=ws2_max, color='royalblue', linestyle=':',linewidth=2)
    plt.axvline(x=silica_min, color='forestgreen', linestyle=':',linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-',linewidth=2)
    plt.xlim(z_min, z_max_mean)
    plt.tight_layout(pad=2)

    
    #plt.axhline(mean_fermi,linestyle='--',alpha=0.5)
    #ax2.set_ylim(z_min_mean, z_max_mean)
    #ax1.set_box_aspect(1)
    #ax2.set_box_aspect(1)
    #ax1.set_aspect('equal', adjustable='box', anchor=(0.5, 0.5))
    #ax2.set_aspect('equal', adjustable='box', anchor=(0.5, 0.5))#prev 0,0.5
    

    plt.savefig('ldos_only_{0}.png'.format(figure_name),dpi=400)#,bbox_inches='tight')

    if system !='h2o':
        print(f'WS2 HOMO ({homomax1:.2f})')
        print(f'SiO2 HOMO ({homomax2:.2f})')
        print(f'WS2 LUMO ({lumomin3:.2f})')
        print(f'SiO2 LUMO ({lumomin4:.2f})')