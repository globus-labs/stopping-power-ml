"""Utility functions for modeling stopping power"""

from ase.io import cube
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matminer.featurizers.site import AGNIFingerprints
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor

def _expand_density(rho):
    """Density from CUBE is on range [0,1) along each axis, make it go from [0,1]
    
    Do so by appending the data at '0' to that at 1 (periodic boundary conditions)"""
    
    # Make a slightly larger array
    rho_new = np.zeros(np.array(rho.shape)+1)
    
    # Copy over the data
    rho_new[:rho.shape[0], :rho.shape[1], :rho.shape[2]] = rho # Bulk of the data
    
    rho_new[:rho.shape[0], :rho.shape[1], rho.shape[2]] = rho[:,:,0] # Faces
    rho_new[:rho.shape[0], rho.shape[1], :rho.shape[2]] = rho[:,0,:]
    rho_new[rho.shape[0], :rho.shape[1], :rho.shape[2]] = rho[0,:,:]
    
    rho_new[:rho.shape[0], rho.shape[1], rho.shape[2]] = rho[:,0,0] # Edges
    rho_new[rho.shape[0], :rho.shape[1], rho.shape[2]] = rho[0,:,0] 
    rho_new[rho.shape[0], rho.shape[1], :rho.shape[2]] = rho[0,0,:]
    
    rho_new[rho.shape[0], rho.shape[1], rho.shape[2]] = rho[0,0,0] # Point
    
    return rho_new

def _get_interpolator(cd):
    """Downsample a charge car to a certain number of samples. 
    
    :return: ndarray with the requested number of samples"""
    
    # Expand the array to go between zero and one
    charge = _expand_density(cd)
    
    # Grid varies between [0,1)
    return RegularGridInterpolator([list(np.linspace(0,1,x)) for x in charge.shape], charge)

def get_charge_density_interpolator(path):
    """Read in CUBE file, create interpolator over that data
    
    :param path: str, path to dataset file
    :return: 
        - function that takes scaled position and returns interpolated density
        - ndarray, cell of the cube file"""
    
    charge, atoms = cube.read_cube_data(os.path.join('256_Al', 'Al_semi_core_gs.cube'))
    return _get_interpolator(charge), atoms.cell

def compute_ewald_force(atoms, acc=12):
    """Compute the force acting on the particle
    
    :param atoms: ase.Atoms, structure being assessed (particle is last site)
    :param acc: int, accurracy of calculation (number of decimal points to which energy is converged)
    :return: float, force acting on particle due to nucleus-nucleus repulsion in Ha/bohr"""
    
    # Get the atoms object as a pymatgen Structure
    strc = AseAtomsAdaptor.get_structure(atoms)
    
    # Convert lattice from Bohr to Angstrom
    strc.scale_lattice((0.529177 ** 3) * strc.volume)
    
    # Assign a charge of Z to each atom
    for site in strc.sites:
        site.charge = site.specie.Z
        
    # Compute the forces
    ewald = EwaldSummation(strc, compute_forces=True, acc_factor=acc)
    
    # Compute force acting against the direction of particle travel
    my_force = ewald.forces[-1,:]
    my_velocity = atoms.get_velocities()[-1,:]
    return -1 * np.dot(my_force, my_velocity) / np.linalg.norm(my_velocity) * 0.03674932