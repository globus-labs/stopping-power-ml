"""Operations devoted to importing/converting TD-DFT data"""

from ase.io import cube, qbox
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

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

def project_force(force, velocity):
    """Get the projection of the force acting against the direction of travel
    
    :param force: ndarray, force acting on particle
    :param velocity: ndarray, velocity of particle
    :return: float, magnitude of force in direction of travel"""
    
    return -1 * np.dot(force, velocity) / np.linalg.norm(velocity)

def load_qbox_data(path):
    """Load in QBox data into a data frame
    
    :param path: str, path to output file
    :return: DataFrame"""
    
    qbox_data = qbox.read_qbox(path, slice(None))
    return pd.DataFrame({
        'atoms': qbox_data,
        'frame_id': list(range(len(qbox_data))),
        'force': [project_force(frame.get_forces()[-1], frame.get_velocities()[-1]) for frame in qbox_data],
        'position': [frame.get_positions()[-1] for frame in qbox_data],
        'velocity': [frame.get_velocities()[-1] for frame in qbox_data],
        'energy': [frame.get_potential_energy() for frame in qbox_data],
        'file_id': [int(path[:-4].split("_")[-1]),]*len(qbox_data)
    })