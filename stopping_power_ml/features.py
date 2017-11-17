"""Functions related to computing features

TBD: Modify pymatgen to store velocities (if they want that)"""

import numpy as np
from matminer.featurizers.site import AGNIFingerprints
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor


class IonIonForce(BaseFeaturizer):
    """Compute the stopping force acting on a particle from ion-ion repulsion
    
    Computes the force from the repulsion of nuclei (i.e., the charge on each atom is 
    its atomic number) projected along the particle's direction of travel. 
    
    Input: ASE atoms object with the projectile as the last atom, units in atomic units.
    
    Parameters:
        acc - float, accuracy of the Ewald summation (default=3)"""
    
    def __init__(self, acc=3):
        self.acc=acc
        
    def feature_labels(self):
        return ["ion-ion repulsion"]
        
    def featurize(self, atoms):

        # Get the atoms object as a pymatgen Structure
        strc = AseAtomsAdaptor.get_structure(atoms)

        # Convert lattice from Bohr to Angstrom
        strc.scale_lattice((0.529177 ** 3) * strc.volume)

        # Assign a charge of Z to each atom
        for site in strc.sites:
            site.charge = site.specie.Z

        # Compute the forces
        ewald = EwaldSummation(strc, compute_forces=True, acc_factor=self.acc)

        # Compute force acting against the direction of particle travel
        my_force = ewald.forces[-1,:]
        my_velocity = atoms.get_velocities()[-1,:]
        return -1 * np.dot(my_force, my_velocity) / np.linalg.norm(my_velocity) * 0.03674932

    def implementors(self):
        return ['Logan Ward',]

    def citations(self):
        return []


class LocalChargeDensity(BaseFeaturizer):
    """Compute the local electronic charge density around a particle.
    
    Specifically, we evaluate the density of the particle in the past and expected future position 
    of the projectile.
    
    Input: ASE atoms object with the projectile as the last atom, units in atomic units
    
    Parameters:
        charge - function that takes fractional coordinates as input, returns density
        times - list of float, times at which to evaluate the density
    """
    
    def __init__(self, charge, times):
        self.charge = charge
        self.times = times
        
    def feature_labels(self):
        return ['log density t='+str(t) for t in self.times]

    def featurize(self, atoms):
        # Compute the positions
        cur_pos = np.array(self.times)[:,np.newaxis] * atoms.get_velocities()[-1, np.newaxis] + atoms.get_positions()[-1]

        # Convert to reduced coordinates
        cur_pos = np.linalg.solve(atoms.cell, cur_pos.T) % 1
        return np.log(self.charge(cur_pos.T))

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []


class ProjectedAGNIFingerprints(BaseFeaturizer):
    """Compute the fingerprints of the local atomic environment using the AGNI method

    We project these fingerprints along the projectiles direction of travel

    Input: ASE atoms object with the projectile as the last atom, units in atomic units

    Parameters:
        etas - list of floats, window sizes used in fingerprints
    """

    def __init__(self, etas, cutoff=16):
        self.agni = AGNIFingerprints(directions=['x','y','z'], etas=etas, cutoff=cutoff)

    @property
    def etas(self):
        return self.agni.etas

    @etas.setter
    def etas(self, x):
        self.agni.etas = x

    @property
    def cutoff(self):
        return self.agni.cutoff

    @cutoff.setter
    def cutoff(self, x):
        self.agni.cutoff = x

    def feature_labels(self):
        return ['AGNI eta=%.2e'%x for x in self.agni.etas]

    def featurize(self, atoms):
        """Get the AGNI fingerprints projected in the direction of travel

        :param atoms: ase.Atoms, structure
        :return: ndarray, fingerprints projected in the direction of travel"""

        # Compute the AGNI fingerprints [i,j] where i is fingerprint, and j is direction
        fingerprints = self.agni.featurize(AseAtomsAdaptor.get_structure(atoms), -1).reshape((3, -1)).T

        # Project into direction of travel
        my_velocity = atoms.get_velocities()[-1,:]
        fingerprints = np.dot(
            fingerprints,
            my_velocity
        ) / np.linalg.norm(my_velocity)

        return fingerprints

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []


class RepulsionFeatures(BaseFeaturizer):
    """Compute features the $1/r^n$ repulsion. Designed to be a faster approximation of the Coulomb repulsion force

    Input: ASE atoms object with the projectile as the last atom, units in atomic units

    Parameters:
        cutoff - float, cutoff distance for potential
        n - int, exponent for the repulsion potential"""

    def __init__(self, cutoff=40, n=6):
        self.cutoff = cutoff
        self.n = n

    def feature_labels(self):
        return ["repulsion force",]

    def featurize(self, atoms):
        # Putting these temporarily in here
        strc = AseAtomsAdaptor.get_structure(atoms)
        proj = strc[-1]

        # Compute the 'force' acting on the projectile
        force = np.zeros(3)
        for n, r in strc.get_neighbors(proj, self.cutoff):
            disp = n.coords - proj.coords
            force += disp * proj.specie.Z * n.specie.Z / np.power(r, self.n + 1)
        my_velocity = atoms.get_velocities()[-1,:]
        return np.dot(force, my_velocity) / np.linalg.norm(my_velocity)

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []