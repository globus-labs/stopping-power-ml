"""Functions related to computing features"""

import abc

import numpy as np
from matminer.featurizers.site import AGNIFingerprints
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor


class ProjectileFeaturizer(BaseFeaturizer):
    """Abstract base class for computing features about a particle traveling in a material.

    Handles determining the primitive cell of a material, adding projectile to the simulation cell, etc."""

    def __init__(self, simulation_cell):
        """

        :param simulation_cell: ase.Atoms, simulation cell, with projectile as the last entry
        """
        # Compute the primitive unit cell vectors (structure minus the projectile.
        self.simulation_cell = AseAtomsAdaptor.get_structure(simulation_cell[:-1])

        # We use the `get_primitive_structure()` operation because it does not
        #  translate the atoms (spglib will). Translations mean the cartesian coordinates
        #  in the simulation cell and primitive cell are not the same, which causes all
        #  kinds of problems
        self.prim_cell = self.simulation_cell.get_primitive_structure()

    def _insert_projectile(self, position):
        """Add the projectile at a certain position into the primitive cell

        :param position: [float]*3, projectile position in cartesian coordinates
        :return: Structure, output of the cell"""

        x = self.prim_cell.copy()
        x.append('H', position, coords_are_cartesian=True)
        return x

    @abc.abstractmethod
    def featurize(self, position, velocity):
        """Compute features for a projectile system"""

        raise NotImplementedError()


class IonIonForce(ProjectileFeaturizer):
    """Compute the stopping force acting on a particle from ion-ion repulsion
    
    Computes the force from the repulsion of nuclei (i.e., the charge on each atom is 
    its atomic number) projected along the particle's direction of travel. 
    
    Input: Position and velocity of projectile
    
    Parameters:
        acc - float, accuracy of the Ewald summation (default=3)"""

    def __init__(self, simulation_cell, acc=3):
        super(IonIonForce, self).__init__(simulation_cell)
        self.acc = acc

    def feature_labels(self):
        return ["ion-ion repulsion"]

    def featurize(self, position, velocity):
        # Get the atoms object as a pymatgen Structure
        strc = self._insert_projectile(position)

        # Convert lattice from Bohr to Angstrom
        strc.scale_lattice((0.529177 ** 3) * strc.volume)

        # Assign a charge of Z to each atom
        for site in strc.sites:
            site.charge = site.specie.Z

        # Compute the forces
        ewald = EwaldSummation(strc, compute_forces=True, acc_factor=self.acc)

        # Compute force acting against the direction of particle travel
        my_force = ewald.forces[-1, :]
        return [-1 * np.dot(my_force, velocity) / np.linalg.norm(velocity) * 0.03674932]

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []


class LocalChargeDensity(ProjectileFeaturizer):
    """Compute the local electronic charge density around a particle.
    
    Specifically, we evaluate the density of the particle in the past and expected future position 
    of the projectile.
    
    Input: Position and velocity of projectile
    
    Parameters:
        charge - function that takes fractional coordinates as input, returns density
        times - list of float, times at which to evaluate the density
    """

    def __init__(self, simulation_cell, charge, times):
        super(LocalChargeDensity, self).__init__(simulation_cell)
        self.charge = charge
        self.times = times

    def feature_labels(self):
        return ['log density t=' + str(t) for t in self.times]

    def featurize(self, position, velocity):
        # Compute the positions
        cur_pos = np.array(self.times)[:, np.newaxis] * np.array(velocity)[-1, np.newaxis] \
                  + position

        # Convert to reduced coordinates
        cur_pos = np.linalg.solve(self.simulation_cell.lattice.matrix, cur_pos.T) % 1
        return np.log(self.charge(cur_pos.T))

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []


class ProjectedAGNIFingerprints(ProjectileFeaturizer):
    """Compute the fingerprints of the local atomic environment using the AGNI method

    We project these fingerprints along the projectiles direction of travel

    Input: Position and velocity of projectile

    Parameters:
        etas - list of floats, window sizes used in fingerprints
    """

    def __init__(self, simulation_cell, etas, cutoff=16):
        super(ProjectedAGNIFingerprints, self).__init__(simulation_cell)
        self.agni = AGNIFingerprints(directions=['x', 'y', 'z'], etas=etas, cutoff=cutoff)

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
        return ['AGNI eta=%.2e' % x for x in self.agni.etas]

    def featurize(self, position, velocity):
        # Compute the AGNI fingerprints [i,j] where i is fingerprint, and j is direction
        strc = self._insert_projectile(position)
        fingerprints = self.agni.featurize(strc, -1).reshape((3, -1)).T

        # Project into direction of travel
        fingerprints = np.dot(
            fingerprints,
            velocity
        ) / np.linalg.norm(velocity)

        return fingerprints

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []


class RepulsionFeatures(ProjectileFeaturizer):
    """Compute features the $1/r^n$ repulsion. Designed to be a faster approximation of the Coulomb repulsion force

    Input: Position and velocity of projectile

    Parameters:
        cutoff - float, cutoff distance for potential
        n - int, exponent for the repulsion potential"""

    def __init__(self, simulation_cell, cutoff=40, n=6):
        super(RepulsionFeatures, self).__init__(simulation_cell)
        self.cutoff = cutoff
        self.n = n

    def feature_labels(self):
        return ["repulsion force", ]

    def featurize(self, position, velocity):
        # Putting these temporarily in here
        strc = self._insert_projectile(position)
        proj = strc[-1]

        # Compute the 'force' acting on the projectile
        force = np.zeros(3)
        for n, r in strc.get_neighbors(proj, self.cutoff):
            disp = n.coords - proj.coords
            force += disp * proj.specie.Z * n.specie.Z / np.power(r, self.n + 1)
        return np.dot(force, velocity) / np.linalg.norm(velocity)

    def implementors(self):
        return ['Logan Ward', ]

    def citations(self):
        return []


class ProjectileVelocity(ProjectileFeaturizer):
    """Compute the projectile velocity
    
    Input: Position and velocity of projectile
    
    Parameters: None"""

    def feature_labels(self):
        return ["velocity_mag", ]

    def featurize(self, position, velocity):
        return [np.linalg.norm(velocity)]
