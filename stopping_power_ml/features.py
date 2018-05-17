"""Functions related to computing features"""

import abc
import itertools
from scipy.integrate import romb

import numpy as np
from matminer.featurizers.site import AGNIFingerprints
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor


class ProjectileFeaturizer(BaseFeaturizer):
    """Abstract base class for computing features about a particle traveling in a material.

    Handles determining the primitive cell of a material, adding projectile to the simulation cell, etc."""

    def __init__(self, simulation_cell, use_prim_cell=True):
        """

        :param simulation_cell: ase.Atoms, simulation cell, with projectile as the last entry
        :param use_prim_cell: bool, whether to use primitive cell in calculation
        """

        # Compute the primitive unit cell vectors (structure minus the projectile.
        self.simulation_cell = AseAtomsAdaptor.get_structure(simulation_cell[:-1])

        self.use_prim_cell = use_prim_cell
        if use_prim_cell:
            # We use the `get_primitive_structure()` operation because it does not
            #  translate the atoms (spglib will). Translations mean the cartesian coordinates
            #  in the simulation cell and primitive cell are not the same, which causes all
            #  kinds of problems
            self.prim_cell = self.simulation_cell.get_primitive_structure()

    def _insert_projectile(self, position):
        """Add the projectile at a certain position into the primitive cell

        :param position: [float]*3, projectile position in cartesian coordinates
        :return: Structure, output of the cell"""

        x = self.prim_cell.copy() if self.use_prim_cell else self.simulation_cell.copy()
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

    def __init__(self, simulation_cell, acc=3, **kwargs):
        super(IonIonForce, self).__init__(simulation_cell, **kwargs)
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
        return ['Logan Ward']

    def citations(self):
        return []


class LocalChargeDensity(ProjectileFeaturizer):
    """Compute the local electronic charge density around a particle.
    
    Specifically, we evaluate the density of the particle in the past and expected future position 
    of the projectile.
    
    Input: Position and velocity of projectile
    
    Parameters:
        charge - function that takes fractional coordinates as input, returns density
    """

    def __init__(self, simulation_cell, charge, **kwargs):
        super(LocalChargeDensity, self).__init__(simulation_cell, **kwargs)
        self.charge = charge

    def feature_labels(self):
        return ['charge density']

    def featurize(self, position, velocity):
        # Convert to reduced coordinates
        cur_pos = self.simulation_cell.lattice.get_fractional_coords(position) % 1
        return np.log([self.charge(cur_pos.T)])

    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []


class ProjectedAGNIFingerprints(ProjectileFeaturizer):
    """Compute the fingerprints of the local atomic environment using the AGNI method

    We project these fingerprints along the projectiles direction of travel, and the
    unprojected fingerprints

    Input: Position and velocity of projectile

    Parameters:
        etas - list of floats, window sizes used in fingerprints
        cutoff - float, cutoff distance for features
    """

    def __init__(self, simulation_cell, etas, cutoff=16, **kwargs):
        super(ProjectedAGNIFingerprints, self).__init__(simulation_cell, **kwargs)
        self.agni = AGNIFingerprints(directions=['x', 'y', 'z', None], etas=etas, cutoff=cutoff)

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
        return ['AGNI projected eta=%.2e' % x for x in self.agni.etas]

    def featurize(self, position, velocity):
        # Compute the AGNI fingerprints [i,j] where i is fingerprint, and j is direction
        strc = self._insert_projectile(position)
        fingerprints = self.agni.featurize(strc, -1).reshape((4, -1)).T

        # Project into direction of travel
        proj_fingerprints = np.dot(
            fingerprints[:, :-1],
            velocity
        ) / np.linalg.norm(velocity)

        return proj_fingerprints

    def implementors(self):
        return ['Logan Ward']

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
    

class TimeOffset(ProjectileFeaturizer):
    """Compute the value of a feature at a different time
    
    The environment of the projectile is determined by using the 
    known velocity of the projectile."""
    
    def __init__(self, structure, featurizer, offsets=(-4,-3,-2,-1,-0.5,0,0.5,1,2)):
        """Initailize the featurizer
        
        Args:
            structure (Structure) - Structure to featurizer
            featurizer (ProjectileFeaturizer) - Featurizer to use
            offsets ([float]) - Times relative to present at which to compute features
            """
        self.structure = structure
        self.featurizer = featurizer
        self.offsets = offsets
        
    def featurize(self, position, velocity):
        positions = np.array(self.offsets)[:, np.newaxis] * \
            np.array([velocity] * len(self.offsets)) + position
        return np.ravel([self.featurizer.featurize(p, velocity) for p in positions])
    
    def feature_labels(self):
        return ['{} at t={:.2f}'.format(f, t) for t, f in itertools.product(self.offsets,
                                                                           self.featurizer.feature_labels())]

class TimeAverage(ProjectileFeaturizer):
    """Compute a weighted average of a feature over time

    The weight of events are weighted by an expontial of their time from
    the present. Users can set weights that determine whether the average of
    features in the future are past are taken into account, how how quickly
    the weights change."""

    def __init__(self, structure, featurizer, strengths=(1, 2, 3, 4, -1, -2),
                 k=5):
        """Initialize the featurizer

        Argss:
            structure (Structure) - Structure to featurizer
            featurizer (ProjectileFeaturizer) - Featurizer to average
            strengths ([float]) - How strongly features contributions varies
                with time from present. Positive weights mean the average
                will be over past events, positive ones deal with the future
            k (float) - 2 ** k + 1 points will be used in average"""
        super(TimeAverage, self).__init__(structure, True)
        self.featurizer = featurizer
        self.strengths = strengths
        self.k = k

    def featurize(self, position, velocity):

        outputs = []
        for s in self.strengths:
            # Determine particle positions and weights
            times = np.linspace(-10/s, 0, 2 ** self.k + 1)
            cur_pos = times[:, np.newaxis] * np.array([velocity] * len(times)) \
                      + position
            dt = abs(times[1] - times[0])
            weights = np.exp(times * s)

            # Evaluate the features at each of these times
            #  Do not use featurize_many because it is parallel
            #  Multiply features by the weights to prepare for integration
            features = [np.multiply(self.featurizer.featurize(pos, velocity),
                        w) for pos, w in zip(cur_pos, weights)]

            # Determine the average using Romberg integration
            outputs.append(romb(features, dx=dt, axis=0))

        # Flatten the output
        return np.squeeze(np.hstack(outputs)).tolist()

    def feature_labels(self):
        return ['time average of {}, strength={:.2f}'.format(f, s)
                for s, f in itertools.product(self.strengths,
                                              self.featurizer.feature_labels())]
