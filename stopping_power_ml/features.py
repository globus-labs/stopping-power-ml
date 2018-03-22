"""Functions related to computing features"""

import abc

import numpy as np
from scipy.integrate import tplquad
from scipy.stats import multivariate_normal
from matminer.featurizers.site import AGNIFingerprints
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor
from itertools import product


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

    def __init__(self, simulation_cell, charge, times, **kwargs):
        super(LocalChargeDensity, self).__init__(simulation_cell, **kwargs)
        self.charge = charge
        self.times = times

    def feature_labels(self):
        return ['log density t=' + str(t) for t in self.times]

    def featurize(self, position, velocity):
        # Compute the positions
        cur_pos = np.array(self.times)[:, np.newaxis] * np.array([velocity]*len(self.times)) \
                  + position

        # Convert to reduced coordinates
        cur_pos = np.linalg.solve(self.simulation_cell.lattice.matrix, cur_pos.T) % 1
        return [self._get_smeared_density(pos, 0.1) for pos in cur_pos.T]

    def _get_smeared_density(self, position, sigma, max_r=5, quad=False):
        """Compute a weighted average of the density around a point, where
        the density is weighted by a Gaussian.

        :param position: 3x1 array, local to be assessed
        :param sigma: float, strength of Gaussian
        :param max_r: float, how far to average out. Radius limit of
            integral will be `max_r * sigma`
        :return: float, local average"""

        # Compute the range of integration
        rmax = sigma * max_r

        def rho(theta, phi, r):
            shifts = np.transpose([np.multiply(r, np.sin(theta) * np.cos(phi)),
                                   np.multiply(r, np.sin(theta) * np.sin(phi)),
                                   np.multiply(r, np.cos(theta))])
            pos = shifts + np.atleast_2d(position)
            pos = np.linalg.solve(self.simulation_cell.lattice.matrix, pos.T) % 1

            output = self.charge(pos.T) * np.exp(-0.5 * np.power(r / sigma, 2)) / \
                     np.sqrt(np.power(np.pi * 2 * sigma ** 2, 3))
            if quad:
                output *= np.sin(theta) * np.power(r, 2)
            return output

        if quad:
            return tplquad(rho, 0, rmax,  # r limits
                           lambda x: 0, lambda x: 2 * np.pi,  # phi limits
                           lambda x, y: 0, lambda x, y: np.pi,  # theta limits
                           epsabs=0.1, epsrel=0.1)[0] / \
                            (4 / 3 * np.pi * (sigma * max_r) ** 3)
        else:
            # Create an interpolation grid
            accuracy = 9
            r = np.linspace(0, rmax, accuracy + 2)[1:-1]
            phi = np.linspace(0, 2 * np.pi, accuracy + 2)[1:-1]
            theta = np.linspace(0, np.pi, accuracy + 2)[1:-1]
            points = np.array(list(product(theta, phi, r)))

            # Unpack the coordinates for each point
            r = points[:, 2]
            phi = points[:, 1]
            theta = points[:, 0]

            # Evaluate the grid points
            my_rho = rho(*points.T)

            # Compute the box sizes
            #  Each voxel is at the center of a spherical section that
            #   extends half of (rmax / accuracy) to the front and back,
            #   half of (2pi / accuracy) for phi to each azimuthal angle
            #   and half of (pi / accuracy) to each inclination angle
            dR = rmax / accuracy / 2
            dPhi = np.pi / accuracy
            dTheta = np.pi / accuracy / 2
            vol = (np.power(r + dR, 3) - np.power(r - dR, 3)) / 3 \
                * (np.cos(theta - dTheta) - np.cos(theta + dTheta)) \
                * (dPhi * 2)

            return np.dot(my_rho, vol)

    def implementors(self):
        return ['Logan Ward', ]

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
        return ['AGNI projected eta=%.2e' % x for x in self.agni.etas] + \
               ['AGNI eta=%.2e' % x for x in self.agni.etas]

    def featurize(self, position, velocity):
        # Compute the AGNI fingerprints [i,j] where i is fingerprint, and j is direction
        strc = self._insert_projectile(position)
        fingerprints = self.agni.featurize(strc, -1).reshape((4, -1)).T

        # Project into direction of travel
        proj_fingerprints = np.dot(
            fingerprints[:, :-1],
            velocity
        ) / np.linalg.norm(velocity)

        return np.hstack((proj_fingerprints, fingerprints[:, -1]))

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
