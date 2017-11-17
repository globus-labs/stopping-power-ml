"""Options relating to determining the stopping power over a path"""

from stopping_power_ml.util import move_projectile
from math import gcd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import linalg
from scipy.integrate import quad
import numpy as np


class TrajectoryIntegrator:
    """Tool used to compute the stopping power along a certain trajectory"""

    def __init__(self, atoms, model, featurizers):
        """Create the class

        :param atoms: ase.Atoms, crystal structure being studied (including particle as the last atom)
        :param model: scikit-learn model, module used to predict stopping force
        :param featurizers: [BaseFeaturizer], tool used to generate inputs to model"""

        # Store the main details
        self.atoms = atoms
        self.model = model
        self.featurizers = featurizers

        # Compute the primitive unit cell vectors (structure minus the projectile.
        strc = AseAtomsAdaptor.get_structure(atoms[:-1])
        spg = SpacegroupAnalyzer(strc)
        self.prim_strc = spg.find_primitive()
        self.conv_strc = spg.get_conventional_standard_structure()

        # Compute the matrix that we will use to map lattice vectors in the conventional cell to ones in the primitive
        self.conv_to_prim = np.round(linalg.solve(self.prim_strc.lattice.matrix, self.conv_strc.lattice.matrix))

    def _compute_trajectory(self, lattice_vector):
        """Given a lattice vector, compute the minimum path length needed to determine stopping power.

        The lattice vector should be for the conventional cell of the structure. The path will be determined using
        the primitive cell.

        :param lattice_vector: [int], lattice vector
        :return: ndarray, 3x1 array defining the shortest path the covers an entire path"""

        # Map the conventional cell lattice vector to the primitive cell
        prim_vector = np.dot(self.conv_to_prim, np.array(lattice_vector, dtype=np.int))
        prim_vector = np.array(np.round(prim_vector), dtype=np.int)

        # Determine the shortest-possible vector
        g = gcd(prim_vector[0], gcd(prim_vector[1], prim_vector[2]))
        prim_vector = [p / g for p in prim_vector]

        # Compute the path length
        return np.dot(self.prim_strc.lattice.matrix, prim_vector)

    def _create_frame_generator(self, start_point, lattice_vector, velocity):
        """Create a function that generates a snapshot of an projectile moving along a certain trajectory

        The function takes a float between 0 and 1 as an argument. A value of 0 returns the projectile at the starting
        point of the trajectory. A value of 1 returns the particle at the first point where the trajectory repeats.
        Values between 0-1 are linearly spaced between the two.

        :param start_point: [float], starting point in conventional cell fractional coordinates
        :param lattice_vector: [int], directional of travel in conventional cell coordinates
        :param velocity: [float], projectile velocity
        :return: function float->ase.Atoms"""

        # Get the trajectory vector
        traj_vec = self._compute_trajectory(lattice_vector)

        # Compute the start point in cartesian coordinates
        start_point = np.dot(self.conv_strc.lattice.matrix, start_point)

        # Compute the velocity vector
        velocity_vec = velocity * np.array(traj_vec, np.float) / np.linalg.norm(traj_vec)

        # Create the function
        def output(x):
            position = traj_vec * x + start_point
            return move_projectile(self.atoms, position, velocity_vec)

        return output

    def _create_force_calculator(self, start_point, lattice_vector, velocity):
        """Create a function that computes the force acting on a projectile at a certain point along a trajectory

        As in `_create_frame_generator`, the function takes a float between 0 and 1 as an argument. A value of 0
        returns the force at the starting point of the trajectory. A value of 1 returns the force at the first
        point where the trajectory repeats. Values between 0-1 are linearly spaced between the two.

        :param start_point: [float], starting point in conventional cell fractional coordinates
        :param lattice_vector: [int], directional of travel in conventional cell coordinates
        :param velocity: [float], projectile velocity
        :return: function float->float"""

        generator = self._create_frame_generator(start_point, lattice_vector, velocity)

        def output(x):
            # Get the structure
            frame = generator(x)

            # Get the inputs to the model
            inputs = []
            for f in self.featurizers:
                x = f.featurize(frame)
                if isinstance(x, float):
                    inputs.append(x)
                else:
                    inputs.extend(x)

            # Evaluate the model
            return self.model.predict([x])[0]
        return output

    def compute_stopping_power(self, start_point, lattice_vector, velocity, abserr=0.001, full_output=0, **kwargs):
        """Compute the stopping power along a trajectory.

        """

        # Create the integration function
        f = self._create_force_calculator(start_point, lattice_vector, velocity)

        # Perform the integration
        return quad(f, 0, 1, epsabs=abserr, full_output=full_output, **kwargs)

if __name__ == '__main__':
    from ase.atoms import Atoms, Atom
    from stopping_power_ml.features import ProjectedAGNIFingerprints
    from sklearn.dummy import DummyRegressor

    # Create the example cell: A 4x4x4 supercell of fcc-Cu
    atoms = Atoms('Cu4', scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0., 0.5], [0, 0.5, 0.5]],
                  cell=[3.52, ]*3, momenta=[[0, 0, 0], ]*4).repeat([4, 4, 4])
    atoms.append(Atom('H', [3.52/4, 3.52/4, 0], momentum=[1, 0, 0]))

    # Create the trajectory integrator
    featurizer = ProjectedAGNIFingerprints(etas=None)
    model = DummyRegressor().fit([featurizer.featurize(atoms)], [1])
    int = TrajectoryIntegrator(atoms, model, [featurizer])

    # Make sure it gets the correct trajectory distance for a [1 1 0] conventional cell lattice vector.
    #  This travels along the face of the FCC conventional cell, and repeats halfway across the face.
    #  So, the minimum trajectory is [0.5, 0.5, 0] in conventional cell coordiantes
    assert np.isclose(int._compute_trajectory([1, 1, 0]), [1.76, 1.76, 0]).all()

    # Another test: [2 0 0]. This trajectory repeats after [1, 0, 0]
    assert np.isclose(int._compute_trajectory([2, 0, 0]), [3.52, 0, 0]).all()

    # Make sure the frame generator works properly
    f = int._create_frame_generator([0, 0, 0], [1, 0, 0], 1)
    temp = f(1)
    assert np.isclose([3.52, 0, 0], temp.get_positions()[-1]).all()
    assert np.isclose([1, 0, 0], temp.get_velocities()[-1]).all()

    f = int._create_frame_generator([0.25, 0.25, 0.25], [1, 1, 1], np.sqrt(3))
    temp = f(0.5)
    assert np.isclose([3.52 * 0.75,]*3, temp.get_positions()[-1]).all()
    assert np.isclose([1, 1, 1], temp.get_velocities()[-1]).all()

    # Test the force generator (make sure it does not crash)
    f = int._create_force_calculator([0.25, 0, 0], [1, 0, 0], 1)
    assert np.isclose(f(0), 1)  # The model should produce 1 for all inputs

    # Test the integrator
    result = int.compute_stopping_power([0.25, 0, 0], [1, 0, 0], 1)
    assert np.isclose(result[0], 1)
