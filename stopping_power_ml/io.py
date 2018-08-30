"""Operations devoted to importing/converting TD-DFT data"""

from ase.io import cube, qbox
import os
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from glob import glob
import sys


def _expand_density(rho):
    """Density from CUBE is on range [0,1) along each axis, make it go from [0,1]

    Do so by appending the data at '0' to that at 1 (periodic boundary conditions)"""

    # Make a slightly larger array
    rho_new = np.zeros(np.array(rho.shape) + 1)

    # Copy over the data
    rho_new[:rho.shape[0], :rho.shape[1], :rho.shape[2]] = rho  # Bulk of the data

    rho_new[:rho.shape[0], :rho.shape[1], rho.shape[2]] = rho[:, :, 0]  # Faces
    rho_new[:rho.shape[0], rho.shape[1], :rho.shape[2]] = rho[:, 0, :]
    rho_new[rho.shape[0], :rho.shape[1], :rho.shape[2]] = rho[0, :, :]

    rho_new[:rho.shape[0], rho.shape[1], rho.shape[2]] = rho[:, 0, 0]  # Edges
    rho_new[rho.shape[0], :rho.shape[1], rho.shape[2]] = rho[0, :, 0]
    rho_new[rho.shape[0], rho.shape[1], :rho.shape[2]] = rho[0, 0, :]

    rho_new[rho.shape[0], rho.shape[1], rho.shape[2]] = rho[0, 0, 0]  # Point

    return rho_new


def _get_interpolator(cd, symmetry=False, cell=None, sym_accuracy=16):
    """Downsample a charge car to a certain number of samples.

    :param cd: ndarray, charge density from CUBE file
    :param symmetry: bool, whether to force stability
    :param cell: ase.Atoms, simulation cell used in calculation
    :param sym_accuracy: int, number of points along each axis for sym interpolator
    :return: function that takes an nx3 array, returns nx1 charge densities"""

    # Expand the array to go between zero and one
    charge = _expand_density(cd)

    # Make the interpolation function
    inter = RegularGridInterpolator([list(np.linspace(0, 1, x)) for x in charge.shape], charge)
    if not symmetry:
        return inter

    # Get symmetry operations
    return _SymmetrizedInterpolator(inter, cell, sym_accuracy)


class _SymmetrizedInterpolator:
    """Interpolator for charge density that takes symmetry into account

    Determines the primitive cell of the lattice for the simulation cell and creates an interpolator only over the
    primitive cell. Also makes sure that the interpolator obeys the same symmetry as the lattice."""

    def __init__(self, inter, cell, n_points=4):
        """Create the interpolator

        :param inter: RegularGridInterpolator, interpolator over the full simulaiton cell
        :param cell: ase.Atoms, simulation cell
        :param n_points: int, number of points along each axis"""
        strc = AseAtomsAdaptor.get_structure(cell)
        spg = SpacegroupAnalyzer(strc)
        prim_strc = spg.get_primitive_standard_structure()

        # Generate a new interpolator
        self.sym_ops = SpacegroupAnalyzer(prim_strc).get_symmetry_operations()
        self.cell_to_prim = np.linalg.solve(prim_strc.lattice.matrix, strc.lattice.matrix)
        prim_to_cell = np.linalg.inv(self.cell_to_prim)

        # Generate an interpolator over the primitive cell
        spacing = np.linspace(0, 1, n_points).tolist()
        xx, yy, zz = np.meshgrid(spacing, spacing, spacing)
        cc = []
        xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
        already_checked = []
        for c in zip(xx, yy, zz):
            pcc = []
            found_equivalent = False
            for op in self.sym_ops:
                sc = op.operate(c) % 1  # Get the point to be sampled

                # Check whether this point has already been sampled
                if len(already_checked) > 0:
                    dists = np.linalg.norm(np.subtract(already_checked, sc), axis=1)
                    best_match = dists.argmin()
                    if np.isclose(dists[best_match], 0):
                        found_equivalent = True
                        break

                # If an equivalent has not *yet* been found, compute density
                pc = np.dot(prim_to_cell, sc)
                pc = pc % 1
                pcc.append(inter(pc))

            # If an equivalent is not found, return
            if found_equivalent:
                cc.append(cc[best_match])
            else:
                cc.append(np.mean(pcc))

            # Mark this row as already checked
            already_checked.append(c)

        # Reshape the charge densities
        cc = np.array(cc).reshape((len(spacing),) * 3)

        # Build an interpolator
        self.prim_inter = RegularGridInterpolator((spacing,) * 3, cc)

        # Store the prim->Cartesian space vector
        self.prim_to_cart = prim_strc.lattice.matrix

    def __call__(self, x):
        """Evaluate the interpolator

        :param x: nx3, coordinates of point in supercell"""
        x = np.array(x)
        X_prim = np.dot(self.cell_to_prim, x.T) % 1
        if X_prim.shape == (3,):
            return self.evaluate_with_symmetry(X_prim)
        return np.array([self.evaluate_with_symmetry(point) for point in X_prim.T])

    def evaluate_with_symmetry(self, point):
        """Evaluate the interpolator across all equivalent points in the primitive cell
        :param point: nx3, """
        points = []
        for op in self.sym_ops:
            points.append(op.operate(point) % 1)  # Get the point to be sampled
        return sum(self.prim_inter(points)) / len(self.sym_ops)


def get_charge_density_interpolator(path, symmetry=False, sym_accuracy=16):
    """Read in CUBE file, create interpolator over that data

    :param path: str, path to dataset file
    :param symmetry: boolean, whether to force the interpolator to be symmetric
    :param sym_accuracy: int, the number of points along each axis for the symmetric interpolator
    :return:
        - function that takes scaled position and returns interpolated density
        - ndarray, cell of the cube file"""

    charge, atoms = cube.read_cube_data(path)
    return _get_interpolator(charge, symmetry=symmetry, cell=atoms, sym_accuracy=sym_accuracy), atoms.cell


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

    # Get the file ID from path
    try:
        file_id = int(path[:-4].split("_")[-1])
    except (IndexError, ValueError):
        file_id = 0

    return pd.DataFrame({
        'atoms': qbox_data,
        'frame_id': list(range(len(qbox_data))),
        'force': [project_force(frame.get_forces()[-1], frame.get_velocities()[-1]) for frame in qbox_data],
        'position': [frame.get_positions()[-1] for frame in qbox_data],
        'velocity': [frame.get_velocities()[-1] for frame in qbox_data],
        'energy': [frame.get_potential_energy() for frame in qbox_data],
        'file_id': [file_id, ] * len(qbox_data)
    })


def load_directory(d, prefix="", suffix=".out"):
    """Load in a directory holding a single trajectory

    Args:
        d (string): Path do a directory
        prefix (string): String at the beginning of files (default is "")
        suffix (string): Extension of the file (default is ".out")
    Returns:
        (DataFrame): The data in all files"""

    # Read in the data files
    data = []
    for file in glob('%s/%s*%s' % (d, prefix, suffix)):
        try:
            frame = load_qbox_data(file)
        except:
            print('File failed to read: %s' % file, file=sys.stderr)
            raise
        frame['file'] = file
        data.append(frame)
    data = pd.concat(data)

    # Sort, assign timestep values
    data.sort_values(['file_id', 'frame_id'], ascending=True, inplace=True)
    data['timestep'] = list(range(len(data)))

    # Compute displacement
    data['displacement'] = (data['position'] - data['position'].iloc[0]).apply(np.linalg.norm)

    # Add tag for the directory
    data['directory'] = d

    return data


if __name__ == "__main__":
    charge, cell = get_charge_density_interpolator(os.path.join('..', 'datasets', '256_Al', 'Al_semi_core_gs.cube'),
                                                   symmetry=True, sym_accuracy=4)
    print(charge([0.25, ] * 3))
    print(charge([[0, 1, 0], [1, 0, 0]]))
    print(charge([0.1, 0.1, 0]) - charge([0.35, 0.1, 0]))
    print(charge([0.1, 0.1, 0]), charge([-0.1, 0.1, 0]))
    print(charge([0.1, 0.1, 0]) - charge([-0.1, 0.1, 0]))
