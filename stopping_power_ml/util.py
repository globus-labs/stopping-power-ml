"""Utility operations"""

import numpy as np
import pandas as pd


def move_projectile(atoms, new_position, new_velocity):
    """Create a copy of the cell with the projectile moved to a new location

    :param atoms: ase.Atoms object describing the projectile/material system.
            projectile must be the last atom
    :param new_position: [float], new position for projectile
    :param new_velocity: [float], new velocity for projectile"""

    output = atoms.copy()
    proj = output.pop()
    proj.position = new_position
    proj.momentum = np.array(new_velocity) * proj.mass
    output.append(proj)
    return output


def extend_trajectory(atoms, new_length):
    """Extend a trajectory to a certain new length

    Assumes that all entries in the trajectory are equally-spaced, and that the
    projectile is the last atom in the object

    :param atoms: list of ase.Atoms, current trajectory (must have >1 entries)
    :param new_length: int, desired length of new trajectory
    :return: list of ase.Atoms, new trajectory"""

    # Get the displacement between frames
    start_position = atoms[0].get_positions()[-1]
    velocity = atoms[0].get_velocities()[-1]
    disp_vector = atoms[1].get_positions()[-1] - start_position

    # Make the new frames
    return [move_projectile(atoms[0], i * disp_vector + start_position, velocity) for
                i in range(new_length)]
