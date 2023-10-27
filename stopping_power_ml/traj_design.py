"""Design a trajectory which samples a diverse set of atomic environments"""
from typing import List
import itertools

from matminer.featurizers.base import MultipleFeaturizer
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap
import pandas as pd
import numpy as np


class TrajectoryOptimizer:
    """Tool for optimizing a trajectory

    Args:
        sampling_data: Potential points within the search space
        manifold: IsoMap trained to reduce sampling points to a reduced feature space
        x_cols: Names of columns to use for manifold projection
        thr_dist: Minimum distance between trajectory and point in sampled space to declare it "sampled"
        featurizers: List of featurizers used to describe the atomic environment of the projectile
    """

    def __init__(self,
                 sampling_data: pd.DataFrame,
                 manifold: Isomap,
                 x_cols: List[str],
                 thr_dist: float,
                 featurizers: MultipleFeaturizer):
        # Store the user-provided data
        self.sampling_data = sampling_data
        self.featurizers = featurizers
        self.manifold = manifold
        self.x_cols = x_cols
        self.thr_dist = thr_dist

        # Fit the nearest-neighbor model
        sampling_proj = manifold.transform(sampling_data[x_cols])
        self.neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(sampling_proj)

    def get_near_points(self, dataset):
        """Find the points in the sampling distribution that a trajectory passes near

        Args:
            dataset (DataFrame): Sampling of space
        Returns:
            ([int]) Indicies of points in the sampling set the dataset is near
        """

        # Get the distances from each point
        dist, ind = self.neighbors.kneighbors(self.manifold.transform(dataset[self.x_cols]))

        # Get the indices where the trajectory passes within the threshold distance
        ind = ind[dist < self.thr_dist]

        return sorted(set(np.squeeze(ind).tolist()))

    def score_sampling_performance(self, dataset):
        """Given a dataset, determine the fraction of points in the random sampling space it passes 'near'

        Args:
            dataset (DataFrame): Sampling of space
        Returns:
            float - Fraction of sampling space it passes 'near'"""

        return len(self.get_near_points(dataset)) / len(self.sampling_data)

    def generate_trajectory(self, start_pos, direction, velocity,
                            traj_len, nsteps):
        """Generate a trajectory given starting position, and direction

        Args:
            start_pos ([float]*3): Starting position in cartesian coordinates
            direction ([float]*2): Polar angle, azimuthal angle in radians
            velocity (float): Magnitude of velocity
            traj_len (float): Length of trajectory
            nsteps (int): Number of steps in trajectory
        Returns:
            (DataFrame) Position, velocity, and features for each timestep
        """
        # Generate the projectile velocity vector
        theta, phi = direction
        vel = np.multiply([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                           np.cos(theta)], velocity)

        # Make the new trajectory
        traj = np.array(start_pos) + np.dot(vel[:, None],
                                            np.linspace(0, traj_len, nsteps)[None, :]).T

        # Turn it into a dataframe
        traj_data = pd.DataFrame(list(itertools.zip_longest(traj, [vel],
                                                            fillvalue=vel)),
                                 columns=['position', 'velocity'])

        # Compute the features
        self.featurizers.featurize_dataframe(traj_data, ['position', 'velocity'], pbar=False)

        return traj_data

    def score_trajectory(self, start_pos, direction, velocity, traj_len, nsteps):
        """Compute the score of a certain candidate trajectory

        Args:
            start_pos ([float]*3): Starting position in cartesian coordinates
            direction ([float]*2): Polar angle, azimuthal angle in radians
            velocity (float): Magnitude of velocity
            traj_len (float): Length of trajectory
            nsteps (int): Number of steps in trajectory
        """
        # Generate the trajectory
        traj_data = self.generate_trajectory(start_pos, direction, velocity, traj_len, nsteps)

        # Score the trajectory
        return self.score_sampling_performance(traj_data)

    def score_multiple_trajectories(self, trajs, velocity, traj_len, nsteps):
        """Get the amount of space sampled by multiple trajectories

        Assumes each trajectory is equal length

        Args:
            trajs ([(float)*5]): Coordinates for multiple different trajectories
            velocity (float): Velocity of the projectile
            traj_len (float): Total distance of the trajectories
            nsteps (int): Total number of steps between each directory
        Return:
            (float) what fraction of the search space was sampled by the trajectory
        """

        # Determine the length of each trajectory
        sub_nsteps = int(nsteps / len(trajs))
        sub_len = traj_len / len(trajs)

        # Get the points sampled by each trajectory
        points = set()
        for traj in trajs:
            data = self.generate_trajectory(traj[:3], traj[3:], velocity, sub_len, sub_nsteps)
            points.update(self.get_near_points(data))

        # Return the score
        return len(points) / len(self.sampling_data)
