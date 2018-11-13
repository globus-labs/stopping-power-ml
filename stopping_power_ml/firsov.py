from stopping_power_ml.features import ProjectileFeaturizer
import numpy as np


class FirsovModel(ProjectileFeaturizer):
    """Implements a slightly modified version of the Firsov model for stopping force.
    
    The Firsov model determines the stopping force as a function of the nuclear charges 
    of the projectile and nearest atom, and the distance between the projectile and the 
    nearest atom. Beyond the normal fitting parameter that controls the scale of the force,
    we also allow the user to specify an "offset" that computes the stopping force at a certain
    point ahead or behind the projectile.
    
    Features:
        Firsov force - Force computed using the Firsov model
    """
    
    def __init__(self, simulation_cell, N=1, offset=0, a=0.47, Z_A=1, Z_B=13, alpha=None):
        """Initialize the featurizer
        
        Args:
            simulation_cell (ase.Atoms): The simulation cell as an ase object
            N (float): Scale factor
            offset (float): Controls whether the force is computed for positions ahead of or
                behind the particle. Position is computed as "position + offset * velocity",
                so this value is effective the time in the past/future
            a (float): Parameter that controls the rate at which force increases with distance
            Z_A (float): Charge on the projectile
            Z_B (float): Charge on the nearest atom
            alpha (float): Misc fitting paramter. If none, uses the default value based on Z_A/Z_B:w

        """
        super(FirsovModel, self).__init__(simulation_cell, True)
        
        # Set the parameters
        self.N = N
        self.offset = offset
        self.a = a
        self.Z_A = 1
        self.Z_B = 13
        self.alpha = alpha
        
    def compute_r(self, position):
        """Compute the distance between a projectile and the nearest atom

        Args:
            position ([float]*3): Position of the projectile in Cartesian coordinates
        Returns:
            (float) Distance to nearest atom
        """

        # Compute the nearest atoms within a shell
        return min(x[1] for x in self.simulation_cell.get_sites_in_sphere(position, 6))
    
    def featurize(self, position, velocity):
        # Get the distance to the nearest atom
        R = self.compute_r(np.add(position, np.multiply(self.offset, velocity)))

        # Compute the alpha value
        if self.alpha is None:
            alpha = 1/(1+(self.Z_B/ self.Z_A)**(1/6))
        else:
            alpha = self.alpha

        # Get the velocity magnitude
        V = np.linalg.norm(velocity)

        return [V * self.N * (self.Z_A ** 2 / (1 + 0.8 * alpha * self.Z_A ** (1/3) * R / self.a) ** 4 
                              + self.Z_B ** 2 / (1 + 0.8 * (1 - alpha) * self.Z_B ** (1/3) * R / self.a) ** 4)]
    
    def feature_labels():
        return ['Firsov force']
