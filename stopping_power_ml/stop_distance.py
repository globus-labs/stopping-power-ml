"""Tools for computing the stopping distance of a projectile"""

from scipy.optimize import minimize_scalar
from scipy.integrate import RK45
from time import perf_counter
from copy import copy
import pandas as pd
import numpy as np
import tempfile
import keras

class StoppingDistanceComputer:
    """Utility tool used compute the stopping distance"""
    
    def __init__(self, traj_int, proj_mass=1837, max_step=np.inf):
        """Initialize the stopping distance computer
        
        Args:
            traj_int (TrajectoryIntegrator): Tool used to create the force calculator
            proj_mass (float): Mass of the projectile in atomic units
            max_step (float): Maximum timestep size allowed by the integrator
        """
        
        self.traj_int = traj_int
        self.proj_mass = proj_mass
        self.max_step = max_step
        
    def __getstate__(self):
        # Save the model embedded in the TrajInt 
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self.traj_int.model, fd.name, overwrite=True)
            model_str = fd.read()
        
        # Assemble the pickle object
        d = self.__dict__.copy()
        d.update({ 'model_str': model_str })
        
        # Modify the traj_int to have model be a placeholder
        d['traj_int'] = copy(self.traj_int)  # So as to not effect self
        d['traj_int'].model = 'placeholder'
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        
        # Load the Keras model from disk 
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        
        # Add it to the traj_int
        self.traj_int.model = model
        
    def _make_ode_function(self, start_point, start_traj):
        """Make the function used to run the ODE

        Args:
            start_point ([float]*3): Starting point of the run
            start_traj ([float]*3): Starting direction
        """

        # Make the force calculator
        force_calc = self.traj_int.create_force_calculator_given_displacement(start_point, start_traj)

        def output(t, y):
            # Get the velocity and displacement
            v, x = y

            # Compute the force
            f = force_calc(x, v)
            return [-f / self.proj_mass, v]
        return output
    
    def compute_stopping_distance(self, start_point, start_velocity, stop_velocity_mag=0.4, max_time=1e5, output=None, status=True):
        """Compute the stopping distance of a projectile
        
        Args:
            start_point ([float]*3): Starting point of the run. In fractional coordinates of conventional cell
            start_velocity ([float]*3): Starting velocity
            stop_velocity_mag (float): Velocity at which to stop the calculation
            max_step (float): Maximum timestep length
            max_time (float): Time at which to stop the solver (assuming an error)
            output (int): Number of timesteps between outputting status information
            status (bool): Whether to print status information to screen
        Returns:
            - (float) Stopping distance
            - (pd.DataFrame) Velocity as a function of position and time
        """
        start_time = perf_counter()

        # Make the force calculator
        fun = self._make_ode_function(start_point, start_velocity)
        
        # Compute the initial velocity
        v_init = np.linalg.norm(start_velocity)
        
        # Create the ODE solver
        rk = RK45(fun, 0, [v_init, 0], max_time, max_step=self.max_step)
        
        # Iterate until velocity slows down enough
        i = 0
        states = [(0, v_init, 0, perf_counter() - start_time)]
        while rk.y[0] > stop_velocity_mag:
            rk.step()
            i += 1
            if output is not None and i % output == 0:
                states.append([rk.t, *rk.y, perf_counter() - start_time])
                if status:
                    print('\rStep: {} - Time: {} - Velocity: {} - Position: {}'.format(i, rk.t, rk.y[0], rk.y[1]), end="")
                    
        # Determine the point at which the velocity crosses the threshold
        #   ODE solvers give you an interpolator over the last timestep
        interp = rk.dense_output()
        res = minimize_scalar(lambda x: np.abs(interp(x)[0] - stop_velocity_mag), bounds=(rk.t_old, rk.t))
        stop_dist = interp(res.x)[1]
                
        # Return the results
        if output is not None:
            print('\rStep: {} - Time: {} - Velocity: {} - Position: {}'.format(i, rk.t, rk.y[0], rk.y[1]))
            states = pd.DataFrame(dict(zip(['time', 'velocity', 'displacement', 'sim_time'], np.transpose(states))))
            return stop_dist, states
        return stop_dist
