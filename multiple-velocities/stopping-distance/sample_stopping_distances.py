from concurrent.futures import as_completed
from parsl.config import Config
from parsl.configs import theta_local_htex_multinode
from parsl import python_app
from tqdm import tqdm
from glob import glob
import numpy as np
import argparse
import parsl
import os


# Make an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Parsl configuration', default='local')
parser.add_argument('velocity', help='Starting velocity', type=float)
parser.add_argument('n_samples', help='Number of trajectories to sample', type=int)

# Parse the arguments
args = parser.parse_args()

# Set the Parsl configuration
if args.config == 'local':
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider
    config = Config(
        executors=[
            HighThroughputExecutor(
                label='local',
                max_workers=1,
                provider=LocalProvider(max_blocks=1, )
            )
        ]
    )
elif args.config == 'theta':
    # Get the number of nodes
    nnodes = int(os.environ['COBALT_JOBSIZE'])
    tasks_per_node = 64

    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider
    from parsl.launchers import AprunLauncher
    from parsl.addresses import address_by_hostname
    config = Config(
        executors=[
            HighThroughputExecutor(
                label='theta',
                max_workers=tasks_per_node,
                address=address_by_hostname(),
                tasks_per_node=tasks_per_node,
                provider=LocalProvider(
                    launcher=AprunLauncher(),
                    init_blocks=1,
                    max_blocks=1,
                    # Command to be run before starting a worker, such as:
                    # 'module load Anaconda; source activate parsl_env'.
                    worker_init='module load miniconda-3.6/conda-4.5.4; ',
                ),
            )
        ],
    )

parsl.load(config)


# Make the Python function
@python_app
def compute_stopping_distance(start_point, start_velocity, output_freq=1000):
    """Compute the stopping distance for a certain trajectory

    Args:
        start_point ([float]*3): Starting point of the run. In fractional coordinates of conventional cell
        start_velocity ([float]*3): Starting velocity
        output_freq (int): Number of timesteps between outputting position
    Returns:
        - (float): Stopping distance
        - (pd.DataFrame): Projectile trajectory
    """
    from keras import backend as K
    import tensorflow as tf
    import pickle as pkl
    import os
    
    # Make Keras run serially
    os.environ['OMP_NUM_THREADS'] = "1"
    cfg = tf.ConfigProto()
    cfg.inter_op_parallelism_threads = 1
    cfg.intra_op_parallelism_threads = 1
    sess = tf.Session(config=cfg)
    K.set_session(sess)

    # Load in the computer
    with open('stop_dist_computer.pkl', 'rb') as fp:
        computer = pkl.load(fp)

    # Run the calculation
    return computer.compute_stopping_distance(start_point, start_velocity,
                                              output=output_freq, status=False)


# Generate random starting points and directions on the unit sphere
u = np.random.uniform(-1, 1, size=(args.n_samples, 1))
v = np.random.uniform(0, 2 * np.pi, size=(args.n_samples, 1))
velocities = np.hstack((
    np.sqrt(1 - u ** 2) * np.cos(v),
    np.sqrt(1 - u ** 2) * np.sin(v),
    u
))
velocities *= args.velocity
positions = np.random.uniform(size=(args.n_samples, 3))

# Launch the calculations
futures = [compute_stopping_distance(u, v) for u, v
           in zip(positions, velocities)]

# Prepare the output directory and determine starting number
output_dir = f'v={args.velocity:.2f}'
result_file = os.path.join(output_dir, 'stop_dists.csv')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
    with open(result_file, 'w') as fp:
        print('run,stopping_dist', file=fp)
run_number = len(glob(os.path.join(output_dir, 'traj_*.json')))

# Store results as they are completed
for future in tqdm(as_completed(futures), total=len(futures)):
    distance, traj = future.result() # Unpack result

    # Append the stopping distance to a running record
    with open(result_file, 'a') as fp:
        print(f'{run_number},{distance}', file=fp)

    # Store the trajectory as a json document
    traj.to_json(os.path.join(output_dir, f'traj_{run_number}.json'))

    # Increment run number
    run_number += 1
