from concurrent.futures import as_completed
from parsl.config import Config
from parsl import python_app
from tqdm import tqdm
from glob import glob
import numpy as np
import argparse
import parsl
import os


# Make an argument parser
parser = argparse.ArgumentParser(description="Run a stopping distance simulation in a single direction")
parser.add_argument('--config', help='Parsl configuration', default='local')
parser.add_argument('--direction', nargs=3, type=int, default=[1, 0, 0], help='Direction vector')
parser.add_argument('--random-dir', action='store_true', help='Projectiles move in a random direction')
parser.add_argument('--random-seed', default=1, type=int)
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
                max_workers=os.cpu_count() // 2,
                cpu_affinity='block',
                provider=LocalProvider(max_blocks=1, init_blocks=1)
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
                cpu_affinity='block',
                provider=LocalProvider(
                    launcher=AprunLauncher(),
                    init_blocks=nnodes,
                    max_blocks=nnodes,
                    worker_init='''
                    export PATH="/home/lward/miniconda3/bin:$PATH"
                    source activate ml_tddft
                    '''
                ),
            )
        ],
    )
else:
    raise ValueError(f'Unrecognized configuration: {args.config}')

parsl.load(config)


# Make the Python function
@python_app
def compute_stopping_distance(start_point, start_velocity, output_freq=10):
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
    
    # Load in the computer
    with open('stop_dist_computer.pkl', 'rb') as fp:
        computer = pkl.load(fp)

    # Run the calculation
    return computer.compute_stopping_distance(start_point, start_velocity,
                                              output=output_freq, status=False)


# Generate random starting points and directions on the unit sphere
rng = np.random.RandomState(args.random_seed)
if not args.random_dir:
    velocity = np.array(args.direction, dtype=np.float)
    velocity *= args.velocity / np.linalg.norm(velocity)
    velocities = np.tile(velocity, (args.n_samples, 1))
    output_dir = f'v={args.velocity:.2f}-d={"_".join(map(str, args.direction))}'
else:
    u = rng.uniform(-1, 1, size=(args.n_samples, 1))
    v = rng.uniform(0, 2 * np.pi, size=(args.n_samples, 1))
    velocities = np.hstack((
        np.sqrt(1 - u ** 2) * np.cos(v),
        np.sqrt(1 - u ** 2) * np.sin(v),
        u
    )) * args.velocity
    output_dir = f'v={args.velocity:.2f}-d=random'
positions = rng.uniform(size=(args.n_samples, 3))

# Launch the calculations
futures = []
for run_id, (u, v) in enumerate(zip(positions, velocities)):
    # Skip if already done
    out_path = os.path.join(output_dir, f'traj_{run_id}.json')
    if os.path.isfile(out_path):
        continue
    
    # Otherwise submit and store the output path
    future = compute_stopping_distance(u, v)
    future.run_id = run_id
    future.out_path = out_path
    futures.append(future)
    
# Prepare the output directory and determine starting number
result_file = os.path.join(output_dir, 'stop_dists.csv')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
    with open(result_file, 'w') as fp:
        print('run,stopping_dist', file=fp)

# Store results as they are completed
for future in tqdm(as_completed(futures), total=len(futures)):
    try:
        distance, traj = future.result() # Unpack result
    except BaseException as exc:
        print(f'Run failed due to: {exc}')
        continue

    # Append the stopping distance to a running record
    with open(result_file, 'a') as fp:
        print(f'{future.run_id},{distance}', file=fp)

    # Store the trajectory as a json document
    traj.to_json(future.out_path, orient='split')
