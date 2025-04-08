from utils import *


FILENAME_ph = "datasets/lift/ph/low_dim_v141.hdf5"
FILENAME_mg = "datasets/lift/mg/low_dim_sparse_v141.hdf5"



airl_iterations = 3000
TRAJ_DIV = 10

T_PH, T_MG, state_mp = get_trajs(FILENAME_ph, FILENAME_mg, 3)
print("finished collecting data")


REW_MP = get_reward_models([T_PH[:6], T_MG[:6]], airl_iterations, \
    state_mp, TRAJ_DIV)

compute_distances(REW_MP)
