from utils import *


FILENAME_ph = "datasets/can/ph/low_dim_v141.hdf5"
FILENAME_mg = "datasets/can/mg/low_dim_sparse_v141.hdf5"


airl_iterations = 3000
TRAJ_DIV = 2

T_PH, T_MG, state_mp = get_trajs(FILENAME_ph, FILENAME_mg, 3)
print("finished collecting data")

method = "intrinsic"
REW_MP = get_reward_models([T_PH, T_MG], airl_iterations, \
    state_mp, TRAJ_DIV, method)

for k in range(5):
    compute_distances(REW_MP, method)
    print("")
