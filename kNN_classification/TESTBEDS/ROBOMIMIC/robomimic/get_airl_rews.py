from utils import *

FILENAME_ph = "datasets/can/ph/low_dim_v141.hdf5"
FILENAME_mg = "datasets/can/mg/low_dim_sparse_v141.hdf5"

gamma = 0.8
airl_iterations = 5000
TRAJ_DIV = 2

T_PH, T_MG, state_mp = get_trajs(FILENAME_ph, FILENAME_mg, 3)
print("finished collecting data")

method = "airl"
REW_MP = get_reward_models([T_PH, T_MG], airl_iterations, \
    state_mp, TRAJ_DIV, method)

res = []
for k in range(10):
    curr = compute_distances(REW_MP, method, gamma)
    print("")
    res += curr

res = np.array(res)
