from utils import *

FILENAME_ph = "datasets/can/ph/low_dim_v141.hdf5"
FILENAME_mg = "datasets/can/mg/low_dim_sparse_v141.hdf5"

gamma = 0.5
maxent_iterations = 100
TRAJ_DIV = 2

T_PH, T_MG, state_mp = get_trajs(FILENAME_ph, FILENAME_mg, 3)
print("finished collecting data")

method = "maxent"
REW_MP = get_reward_models([T_PH, T_MG], maxent_iterations, \
    state_mp, TRAJ_DIV, method)

res = []
for k in range(10):
    curr = compute_distances(REW_MP, method, gamma)
    print("")
    res += curr

res = np.array(res)
