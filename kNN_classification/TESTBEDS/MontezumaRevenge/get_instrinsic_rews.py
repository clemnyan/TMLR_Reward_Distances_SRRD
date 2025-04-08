from utils import *


DATA_DIR = "DATA/rew_state_mp.pkl"

gamma = 0.2
iterations = 3000
TRAJ_DIV = 1


with open(DATA_DIR, "rb") as f:
    traj_mp, state_mp = pickle.load(f)
print("finished collecting data")

method = "intrinsic"
REW_MP = get_reward_models(traj_mp, iterations, \
    state_mp, TRAJ_DIV, method)


res = []
for k in range(10):
    curr = compute_distances(REW_MP, method, gamma)
    print("")
    res += curr

res = np.array(res)
