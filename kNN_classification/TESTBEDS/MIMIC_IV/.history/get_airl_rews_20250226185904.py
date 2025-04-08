from utils import *


DATA_DIR = "health_data.pkl"

gamma = 0.8
iterations = 7000
TRAJ_DIV = 1


with open(DATA_DIR, "rb") as f:
    
    traj_mp, state_mp = pickle.load(f)
print("finished collecting data")

method = "airl"
REW_MP = get_reward_models(traj_mp, iterations, \
    state_mp, TRAJ_DIV, method)

res = []
for k in range(10):
    curr = compute_distances(REW_MP, method, gamma)
    print("")
    res += curr

res = np.array(res)
