from utils import *


DATA_DIR = "DATA/rew_state_mp.pkl"


iterations = 3000
TRAJ_DIV = 2


with open(DATA_DIR, "rb") as f:
    traj_mp, state_mp = pickle.load(f)

print(len(traj_mp))
sys.exit()

print("finished collecting data")

method = "intrinsic"
REW_MP = get_reward_models([DATA_DIR, iterations, \
    state_mp, TRAJ_DIV, method])

for k in range(5):
    compute_distances(REW_MP, method)
    print("")
