from utils import *
import pickle 

DATA_DIR = "health_data.pkl"

gamma = 0.8
iterations = 7000
TRAJ_DIV = 1


with open(DATA_DIR, "rb") as f:
    mp_diabetes_resp, mp_diabetes_substance, mp_brain_substance, mp_resp_substance = pickle.load(f)

state_mp, action_mp = {}, {}
s1 = get_state_action_mp(mp_diabetes_resp, state_mp, action_mp)

print("finished collecting data")

"""
method = "airl"
REW_MP = get_reward_models(traj_mp, iterations, \
    state_mp, TRAJ_DIV, method)

res = []
for k in range(10):
    curr = compute_distances(REW_MP, method, gamma)
    print("")
    res += curr

res = np.array(res)
"""