from utils import *
import pickle 

DATA_DIR = "health_data.pkl"

gamma, iterations, TRAJ_DIV, method = 0.8, 7000, 1, "airl"

with open(DATA_DIR, "rb") as f:
    mp_diabetes_resp, mp_diabetes_substance, mp_brain_substance, mp_resp_substance = pickle.load(f)
print("done loading files ")

state_mp, action_mp = {}, {}
s1 = get_state_action_mp(mp_diabetes_resp, state_mp, action_mp)
s2 = get_state_action_mp(mp_diabetes_substance, state_mp, action_mp)
s3 = get_state_action_mp(mp_brain_substance, state_mp, action_mp)
s4 = get_state_action_mp(mp_resp_substance, state_mp, action_mp)

REW_MP = get_reward_models([s1, s2, s3, s4], iterations, state_mp, action_mp, TRAJ_DIV, method)

res = []
for k in range(10):
    curr = compute_distances(REW_MP, method, gamma)
    print("")
    res += curr

res = np.array(res)