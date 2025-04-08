from utils import *
import pickle 

DATA_DIR = "health_data.pkl"

gamma, iterations, TRAJ_DIV, method = 0.8, 7000, 2, "airl"

with open(DATA_DIR, "rb") as f:
    mp_diabetes, mp_kidney, mp_legs, mp_resp, mp_sub = pickle.load(f)
print("done loading files ")

state_mp, action_mp = {}, {}

s1 = get_state_action_mp(mp_diabetes, state_mp, action_mp) # type: ignore
s2 = get_state_action_mp(mp_kidney, state_mp, action_mp) # type: ignore
s3 = get_state_action_mp(mp_legs, state_mp, action_mp) # type: ignore
s4 = get_state_action_mp(mp_resp, state_mp, action_mp) # type: ignore
s5 = get_state_action_mp(mp_sub, state_mp, action_mp) # type: ignore


#REW_MP = get_reward_models([s1, s2, s3, s4,s5, s6, s7, s8, s9, s10], iterations, state_mp, action_mp, TRAJ_DIV, method)
REW_MP = get_reward_models([s1, s2, s3, s4,s5], iterations, state_mp, action_mp, TRAJ_DIV, method)

res = []
for k in range(10):
    curr = compute_distances(REW_MP, method, gamma)
    print("")
    res += curr

res = np.array(res)