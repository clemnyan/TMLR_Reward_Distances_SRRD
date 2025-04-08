from utils import *
import sys


TRAJ_DIR = "TEMP_FILES_1_2____10"

progress_check = TRAJ_DIR + '/rewards_prog'
if not os.path.isdir(progress_check):
    os.mkdir(progress_check)

count = 10
p_len = len(os.listdir(progress_check))
print("total len = ", len(os.listdir(TRAJ_DIR + '/res_store_folder')))

for _ in range(10):

    count -= 1
    while len(os.listdir(TRAJ_DIR + '/res_store_folder')) > p_len and count > 0:   # means we are not done

        f_name = progress_check + '/res' + str(p_len)
        print("f_name = ", f_name)
        with open(f_name, 'wb') as f:
            pickle.dump([], f)

        r = unravel_trajs(TRAJ_DIR, p_len, f_name)
        p_len = len(os.listdir(progress_check))

        pickle.dump(r, open(f_name, 'wb'))

rew_mp = {}
for d in os.listdir(progress_check):
    f_new = progress_check + '/' + d
    print(f_new)

    with open(f_new, 'rb') as f:
        X = pickle.load(f)

        for element in X:
            rset, label = element[0]
            rew_mp[label] = rset

# store these rewards in the sc2 dir
with open('SC2/ptirl_rewards.pkl', 'wb') as f:
    pickle.dump(rew_mp, f)
