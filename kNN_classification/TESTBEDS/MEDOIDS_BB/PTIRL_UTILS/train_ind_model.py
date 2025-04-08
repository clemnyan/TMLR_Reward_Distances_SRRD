from utils import *
import sys



TRAJ_DIR = "TEMP_FILES_3_7_5_9____5"

progress_check = TRAJ_DIR + '/rewards_prog'
if not os.path.isdir(progress_check):
    os.mkdir(progress_check)

rewards_store = []
for _ in range(10):
    p_len = len(os.listdir(progress_check))
    print("total len = ", len(os.listdir(TRAJ_DIR + '/res_store_folder')))
    if len(os.listdir(TRAJ_DIR + '/res_store_folder')) > p_len:   # means we are not done

        f_name = progress_check + '/res' + str(p_len)

        with open(f_name, 'wb') as f:
            pickle.dump([], f)
        unravel_trajs(TRAJ_DIR, p_len, f_name, rewards_store)

    else:
        if not os.path.isfile("ptirl_rewards.pkl"):
            with open("ptirl_rewards.pkl", "wb") as f:
                pickle.dump(rewards_store, f)
