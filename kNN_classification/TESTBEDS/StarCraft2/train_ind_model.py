from utils import *
import sys

"""
#selected_classes = [i for i in range(90) if (i%2 == 0) or (i%3 == 0)]
selected_classes = str([i for i in range(90) if (i%20 == 0)])
num_trajs = 7

# process the directory to save results
s = str(selected_classes).replace(' ', '').replace(',', '_').replace('  ', '')
s = s.replace('[', '_').replace(']','_')
s += ('___' +  str(num_trajs))
TRAJ_DIR = 'TEMP_FILES' + s
print(TRAJ_DIR)
"""
TRAJ_DIR = "TEMP_FILES_0_3_6_9_12_15_18_21_24_27_30_33_36_39_42_45_48_51_54_57_60_63_66_69_72_75_78_81_84_87____7"
#TRAJ_DIR = "TEMP_FILES_0_20_40_60_80____7"
#TRAJ_DIR = "TEMP_FILES_0_9____7"
CLASS_LEN = 10
class_mp = get_class_mp(CLASS_LEN)


progress_check = TRAJ_DIR + '/rewards_prog'
if not os.path.isdir(progress_check):
    os.mkdir(progress_check)


for _ in range(10):

    p_len = len(os.listdir(progress_check))

    print("total len = ", len(os.listdir(TRAJ_DIR + '/res_store_folder')))
    if len(os.listdir(TRAJ_DIR + '/res_store_folder')) > p_len:   # means we are not done

        f_name = progress_check + '/res' + str(p_len)
        with open(f_name, 'wb') as f:
            pickle.dump([], f)
        unravel_trajs(TRAJ_DIR, p_len, f_name)

    else:
        print("done now training ... ")
        # load results

        X, Y = [], []

        for file in os.listdir(progress_check):
            with open(progress_check + '/' + file, 'rb') as f:
                arr = pickle.load(f)

                for r in arr:
                    rew, act_lab = r
                    label = get_class(class_mp, CLASS_LEN, act_lab, 'dir')
                    X.append(rew)
                    Y.append(label)

        X, Y = np.array(X), np.array(Y)
        #X, Y = create_XY(res_arr)
        # train nueral network here
        network_res, history, model = feed_forward_classifier ([X, Y, TRAJ_DIR])
        res_ans = []

        c_mp = {class_mp[k]:k for k in class_mp}
        for val in network_res:
            a, b = val
            a_ans = [c_mp[index] for index, k in enumerate(a) if k == 1]
            b_ans = [c_mp[index] for index, k in enumerate(b) if k == 1]
            res_ans.append([a_ans, b_ans])

        class_rp , acc_mp = compute_stats(res_ans)
        sys.exit()
"""

import pickle
res_ans = pickle.load(open('binaryloss.p', 'rb'))
#res_ans = pickle.load(open('f1.p', 'rb'))
class_rp , acc_mp = compute_stats(res_ans[:10000])
print([np.mean(acc_mp[:, 0]), np.mean([acc_mp[:, 1]]), np.mean([acc_mp[:, 2]])])
"""
