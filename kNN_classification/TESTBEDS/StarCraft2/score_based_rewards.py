
import os
import pandas as pd
import numpy as np
from utilities import *
import multiprocessing


arr_set = divide_arr(get_score_args(), 30)

res_common = {}

count = 0
for i, args in enumerate(arr_set):
    print("finished running arg number = {}/{}".format(i, len(arr_set)))
    with multiprocessing.Pool() as pool:
        result = pool.map(sc2_distance_trial, args)
        pool.close()
        pool.join()

        for r in result:
            print("result number = ", count)
            count += 1

            r_label, r_common = r
            if r_label not in res_common:
                res_common[r_label] = [list(r_common)]
            else:
                res_common[r_label].append(list(r_common))

#import pickle
#res_common = pickle.load(open('res_common.pkl', 'rb'))

class_stats = []
for label in res_common:
    data = np.array(res_common[label])

    class_stats.append([label, len(data), \
        np.mean(data[:, 1]), np.mean(data[:, 2]), np.mean(data[:, 3]),
        np.std(data[:, 1]), np.std(data[:, 2]), np.std(data[:, 3])])

df = pd.DataFrame(class_stats, columns = ['label', 'data_len', \
    'epic_mean', 'fird_mean', 'rew_mean', \
    'epic_std', 'fird_std', 'rew_std'])
