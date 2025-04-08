import os
import pandas as pd
from utilities import *

selected_classes = str([1, 2])
#selected_classes = str([i for i in range(90) if (i%70 == 0)])
#selected_classes = str([i for i in range(90) if (i%3 == 0)])
num_trajs = 10


DATA_DIR = 'SC2/NEW_PROCESSED_FILES'
save_dir = '/tmp/CNEIGH_'

m = selected_classes.replace("[", '').replace("]", '').replace(' ', '').split(',')
class_arr = [int(i) for i in m if i!= '']

# process the directory to save results
s = str(selected_classes).replace(' ', '').replace(',', '_').replace('  ', '')
s = s.replace('[', '_').replace(']','_')
s += ('___' +  str(num_trajs))
TEMP_DIR = 'TEMP_FILES' + s
print(TEMP_DIR)

dir_mp = set_up_directories(DATA_DIR, TEMP_DIR, save_dir, class_arr, num_trajs)

def sc2_ptirl ():
    """ Perform the PT-IRL algorithm to get the rewards """
    for _ in range(1):
        make_run(TEMP_DIR, dir_mp)

sc2_ptirl()
