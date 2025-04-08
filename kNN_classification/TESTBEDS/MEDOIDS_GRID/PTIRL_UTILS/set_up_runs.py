import os
import pandas as pd
from utilities import *


selected_classes = str([3, 7, 5, 9])

num_trajs = 5

DATA_DIR = 'NEW_DATA'
save_dir = '/tmp/CNEIGH_'

m = selected_classes.replace("[", '').replace("]", '').replace(' ', '').split(',')
class_arr = [int(i) for i in m if i!= '']

# process the directory to save results
s = str(selected_classes).replace(' ', '').replace(',', '_').replace('  ', '')
s = s.replace('[', '_').replace(']','_')
s += ('___' +  str(num_trajs))
TEMP_DIR = 'TEMP_FILES' + s

dir_mp = set_up_directories(DATA_DIR, TEMP_DIR, save_dir, class_arr, num_trajs)

def sc2_ptirl ():
    #Perform the PT-IRL algorithm to get the rewards
    for _ in range(10):
        make_run(TEMP_DIR, dir_mp)

sc2_ptirl()
