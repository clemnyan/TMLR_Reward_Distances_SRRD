import os
import shutil

# rm TEMP_FILES

try:
    shutil.rmtree('TEMP_FILES')
except:
    print("TEMP_FILES not existent")

try:
    shutil.rmtree("DATA/PROCESSED_DATA")
except:
    print("PROCESSED_DATA not existent")
