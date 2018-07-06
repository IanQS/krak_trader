from constants import STORAGE_PATH
import os

small_test_file = ''
SMALL_TEST_FILE = STORAGE_PATH.format(small_test_file)

large_test_file = '1530759203.3338196.npz'
LARGE_TEST_FILE = STORAGE_PATH.format(large_test_file)

ALL_DATA = [STORAGE_PATH.format(f)
            for f in os.listdir(STORAGE_PATH.split('{}')[0])
            if f.endswith('npz')]
