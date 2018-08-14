from constants import KRAKEN_PATH
import os

ZERO_PLACEHOLDER = 1e-10  # When we try to normalize, we may encounter 0s, so we

SUMMARY_PATH = './summaries/{}'

CONV_INPUT_SHAPE = [100, 2, 2]

small_test_file = ''
SMALL_TEST_FILE = KRAKEN_PATH.format(small_test_file)

large_test_file = '1530759203.3338196.npz'
LARGE_TEST_FILE = KRAKEN_PATH.format(large_test_file)

ALL_DATA = [KRAKEN_PATH.format(f)
            for f in os.listdir(KRAKEN_PATH.split('{}')[0])
            if f.endswith('npz')]
