import os
import sys
sys.path.append(os.path.dirname(__file__).rsplit('/', 1)[0])
from build import fastpreprocess
import numpy as np
import cv2
import tqdm

frame = cv2.imread("test_example/demo.jpg").astype(np.uint8)
print("input: ", frame.shape)

TEST_TIMES = 1000

batch_size = 16

# BGR
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape((3, 1, 1))
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape((3, 1, 1))

batched_input = np.array([frame] * batch_size)
batched_results1 = np.zeros((batch_size, 3, frame.shape[0], frame.shape[1]), dtype=np.float32)

for i in tqdm.trange(TEST_TIMES):
    batched_results2 = batched_input.transpose(0, 3, 1, 2)
    batched_results2 = (batched_results2 - np.array([MEAN])) / np.array([STD])
for i in tqdm.trange(TEST_TIMES):
    for bi, mini_batch_input in enumerate(batched_input):
        batched_results1[bi] = fastpreprocess.hwc_to_chw_normalize(mini_batch_input, [103.53, 116.28, 123.675], [57.375, 57.12, 58.395], flip_rb = False)
for i in tqdm.trange(TEST_TIMES):
    batched_results2 = fastpreprocess.hwc_to_chw_normalize_batched(batched_input, [103.53, 116.28, 123.675], [57.375, 57.12, 58.395], flip_rb = False)

print(batched_results1.shape, batched_results2.shape)
print(batched_results1[0, 1, 10: 15, 10: 15])
print(batched_results2[0, 1, 10: 15, 10: 15])