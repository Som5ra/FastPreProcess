import os
import sys
sys.path.append(os.path.dirname(__file__).rsplit('/', 1)[0])
from build import fastpreprocess
import numpy as np
import cv2
import tqdm

frame = cv2.imread("test_example/demo.jpg").astype(np.uint8)
print("input: ", frame.shape)

TEST_TIMES = 100


# BGR
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape((3, 1, 1))
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape((3, 1, 1))


rgb_frame1 = frame.transpose(2, 0, 1)
rgb_frame1 = (rgb_frame1 - MEAN) / STD
rgb_frame1 = rgb_frame1[::-1]
print(rgb_frame1[0, 0: 3, 0: 3])


rgb_frame2 = fastpreprocess.hwc_to_chw_normalize(frame, [103.53, 116.28, 123.675], [57.375, 57.12, 58.395], flip_rb = True)
print(rgb_frame2[0, 0: 3, 0: 3])


print("".join(["*"] * 100))


TEST_TIMES = 1000

for i in tqdm.trange(TEST_TIMES):
    frame1 = frame.transpose(2, 0, 1)
    frame1 = (frame1 - MEAN) / STD

for i in tqdm.trange(TEST_TIMES):
    frame2 = np.ascontiguousarray(frame.transpose(2, 0, 1))
    frame2 = fastpreprocess.chw_channel_normalize(frame2, [103.53, 116.28, 123.675], [57.375, 57.12, 58.395])
    # frame1 = (frame1 - MEAN) / STD

for i in tqdm.trange(TEST_TIMES):
    frame3 = fastpreprocess.hwc_to_chw(frame).astype(np.uint8)
    frame3 = fastpreprocess.chw_channel_normalize(frame3, [103.53, 116.28, 123.675], [57.375, 57.12, 58.395])
    # frame3 = fastpreprocess.hwc_to_chw_normalize(frame1, [103.53, 116.28, 123.675], [57.375, 57.12, 58.395])


for i in tqdm.trange(TEST_TIMES):
    frame4 = fastpreprocess.hwc_to_chw_normalize(frame, [103.53, 116.28, 123.675], [57.375, 57.12, 58.395], flip_rb = False)

print(frame[0: 3, 0: 3, 0])
print(frame1[0, 0: 3, 0: 3])
print(frame2[0, 0: 3, 0: 3])
print(frame3[0, 0: 3, 0: 3])
print(frame4[0, 0: 3, 0: 3])


