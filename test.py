import cv2
import glob
import numpy as np

img_paths = glob.glob(r"./data/images/**/**/**.jpg")
depth_img_paths = glob.glob(r"./data/images(depth)/**/**/**.png")

for idx, (img_p, depth_p) in enumerate(zip(img_paths, depth_img_paths)):
    img = cv2.imread(img_p)
    depth = cv2.imread(depth_p, cv2.IMREAD_GRAYSCALE)
    depth = np.expand_dims(depth, axis=2)
    combine = np.concatenate((img, depth), axis=2)
    print(combine.shape)
    input("C")