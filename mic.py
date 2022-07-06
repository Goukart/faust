import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch


# Function that down-samples image x number of times
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col//2, row//2))
    return image


# Load the DNN model
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")


def main():
    print("ll")

