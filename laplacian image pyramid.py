import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/Users/farzaneh/Desktop/farzaneh/Courses/machine vision/hw1/building.jpg")
# Gaussian Pyramid
layer = img.copy()
gaussian_pyramid = [layer]
for i in range(3):
    layer = cv2.pyrDown(layer)
    cv2.imshow(str(i), layer)
    gaussian_pyramid.append(layer)
    
# Laplacian Pyramid
gaussian_expanded_list=[]
layer = gaussian_pyramid[3]
laplacian_pyramid = [layer]
for i in range(3, 0, -1):
    size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
    gaussian_expanded_list.append(gaussian_expanded)
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
    laplacian_pyramid.append(laplacian)

#reconstructing image
small = gaussian_pyramid[3]
level2Reconstructed = cv2.add(gaussian_expanded_list[0],laplacian_pyramid[1])
cv2.imshow("3+", level2Reconstructed)
cv2.imshow("3", gaussian_expanded_list[0])

plt.show()
