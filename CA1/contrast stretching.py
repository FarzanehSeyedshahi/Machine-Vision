import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
img1 = cv2.imread('C:/Users/farzaneh/Desktop/farzaneh/Courses/machine vision/hw1/x-ray.PNG',0)

# Create zeros array to store the stretched image
minmax_img = np.zeros((img1.shape[0],img1.shape[1]),dtype = 'uint8')
percentile_minmax_img = np.zeros((img1.shape[0],img1.shape[1]),dtype = 'uint8')

# Loop over the image and apply Min-Max formulae
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        percentile_minmax_img[i,j] = 255*(img1[i,j]-70)/(100)
        minmax_img[i,j] = 255*(img1[i,j]-np.min(img1))/(np.max(img1)-np.min(img1))

# Displat the stretched image
hist1 = cv2.calcHist([img1],[0],None,[255],[0,255])
hist2 = cv2.calcHist([minmax_img],[0],None,[255],[0,255])
hist3 = cv2.calcHist([percentile_minmax_img],[0],None,[255],[0,255])

plt.figure(1)
plt.subplot(131),plt.imshow(img1, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(minmax_img, cmap = 'gray')
plt.title('min-max image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(percentile_minmax_img, cmap = 'gray')
plt.title('percentile min-max image'), plt.xticks([]), plt.yticks([])
plt.figure(2)
plt.subplot(131),plt.plot(hist1)
plt.title('min max histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.plot(hist2)
plt.title('original histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.plot(hist3)
plt.title('percentile histogram'), plt.xticks([]), plt.yticks([])

plt.show()

