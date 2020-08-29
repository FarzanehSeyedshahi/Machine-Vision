import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import scipy
import image
import math

def sobel(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndi.filters.convolve(img, Kx)
    Iy = ndi.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    plt.figure("second step")
    plt.imshow(G, cmap = 'gray')
    return (G, theta)

def non_maximum_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

if __name__ =='__main__':
    
    #first step
    img = cv2.imread('C:/Users/farzaneh/Desktop/farzaneh/Courses/machine vision/hw1/building.jpg',0)
    sigma = 1
    imgdata = np.array(img, dtype = float)
    G = ndi.filters.gaussian_filter(imgdata, sigma)
    plt.figure("first step")
    plt.imshow(G, cmap = 'gray')

    geradian, tetha = sobel(G)
    plt.figure("final image")
    final = non_maximum_suppression(geradian, tetha)
    plt.imshow(final, cmap = 'gray')
    plt.show()
