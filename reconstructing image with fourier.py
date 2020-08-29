import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/farzaneh/Desktop/farzaneh/Courses/machine vision/hw1/building.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
magnitude_phase = np.angle(fshift)

plt.figure(1)
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(magnitude_phase, cmap = 'gray')
plt.title('Magnitude phase'), plt.xticks([]), plt.yticks([])

cf1=np.fft.ifft2(np.cos(magnitude_phase)+1j*np.sin(magnitude_phase))
cf2=np.fft.ifft2(np.abs(fshift))
cf3 =np.fft.ifft2((np.abs(fshift))*np.cos(magnitude_phase)+1j*(np.abs(fshift))*np.sin(magnitude_phase))
plt.figure(2)
plt.subplot(131),plt.imshow(np.abs(cf1), cmap = 'gray')
plt.title('|F| = 1'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(np.abs(cf2), cmap = 'gray')
plt.title('tetha = 0'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(np.abs(cf3), cmap = 'gray')
plt.title('Reconstructed image'), plt.xticks([]), plt.yticks([])

plt.show()
