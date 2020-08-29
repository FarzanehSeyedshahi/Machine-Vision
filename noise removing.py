import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

img = cv2.imread('C:/Users/farzaneh/Desktop/farzaneh/Courses/machine vision/hw1/noiseball.PNG',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.figure(1)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

im_fft2 = fshift
im_fft2[73:83,130:140] = 0
im_fft2[73:83,165:175] = 0
im_fft2[175:185,145:155] = 0
im_fft2[175:185,180:190] = 0
im_fft2[180:190,160] = 0
im_fft2[65:75,160] = 0

plt.figure(2)
cf3 =np.fft.ifft2((np.abs(im_fft2))*np.cos(np.angle(fshift))+1j*(np.abs(im_fft2))*np.sin(np.angle(fshift)))
plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.subplot(122),plt.imshow(20*np.log(np.abs(im_fft2)), cmap = 'gray')

plt.figure(3)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.subplot(122),plt.imshow(np.abs(cf3), cmap = 'gray')
plt.show()
