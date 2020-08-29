import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def read_images():
    images =[]
    mp1 = cv2.imread('monalisa_piece1.jpg')
    images.append(mp1)
    images.append(ndimage.rotate(mp1, 15))
    images.append(ndimage.rotate(mp1, 20))
    images.append(increase_brightness(mp1,30))

    mp2 = cv2.imread('monalisa_piece2.jpg')
    images.append(mp2)
    images.append(ndimage.rotate(mp2, 15))
    images.append(ndimage.rotate(mp2, 20))
    images.append(increase_brightness(mp2,30))

    mp3 = cv2.imread('monalisa_piece3.jpg')
    images.append(mp3)
    images.append(ndimage.rotate(mp3, 15))
    images.append(ndimage.rotate(mp3, 20))
    images.append(increase_brightness(mp3,30))

    mp4 = cv2.imread('monalisa_piece4.jpg')
    images.append(mp4)
    images.append(ndimage.rotate(mp4, 15))
    images.append(ndimage.rotate(mp4, 20))
    images.append(increase_brightness(mp4,30))

    return images


def siftImpl(monalisa_ORG, monalisa_piece):
    ratios = []
    k1, d1 = sift.detectAndCompute(monalisa_ORG, None)
    k2, d2 = sift.detectAndCompute(monalisa_piece, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)
    for m, n in matches:
        if m.distance < 0.72 * n.distance:
            ratios.append([m])
    matched_image = cv2.drawMatchesKnn(monalisa_ORG, k1, monalisa_piece, k2,ratios, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return matched_image

if __name__=='__main__':
    monalisa_ORG = cv2.imread('monalisa.jpg')
    sift = cv2.xfeatures2d.SIFT_create()
    images = []
    images = read_images()
    for i in range(len(images)):
        matched_image = siftImpl(monalisa_ORG, images[i])
        plt.figure()
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.show()


