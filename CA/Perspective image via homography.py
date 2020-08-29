import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
 
if __name__ == '__main__' :
    im_src = cv2.imread('pool.jpg')
    #pts_FLAGE = np.array([[361,560], [528,522], [438,595],[644, 550]]) 
    im_dst = cv2.imread('canada.PNG')
    #pts_dst = np.array([[0, 0],[316,0], [0, 156], [316, 156]])
    POOL = [[361, 438, 644, 528],[560, 595, 550, 522]]
    FLAGE = [[0, 0, 316, 316],[0, 156, 156, 0]]
    A = np.array([
         [ -FLAGE[0][0], -FLAGE[1][0], -1, 0, 0, 0, (FLAGE[0][0]*POOL[0][0]), (FLAGE[1][0]*POOL[0][0]), POOL[0][0]],
         [0, 0, 0, -FLAGE[0][0], -FLAGE[1][0], -1, (FLAGE[0][0]*POOL[1][0]), (FLAGE[1][0]*POOL[1][0]), POOL[1][0]],
         [ -FLAGE[0][1], -FLAGE[1][2], -1, 0, 0, 0, (FLAGE[0][1]*POOL[0][1]), (FLAGE[1][1]*POOL[0][1]), POOL[0][1]],
         [0, 0, 0, -FLAGE[0][1], -FLAGE[1][1], -1, (FLAGE[0][1]*POOL[1][1]), (FLAGE[1][1]*POOL[1][1]), POOL[1][1]],
         [ -FLAGE[0][2], -FLAGE[1][2], -1, 0, 0, 0, (FLAGE[0][2]*POOL[0][2]), (FLAGE[1][2]*POOL[0][2]), POOL[0][2]],
         [0, 0, 0, -FLAGE[0][2], -FLAGE[1][2], -1, (FLAGE[0][2]*POOL[1][2]), (FLAGE[1][2]*POOL[1][2]), POOL[1][2]],
         [ -FLAGE[0][3], -FLAGE[1][3], -1, 0, 0, 0, (FLAGE[0][3]*POOL[0][3]), (FLAGE[1][3]*POOL[0][3]), POOL[0][3]],
         [0, 0, 0, -FLAGE[0][3], -FLAGE[1][3], -1, (FLAGE[0][3]*POOL[1][3]), (FLAGE[1][3]*POOL[1][3]), POOL[1][3]]])

    u, s, vh = np.linalg.svd(A)
    hp = vh[-1,:]
    hn = hp / hp[-1]
    hemo_mtrx =hn.reshape((3,3))

    im1reg = cv2.warpPerspective(im_dst, hemo_mtrx, (im_src.shape[1], im_src.shape[0]))
    #im2reg =np.zeros((im_src.shape[1], im_src.shape[0]))
    final = cv2.bitwise_or(im_src,im1reg)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.show()
    
