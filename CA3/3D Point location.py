import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import math

def findFundMatrix(a,b):
    A = []
    a = np.array(a)
    b = np.array(b)
    for i in range (0,7):
        temp = [(a[i][0]*b[i][0]), (a[i][0]*b[i][1]), a[i][0], (a[i][1]*b[i][0]), (a[i][1]*b[i][1]), a[i][1], b[i][0], b[i][1], 1]
        A.append(temp)

    u, s, vh = np.linalg.svd(A)
    hp = vh[-1,:]
    hn = hp / hp[-1]
    FunMat =hn.reshape((3,3))
    return FunMat

def findPandPprim(finalF):
    u, s, vh = np.linalg.svd(finalF)
    e = vh[-1].reshape(3, 1)
    
    P = np.append(np.identity(3),np.zeros([3,1]),1)
    eprim = [[0, -e[2], e[1]],
             [e[2], 0, -e[0]],
             [-e[1], e[0], 0]]

    Pprim = np.append(np.dot(eprim, finalF),eprim,1)
    return P,Pprim

def threeDPointX(P,Pprim,pts1,pts2):
    A = np.array([
    [pts1[0][1] * P[2,0] - P[1,0], pts1[0][1] * P[2,1] - P[1, 1], pts1[0][1] * P[2,2] - P[1, 2]],
    [P[0,0] - pts1[0][0] * P[2,0], P[0,1] - pts1[0][0] * P[2,1], P[0,2] - pts1[0][0] * P[2,2]],
    [pts2[0][1] * Pprim[2,0] - Pprim[1,0], pts2[0][1] * Pprim[2,1] - Pprim[1,1], pts2[0][1] * Pprim[2,2] - Pprim[1,2]],
    [Pprim[0,0] - pts2[0][0] * Pprim[2,0], Pprim[0,1] - pts2[0][0] * Pprim[2,1], Pprim[0,2] - pts2[0][0] * Pprim[2, 2]]
     ], dtype='float')
    u, s, vh = np.linalg.svd(A)
    Aprime = vh[-1, :]
    X = Aprime / Aprime[-1]
    print('3D point location = ' ,X)
    return X


if __name__=='__main__':
    ######### first : find matched points
    img0 = cv2.imread('im_0.png',0)  
    img1 = cv2.imread('im_1.png',0)
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img0,None)
    kp2, des2 = sift.detectAndCompute(img1,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []
    good = []
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    ######### secound : compute Fundamental Matrix
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    bestF, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    
    for c in range(1000):
        r = 0
        maxR = 2000
        thr = 100
        sample1 =[]
        sample2 =[]
        for i in range (0,7):
            j = random.randint(0,len(pts1)-1)        
            sample1.append([pts1[j][0],pts1[j][1]])
            sample2.append([pts2[j][0],pts2[j][1]])
    
        F = np.array(findFundMatrix(sample1,sample2))
        FT = np.transpose(F)
        

        for i in range(0, len(pts1)-1):
            epL1 = np.array(np.matmul(F, [pts1[i][0],pts1[i][1],1]))
            epL2 = np.array(np.matmul(FT,[pts2[i][0],pts2[i][1],1]))

            d1 = abs(np.matmul(epL2, [pts1[i][0],pts1[i][1],1])/ math.sqrt(epL1[0]**2+epL1[1]**2))
            d2 = abs((np.matmul(epL1, [pts2[i][0],pts2[i][1],1])/ math.sqrt(epL2[0]**2+epL2[1]**2)))

            if d1<thr and d2<thr:
                r = r + 1
        if r> maxR :
            finalF = F
            maxR = r
        else:
            finalF = bestF
    
    P,Pprim = findPandPprim(finalF)
    print(Pprim)
    X = threeDPointX(P,Pprim,pts1,pts2)
    

