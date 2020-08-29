import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def hough_line(img):
  w,h = img.shape
  theta = np.linspace(-90.0, 0.0, 90.0)
  theta = np.concatenate((theta, -theta[len(theta)-2::-1]))

  dist = np.sqrt((w - 1)**2 + (h - 1)**2)
  rho = np.linspace(-dist, dist, 2*dist+1)
  H = np.zeros((len(rho), len(theta)))
  
  for j in range(w):
    for i in range(h):
      if img[j, i]:
        for k in range(len(theta)):
            
          rhoVal = i*np.cos(theta[k]*np.pi/180.0) + j*np.sin(theta[k]*np.pi/180)
          
          rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
          
          H[rhoIdx[0], k] += 1
  return rho, theta, H

def max_voting(ht_acc_matrix, n, rhos, thetas):
    rho_theta = []
    x_y = []
    flat = list(set(np.hstack(ht_acc_matrix)))
    flat_sorted = sorted(flat, key = lambda n: -n)
    coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
    for coords_for_val_idx in range(0, len(coords_sorted), 1):
      coords_for_val = coords_sorted[coords_for_val_idx]
      for i in range(0, len(coords_for_val), 1):
        n,m = coords_for_val[i]
        rho = rhos[n]
        theta = thetas[m]
        rho_theta.append([rho, theta])
        x_y.append([m, n])
    return [rho_theta[0:n], x_y]

def draw_rho_theta_pairs(target_im, pairs):
  y, x, channels = np.shape(target_im)
  for i in range(0, len(pairs), 1):
    point = pairs[i]
    rho = point[0]
    theta = point[1] * np.pi / 180
    if theta != 0:
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        left = (0, b)
        right = (x, x * m + b)
        top = (-b / m, 0)
        bottom = ((y - b) / m, y)

        pts = [pt for pt in [left, right, top, bottom]
               if (pt[0] <= x and pt[0] >= 0 and pt[1] <= y and pt[1] >= 0)]
        
        if len(pts) == 2:
            a,b = [int(round(num)) for num in pts[0]]
            c,d = [int(round(num)) for num in pts[1]]
            cv2.line(target_im, (a,b),(c,d) , (0,255,0))


if __name__== '__main__':
    img = cv2.imread('pathway.jpg')
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),200,400)
    rhos, thetas, H = hough_line(edges)
    rho_theta_pairs, x_y_pairs = max_voting(H, 15, rhos, thetas)
    im_w_lines = img.copy()
    draw_rho_theta_pairs(im_w_lines, rho_theta_pairs)
    plt.imshow(im_w_lines, cmap = 'gray')
    plt.show()
