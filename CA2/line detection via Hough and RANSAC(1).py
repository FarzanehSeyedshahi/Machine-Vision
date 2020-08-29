import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ransacImpl(edgesPoints):
    count = 0
    total_count = 0
    m_total = 0
    b_total = 0
    for i in range(1000):
        k = np.random.choice(len(edgesPoints), 2, replace=False)
        m = (edgesPoints[k[1]][0] - edgesPoints[k[0]][0]) / (edgesPoints[k[1]][1] - edgesPoints[k[0]][1] + 0.0000000000000000000000001)
        b = (-m * edgesPoints[k[1]][1]) + edgesPoints[k[0]][0]
        for i, j in edgesPoints:
            if (abs((i - (m * j) - b))/(math.sqrt(1 + m ** 2))) <= threshold:
                count = count+1
        if count > total_count:
            m_total = m
            b_total = b
            total_count = count
        count = 0
    return m_total,b_total,total_count

if __name__== '__main__':
    img = cv2.imread('horizons.jpg')
    threshold = 0.1
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),254,255)
    edgePoints = [(x, y) for x in range(edges.shape[0]) for y in range(edges.shape[1]) if edges[x, y] == 255]
    m,b,n = ransacImpl(edgePoints)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x = np.linspace(0,edges.shape[1],500,endpoint=False)
    plt.plot(x , m*x+b , 'r')
    plt.show()
    


