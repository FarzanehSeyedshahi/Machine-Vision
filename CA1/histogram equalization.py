from matplotlib import pyplot as plt
import numpy as np

def histEqual(array):
    #defining data
    cal = 0
    cdf = []
    pr = []
    y = array
    x = np.arange(len(y))
    prs = np.zeros(len(x))
    totall = np.sum(y, dtype=np.int32)

    #calculate probability and cdf
    for i in y:
         p = i/totall
         cal = cal + p
         pr.append(p)
         cdf.append(int(round(7*cal)))

    #calculate the result of equalization
    for i in range(0,len(prs)):
        for j in range(0,len(cdf)):
            if i == cdf[j]:
                prs[cdf[j]]=prs[cdf[j]]+ pr[j]
            elif cdf[j]>i:
                break
    #plot
    plt.figure(figsize=(9,3))
    plt.subplot(131)
    plt.scatter(x,cdf )
    plt.subplot(132)
    plt.scatter(x,pr)
    plt.subplot(133)
    plt.scatter(x,prs)
    plt.suptitle('Categorical Plotting')
    plt.show()
    
if __name__ == '__main__':
    histEqual(np.array([790,1023,850,656,329,245,122,81]))
