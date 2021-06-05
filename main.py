#!/usr/bin/python3.8
import  matplotlib.pyplot as plt
import numpy as np
import random
import math


from numpy.core.fromnumeric import size

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def calculate_length(vec, d):
    sum = 0
    for i in range(0, d):      
        if d == 1:
            sum = sum + vec[i]
        else:
            sum = sum + vec[i] * vec[i]

    ret = 0
    if d == 1:
        ret = sum
    else:
        ret = math.sqrt(sum)

    return ret


# n - numer of steps total
# d - dimention
# len - length of one step
def calculate_steps(n, d, len = 1):
    # array containing positions in d dimentions
    # starting from point (0, 0, ...)
    p = np.zeros((1, d))

    # create list of possible vectors
    # unit vectors with "-units"
    e = []

   

    for k in range(d):
        l1 = np.zeros((1,d))
        l2 = np.zeros((1,d))
        l1[0][k] = 1
        l2[0][k] = -1
        #print(l1)
        #print(l2)
        e.append(l1)
        e.append(l2)


   # loop for n steps
    for i in range(0, n): 
         # random index from (0, 2d>
          
        r = random.randint(0, 2*d - 1)
 
        # add e[r] to p[i]    
        p[0] = p[0] + e[r]
        

    l = calculate_length(p[0], d) 
    
    return l 

    
# max - number of simulations
def make_histogram(n, d, len, max):
    l = np.zeros(max, dtype=object)
    for i in range(max):
        l[i] = calculate_steps(n, d, len)

    return l


N = 200
d = 1
l = make_histogram(N, d, 1, 100000)

b = 20
n, bins, patches = plt.hist(l, b, histtype='bar')
plt.close()
n = n/len(l)
n = np.append(n, 0)
print(sum(n))

bins = np.array(bins, dtype=int)


leng = len(bins)                          #the number of data
mean = sum(bins*n)/leng                 #note this correction
sigma = sum(n*(bins-mean)**2)/leng        #note this correction


popt,pcov = curve_fit(gaus,bins,n,p0=[1,mean,sigma])
x =  np.arange(round(min(bins)), round(max(bins)), 0.1)
plt.scatter(bins,n,label='data')
plt.plot(x,gaus(x,*popt),'r-',label='fit')
plt.legend()
plt.title('d = ' + str(d) + ', ' + 'N = ' + str(N))
plt.xlabel('Distance')
plt.ylabel('P')
plt.show()