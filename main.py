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


N1 = 200
d = 2
maxx = 100000
l1 = make_histogram(N1, d, 1, maxx)
print("1")

N2 = 100
l2 = make_histogram(N2, d, 1, maxx)
print("2")

N3 = 20
l3 = make_histogram(N3, d, 1, maxx)
print("3")



b = 20
n1, bins1, patches1 = plt.hist(l1, b, histtype='bar')
n2, bins2, patches2 = plt.hist(l2, b, histtype='bar')
n3, bins3, patches3 = plt.hist(l3, b, histtype='bar')

plt.close()
n1 = n1/len(l1)
n1 = np.append(n1, 0)
n2 = n2/len(l2)
n2 = np.append(n2, 0)
n3 = n3/len(l3)
n3 = np.append(n3, 0)

print(sum(n1))
print(sum(n2))
print(sum(n3))


bins1 = np.array(bins1, dtype=int)
bins2 = np.array(bins2, dtype=int)
bins3 = np.array(bins3, dtype=int)



leng1 = len(bins1)                          
mean1 = sum(bins1*n1)/leng1                 
sigma1 = sum(n1*(bins1-mean1)**2)/leng1        

leng2 = len(bins2)                          
mean2 = sum(bins2*n2)/leng2              
sigma2 = sum(n2*(bins2-mean2)**2)/leng2 

leng3 = len(bins3)                          
mean3 = sum(bins3*n3)/leng3                 
sigma3 = sum(n3*(bins3-mean3)**2)/leng3 



popt1,pcov1 = curve_fit(gaus,bins1,n1,p0=[1,mean1,sigma1])
x1 =  np.arange(round(min(bins1)), round(max(bins1)), 0.1)

popt2,pcov2 = curve_fit(gaus,bins2,n2,p0=[1,mean2,sigma2])
x2 =  np.arange(round(min(bins2)), round(max(bins2)), 0.1)

popt3,pcov3 = curve_fit(gaus,bins3,n3,p0=[1,mean3,sigma3])
x3 =  np.arange(round(min(bins3)), round(max(bins3)), 0.1)



plt.scatter(bins1,n1, color='red', label="N = " + str(N1))
plt.scatter(bins2,n2, color='green', label="N = " + str(N2))
plt.scatter(bins3,n3,color='blue', label="N = " + str(N3))


plt.plot(x1,gaus(x1,*popt1),'r-')
plt.plot(x2,gaus(x2,*popt2),'g-')
plt.plot(x3,gaus(x3,*popt3),'b-')


plt.legend()
plt.title('d = ' + str(d))
plt.xlabel('Distance')
plt.ylabel('P')
plt.show()