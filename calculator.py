# import math
# delta = 2/5
# rhalo = 1/(1/(1/2-delta)-1)

# n = 5000000000

# x = (1-rhalo)*(n-1)/2- (1+rhalo)*math.floor(n*delta)
# print(x)

import numpy as np
n=19
pi = np.pi
zetas = np.exp([1j*k*pi*2/n for k in range(n)])

def sigma(zetak, n):
    lst = [zetak**i for i in range(1,(n-1)//2+1)]
    return np.sum(lst)

def sigma2(zetak, n):
    return (zetak**((n+1)/2)-zetak)/(zetak-1)

for k in range(n):
    print(sigma(zetas[k], n))
    print(sigma2(zetas[k], n))
    print()