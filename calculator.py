import math
delta = 2/5
rhalo = 1/(1/(1/2-delta)-1)

n = 5000000000

x = (1-rhalo)*(n-1)/2- (1+rhalo)*math.floor(n*delta)
print(x)