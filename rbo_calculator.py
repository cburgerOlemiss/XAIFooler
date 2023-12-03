# -*- coding: utf-8 -*-
import math
import numpy as np

def p_generator(p,d):
    def sum_series(p, d):
       # tail recursive helper function
       def helper(ret, p, d, i):
           term = math.pow(p, i)/i
           if d == i:
               return ret + term
           return helper(ret + term, p, d, i+1)
       return helper(0, p, d, 1)
    
    return  1 - math.pow(p, d-1) + (((1-p)/p) * d *(np.log(1/(1-p)) - sum_series(p, d-1)))

top_n_mass = 1.0 #What percentage of mass we wish to have on the top n features.
n = 2 #Number of top features
for i in range(1,100,1):
    p = i/100
    output = p_generator(p,n)
    if abs(output-top_n_mass) < 0.01:
        print("Set rbo_p = ",p, " for ", output*100, "% mass to be assigned to the top ", n, " features." )
        break