import math

    # compute RBO
def RBO(list1,list2,p):
	comparisonLength = min(len(list1),len(list2))
	set1 = set()
	set2 = set()
	summation = 0
	for i in range(comparisonLength):
		set1.add(list1[i])
		set2.add(list2[i])            
		summation += math.pow(p,i+1) * (len(set1&set2) / (i+1))
	return ((len(set(list1)&set(list2))/comparisonLength) * math.pow(p,comparisonLength)) + (((1-p)/p) * summation)


from scipy import stats

def SM(list1, list2):
	coef, p = stats.spearmanr(list1, list2)
	return 1 - max(0, coef)

list1 = [1,2,3]
list2 = [2,3,1]
print(RBO(list1, list2, p=0.9))
print(RBO(list1, list2, p=1.0))
print(SM(list1, list2))

