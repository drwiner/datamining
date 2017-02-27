import numpy as np
from hierarchicalClustering import Cluster, NamedPoint
import collections

def dist(a1, a2):
	return np.dot(a1 - a2, a1 - a2)
def dotDistance(s1, s2):
	return np.dot(s1.point - s2.point, s1.point - s2.point)

def Gonzales(X, k):
	n = len(X)
	# c_i = X[0]
	C = collections.defaultdict()
	C[0] = X[0]
	# C = dict({0 : X[0]})
	phi = [0 for i in range(n)]
	for i in range(1, k):
		m = 0
		C[i] = X[0]
		# find the x which is farthest from the center current mapped to x
		#  = X[0]
		for j in range(n):
			# for each x, find the one which is maximum distance from its center, save the i it occurs
			d = dotDistance(X[j], C[phi[j]])
			if d > m:
				m = d
				# this is the worst center so far
				C[i] = X[j]
		for j in range(n):
			if dotDistance(X[j], C[phi[j]]) > dotDistance(X[j], C[i]):
				# update each mapping to the closest center.
				phi[j] = i
	return C, phi

def kMeans(X, k):
	n = len(X)
	C = collections.defaultdict()
	phi = [0 for i in range(n)]
	C[0] = X[0]
	for i in range(1, k):
		# C[i] = X[0]
		choices = [(x, dotDistance(x, C[i-1])**2) for x in X]
		x = weightedChoice(choices)
		C[i] = x
		for j in range(n):
			if dotDistance(X[j], C[phi[j]]) > dotDistance(X[j], C[i]):
				phi[j] = i
	return C, phi

import random

def weightedChoice(choices):
	total = sum(w for c, w in choices)
	r = random.uniform(0, total)
	upto = 0
	for c, w in choices:
		if upto + w >= r:
			return c
		upto += w
	assert False, "shouldn't get here"



# read C2.text
# x = [], C = []
C2 = open('C2.txt')
x = set()
for line in C2:
	split_line = line.split()
	p = np.array([float(i) for i in split_line[1:]])
	x.add(NamedPoint(int(split_line[0]), p))

k = 3
x_ref = list(x)
x = list(x_ref)
import math
C, phi = Gonzales(x, k)
cost_max_3centers = max(dotDistance(elm, C[phi[i]]) for i, elm in enumerate(x))

def threeMeansCost(x, C, phi):
	sig = sum(dotDistance(elm, C[phi[i]])**2 for i, elm in enumerate(x))
	cost_3means = math.sqrt((1/len(x)) * sig)
	return cost_3means

print('gonzales, max 3centers: ' + str(cost_max_3centers))
print('gonzales, cost 3 means: ' + str(threeMeansCost(x,C, phi)))

def cuDensity(tmc):
	cu = 0
	cu_frac = []
	# stmc = sorted(tmc)
	s = sum(tmc)
	for t in tmc:
		cu += t / s
		cu_frac.append(cu)
	return cu_frac

def cuDensityPlot(cu_list, show=True):
	plt.plot(list(range(len(cu_list))), cu_list,  'ro')
	if show:
		plt.show()

# 3 center cost max x in X d(x, phi[x])^2

# C is the centers, not clusters
# phi is the labeling/mapping
# Clusters = [Cluster(i,  for i, c in C)]
print('here')

import matplotlib.pyplot as plt

colours = ['r', 'g', 'b', 'y']


########################
########## K Means ++ ##
########################
# tmc = []
# for i in range(10):
# 	C_kmeans, phi_kmeans = kMeans(x, k)
# 	if C_kmeans == C or phi_kmeans == phi:
# 		print('this time')
# 	tmc_value = threeMeansCost(x, C_kmeans, phi_kmeans)
# 	tmc.append(tmc_value)

from collections import Counter

print('check')

def closestToPhi(C, X, closest_centers):
	phi = [0 for i in range(len(X))]
	for i, x in enumerate(X):
		phi[i] = C.index(closest_centers[x])
	return phi

def Lloyds(X, C, k):
	# arbitrary choice
	changed = True
	closest_centers = collections.defaultdict(int)
	while changed:
		changed = False
		for x in X:
			# find nearest center
			for i, c in enumerate(C):
				closest = C[closest_centers[x]]
				if dotDistance(x, c) < dotDistance(x, closest):
					closest_centers[x] = i
					changed = True
		for i, c in enumerate(C):
			subset = [x.point for x in X if closest_centers[x] == i]
			s = sum(subset)
			if len(subset) == 0:
				print('something went wrong')
				break
			avg = s / len(subset)
			n = NamedPoint('c' + str(i), avg)
			if C[i] != n:
				# changed = True
				C[i] = n
	return C, closest_centers


# problem a
k = 3
# Clusters = x[0:k]
# Clusters, closest_centers_map = Lloyds(x, Clusters, k)
# phi = closestToPhi(Clusters, x, closest_centers_map)
print('dude')

colours = ['r', 'g', 'b', 'y']
# s1 = list(Clusters)
# for i in range(k):
# 	rel_points = [p for p in x if closest_centers_map[p] == i]
# 	xes = [p.point[0] for p in rel_points]
# 	yes = [p.point[1] for p in rel_points]
# 	plt.scatter(xes, yes, c=colours[i])
# 	plt.scatter(Clusters[i].point[0], Clusters[i].point[1], c='y')

# plt.show()

# for p in Clusters:
# 	print(p.point)
# lloyds_a_3means = threeMeansCost(x, Clusters, closest_centers_map)
# print(lloyds_a_3means)

# C is the gonzales output before
# Clusters, closest_centers_map = Lloyds(x, list(C.values()), k)
# colours = ['r', 'g', 'b', 'y']
# s1 = list(Clusters)
# for i in range(k):
# 	rel_points = [p for p in x if closest_centers_map[p] == i]
# 	xes = [p.point[0] for p in rel_points]
# 	yes = [p.point[1] for p in rel_points]
# 	plt.scatter(xes, yes, c=colours[i])
# 	plt.scatter(Clusters[i].point[0], Clusters[i].point[1], c='y')
#
# plt.show()
#
# for p in Clusters:
# 	print(p.point)
# lloyds_a_3means = threeMeansCost(x, Clusters, closest_centers_map)
# print(lloyds_a_3means)

tmc = []
sames = 0
d = 40
z = range(d)
for i in z:
	C_kmeans, phi_kmeans = kMeans(x, k)
	clusters, phi = Lloyds(x, list(C_kmeans.values()), k)
	if clusters == C_kmeans:
		sames += 1
	tmc_value = threeMeansCost(x, clusters, phi)
	tmc.append(tmc_value)
	# for i in range(k):
	# 	rel_points = [p for p in x if phi[p] == i]
	# 	xes = [p.point[0] for p in rel_points]
	# 	yes = [p.point[1] for p in rel_points]
	# 	plt.scatter(xes, yes, c=colours[i])
	# 	plt.scatter(clusters[i].point[0], clusters[i].point[1], c='y')
	# plt.show()
print(tmc)
print('same time: ' + str(sames/d))

from collections import Counter

TC = Counter([math.floor(t) for t in tmc])

cu_dict = dict()
cu = 0
for i in range(max(TC.keys())):
	cu += TC[i]/d
	cu_dict.update({i: cu})
plt.scatter(list(cu_dict.keys()), list(cu_dict.values()))
plt.show()
