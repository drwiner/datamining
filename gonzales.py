import numpy as np
from hierarchicalClustering import Cluster, NamedPoint

def dist(a1, a2):
	return np.dot(a1 - a2, a1 - a2)

def gonzales(X, C, start_c):
	n = len(X)
	k = len(C)
	phi = np.ones(n)
	c_i = start_c
	for i in range(2, k):
		m = 0
		# c_i = X[0]
		for j in range(1, n):
			d = dist(X[j], phi[j])
			if d > m:
				m = d
				c_i = X[j]
		for j in range(1, n):
			if dist(X[j], phi[j]) > dist(X[j], c_i):
				phi[j] = i
	return phi

# read C2.text
# x = [], C = []
C2 = open('C2.text')
points = set()
for line in C2:
	split_line = line.split()
	p = np.array([float(i) for i in split_line[1:]])
	points.add(NamedPoint(int(split_line[0]), p))