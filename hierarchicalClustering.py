import numpy as np
from itertools import product
import math
# import collections

# NamedPoint = collections.namedtuple('NamedPoint', ['id', 'point'])

class NamedPoint:
	def __init__(self, int_id, point):
		self.int_id = int_id
		self.point = point

class Cluster:
	def __init__(self, points):
		self.points = points
		# self.single_link_sim = dict()
		# self.complete_link_sim = dict()
		# self.mean_link_sim = dict()
	def absorb(self, cluster):
		self.points.update(cluster.points)

	def __len__(self):
		return len(self.points)

def dotDistance(s1, s2):
	return np.dot(s1.point - s2.point, s1.point - s2.point)

def singleLink(S1, S2):
	# S1 and S2 are clusters, possibly with just 1 entity
	# each entity has a point
	S_prod = set(product(S1,S2))
	return min(dotDistance(s1,s2) for s1,s2 in S_prod)

def completeLink(S1, S2):
	S_prod = set(product(S1, S2))
	return min(dotDistance(s1,s2) for s1,s2 in S_prod)

def meanLink(S1, S2):
	a1 = (1/len(S1))*sum(s.point for s in S1)
	a2 = (1/len(S2))*sum(s.point for s in S2)
	return dotDistance(a1, a2)

# initial loading
c1 = open('C1.txt')
points = set()
for line in c1:
	split_line = line.split()
	p = np.array([float(i) for i in split_line[1:]])
	points.add(NamedPoint(split_line[0], p))

# algorithm

# initialization:
singleLink_clusters = set()
completeLink_clusters = set()
meanLink_clusters = set()
for point in points:
	singleLink_clusters.add(Cluster(point))
	completeLink_clusters.add(Cluster(point))
	meanLink_clusters.add(Cluster(point))

def h_clustering(clusters, k, dist_method):
	clusts = set(clusters)
	while len(clusts) > k:
		pairwise_clusters = set(product(clusts, clusts))
		arg_mins = None
		m = float("inf")
		for c1, c2 in pairwise_clusters:
			value = dist_method(c1, c2)
			if value < m:
				m = value
				arg_mins = (c1, c2)
		if arg_mins is None:
			print('wtf')
		c1, c2 = arg_mins
		if len(c1) < len(c2):
			c2.absorb(c1)
			clusts -= c1
		else:
			c1.absorb(c2)
			clusts -= c2
	return clusts

sl_clusts = h_clustering(singleLink_clusters, 4, singleLink)
cl_clusts = h_clustering(completeLink_clusters, 4, completeLink)
ml_clusts = h_clustering(meanLink_clusters, 4, meanLink)
