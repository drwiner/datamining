import numpy as np
from itertools import product
# import collections

# NamedPoint = collections.namedtuple('NamedPoint', ['id', 'point'])

class NamedPoint:
	def __init__(self, int_id, point):
		self.int_id = int_id
		self.point = point
		self.single_link_sim = dict()
		self.complete_link_sim = dict()
		self.mean_link_sim = dict()

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


c1 = open('C1.txt')
points = set()
for line in c1:
	split_line = line.split()
	p = np.array([float(i) for i in split_line[1:]])
	points.add(NamedPoint(split_line[0], p))

for point in points:
	for p in points:
		pass
		# single_link, complete_link, and mean_link calculations

	sims = []

	# sim_values[point.int_id] =

# Step 2 - create hierarchical methods (3)

