import math

def D(d):
	num = 2**d * math.factorial(d/2)
	den = math.pi**(d/2)
	frac = num/den
	return frac**(1./d)


rg = range(2,21,2)
d = [D(i) for i in rg]

# import matplotlib.pyplot as plt
# plt.scatter(d, rg)