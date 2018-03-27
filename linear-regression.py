import sys
import numpy as np
# import matplotlib.pyplot as plt
from itertools import product

class LinearRegression():

	def __init__(self, alpha = 0.01, maxiter = 100, datafile = 'input2.csv'):
		data = np.genfromtxt(datafile, delimiter = ',')
		self.x = self.augment(data[:, :2])
		self.y = data[:, 2]
		self.w = np.zeros(3)
		self.alpha = alpha
		self.maxiter = maxiter
		self.normalize()

	def augment(self, x):
		aug_x = np.ones((x.shape[0], x.shape[1] + 1))
		aug_x[:, 1:] = x
		return aug_x

	def normalize(self):
		self.x[:, 1:] = (self.x[:, 1:] - self.x[:, 1:].mean(axis = 0))/self.x[:, 1:].std(axis = 0)

	def train(self):
		n = len(self.x)
		for ei in range(self.maxiter):
			pred = np.array([self.predict(x) for x in self.x])
			error = np.sum((pred - self.y) ** 2) / (2 * n)
			delta = np.dot((pred - self.y), self.x) * self.alpha / n
			self.w -= delta

	def predict(self, x):
		y = np.dot(self.w, np.array(x))
		return y

	def datavis(self):
		fig = plt.figure()
		graph = Axes3D(fig)
		graph.scatter(self.x[:,1], self.x[:,2], self.y)
		plt.show()
		
def main():	
	alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.02]
	maxiter = 100
	file = open(sys.argv[2], 'w')
	for a in alpha:
		lr = LinearRegression(a, maxiter, sys.argv[1])
		lr.train()
		file.write(str(a) + ',' + str(maxiter) + ',' + str(lr.w[0]) + ',' + str(lr.w[1]) + ',' + str(lr.w[2]) + '\n')
	file.close()


if __name__ == "__main__":
	main()