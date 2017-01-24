import numpy as np
import matplotlib.pyplot as plt

noOfClusters = 3

x_coordinates = np.random.randint(1,100,20)
y_coordinates = np.random.randint(1,100,20)

plt.scatter(x_coordinates, y_coordinates)
plt.show()

points = []

for x,y in zip(x_coordinates, y_coordinates):
	points.append((x,y))

points = np.array(points)

randomIndices = np.random.random_integers(1,20,noOfClusters)

centroids = list(points[randomIndices])
tags = [ i for i in range(len(points)) ]
sumX = [0]*noOfClusters
sumY = [0]*noOfClusters

for i in range(10):
	sumX = [0]*noOfClusters
	sumY = [0]*noOfClusters
	sizeOfClusters = [0]*noOfClusters
	distanceFromCentroids = [ i for i in range(noOfClusters) ] 
	for j in range(len(points)):
		minDistance = 500
		clusterNumber = 0
		for i in range(noOfClusters):
			d_x = abs(centroids[i][0] - points[j][0])
			d_y = abs(centroids[i][1] - points[j][1])
			distanceFromCentroids[i] = np.sqrt(d_x**2 + d_y**2)
			if(distanceFromCentroids[i] < minDistance):
				minDistance = distanceFromCentroids[i]
				clusterNumber = i
				tags[j] = clusterNumber
	
	for i in range(len(points)):
		sumX[tags[i]] += points[i][0]
		sumY[tags[i]] += points[i][1]
		sizeOfClusters[tags[i]] += 1

	for i in range(noOfClusters):
		if sizeOfClusters[i] is not 0:
			centroids[i] = (float(sumX[i])/sizeOfClusters[i], float(sumY[i])/sizeOfClusters[i])
		else:
			centroids[i] = (0, 0)

print centroids 
		


