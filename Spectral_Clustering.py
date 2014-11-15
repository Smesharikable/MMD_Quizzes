import os
import random
import numpy as np
from numpy.linalg import linalg

__author__ = 'Ilya'
"""
Find two clusters in graph with given
Laplacian matrix using spectral clustering.
"""


# Initialize Laplacian matrix.
laplacian = np.array([[ 2, -1, -1,  0,  0,  0],
                      [-1,  3,  0, -1,  0, -1],
                      [-1,  0,  2, -1,  0,  0],
                      [ 0, -1, -1,  3, -1,  0],
                      [ 0,  0,  0, -1,  2, -1],
                      [ 0, -1,  0,  0, -1,  2]])

# Find eigenvalues and eigenvectors.
e_values, e_vectors = linalg.eigh(laplacian)

# Find the second smallest eigenvalue and corresponding eigenvector.
idx = e_values.argsort()
second_e_val = e_values[idx[2]]
second_e_vec = e_vectors[idx[2]]

# Cluster graph into two parts by splitting second eigenvector at the mean value.
threshold = np.mean(second_e_vec)
first_cluster = []
second_cluster = []
for i in range(0, len(second_e_vec)):
    v = second_e_vec[i]
    # Choose cluster at random if point is a tie.
    if np.allclose(v, threshold, rtol=0):
        if random.randint(0, 1):
            first_cluster.append(i)
        else:
            second_cluster.append(i)
    elif v < threshold:
        first_cluster.append(i)
    else:
        second_cluster.append(i)

np.set_printoptions(precision=4, linewidth=100)
print("Eigenvalues:")
print(e_values, os.linesep)
print("Eigenvectors:")
print(e_vectors, os.linesep)

print("First cluster:", first_cluster)
print("Second cluster:", second_cluster)