import numpy as np

A = np.asarray([[0, 0, 0], [1, 0, 1], [1, 0, 0]])
cap = np.asarray([0, 1, 0])
weighted_sum = A.T @ cap