import random
import numpy as np
from causallearn.search.HiddenCausal.GIN.GIN import GIN
import pdb

sample_size = 500
random.seed(0)
np.random.seed(0)


L1 = np.random.uniform(-1, 1, size=sample_size)
L2 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(-1, 1, size=sample_size)
L3 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(1.2, 1.8) * L2 + np.random.uniform(-1, 1, size=sample_size)
X1 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X2 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X3 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X4 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X5 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X6 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X7 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X8 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X9 = np.random.uniform(1.2, 1.8) * L3 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
data = np.array([X1, X2, X3, X4, X5, X6, X7, X8, X9]).T
G, K = GIN(data)
pdb.set_trace()