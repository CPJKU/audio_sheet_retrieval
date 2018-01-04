
import numpy as np

if __name__ == '__main__':

    idxs = np.arange(0, 1000)
    n_idxs = len(idxs)

    visited = np.zeros_like(idxs, dtype=np.bool)

    first_idx = np.random.randint(0, n_idxs)