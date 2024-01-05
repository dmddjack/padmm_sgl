from static_signed import learn_a_static_signed_graph
import pandas as pd
import numpy as np
from time import time
import sys, getopt

if __name__ == "__main__":
    seed = 0
    density = 0
    opts, args = getopt.getopt(sys.argv[1:], "s:")
    for opt, arg in opts:
        if opt == '-s':
            seed = int(arg)
        else:
            raise IOError
    # print(seed)

    df = pd.read_csv("data/X_s_{}.csv".format(seed), header=None)
    X_noisy = np.array(df)
    DIM, NUM = X_noisy.shape

    _, _, params = learn_a_static_signed_graph(X_noisy, 0.1, 0.1)

    tic = time()
    w_pos, w_neg, params = learn_a_static_signed_graph(X_noisy, 0.1, 0.1, **params)
    
    # print(toc)
    w = np.array(w_pos) - np.array(w_neg)
    # print(w[0].shape)
    # print(w.shape)
    W = np.zeros((DIM, DIM))
    upper = np.triu_indices_from(W, k=1)
    W[upper] = np.squeeze(w)
    W = W + W.T
    toc = time() - tic
    print(toc)

    df = pd.DataFrame(W)
    df.to_csv('data/W_SGL_{}.csv'.format(seed), index=False, header=False)
